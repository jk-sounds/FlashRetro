import os
import torch
import click
from omegaconf import OmegaConf
from tqdm import tqdm
from flashretro.data.deltagraph_dataset import DeltaGraphDataModule
from flashretro.models.wrapper import flashretroWrapper
from flashretro.models.diffusion.utils import prepare_source_graph, prepare_ground_truth, transform_to_adjacency
import torch.nn.functional as F
from flashretro.chem.utils import result2mol, get_canonical_smiles
import numpy as np
from multiprocessing import get_context, cpu_count
from flashretro.utils.eval_helpers import evaluate_single_sample, canonicalize_smiles, get_unique_smiles, process_and_save_smiles

@click.command()
@click.option("--ckpt_path", type=str, required=True)
@click.option("--config_path", type=str, required=True)
@click.option("--batch_size", type=int, default=4)
@click.option("--k", type=int, default=10) 
@click.option("--device", type=str, default="cuda")
@click.option("--num_workers_eval", type=int, default=8)
@click.option("--save_smiles", type=bool, default=False)
def main(ckpt_path, config_path, batch_size, k, device, num_workers_eval, save_smiles):
    # Load config
    config = OmegaConf.load(config_path)
    
    # Load model
    print(f"Loading model from {ckpt_path}...")
    model = flashretroWrapper.load_from_checkpoint(
        ckpt_path,
        config=config,
        strict=False,
    )
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # Load data
    print("Loading data...")
    datamodule = DeltaGraphDataModule(
        config,
        batch_size=batch_size,
        num_workers=4, 
        **config.data,
    )
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    
    print(f"Testing on full test set with Sampling Budget num={k}...")
    
    mode = config.train.mode if hasattr(config.train, 'mode') else 'retro'
    
    
    k_list = [1, 3, 5, 10] 
    k_list = [x for x in k_list if x <= k]
    eval_topk = max(k_list) if k_list else k
    
    
    all_accuracies = []
    all_validities = []
    total_samples = 0
    
    
    smiles_file = None
    if save_smiles:
        save_dir = os.path.dirname(ckpt_path)
        smiles_path = os.path.join(save_dir, f"sampling_smiles_{mode}_kk{k}.csv")
        smiles_file = open(smiles_path, "w")
        smiles_file.write("batch_id,sample_id,k_idx,mode,product_smiles,true_reactant_smiles,pred_reactant_smiles\n")
        print(f"SMILES will be saved to: {smiles_path}")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Sampling & Evaluating")):
            batch = {
                key: val.to(device) if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
            
            src_graph = prepare_source_graph(batch, mode=mode)
            ground_truth = prepare_ground_truth(batch, mode=mode)
            
            # Store source graph elements for product SMILES extraction (if saving SMILES)
            src_graph_for_smiles = None
            if save_smiles and smiles_file is not None and mode == 'retro':
                # For retro mode, we need product data for SMILES extraction
                # Convert product bonds from neighbor indices to adjacency matrix
                product_bond_masks = torch.einsum("bl,bk->blk", ~batch['product_masks'], ~batch['product_masks'])
                product_bonds_adjacency = transform_to_adjacency(
                    batch['product_bonds'],
                    product_bond_masks,
                    remove_self_loops=True
                ).cpu()
                
                # Convert aromatic bonds (type 4) to single bonds (type 1) for result2mol compatibility
                # result2mol will use aroma attribute to determine aromaticity
                product_bonds_adjacency[product_bonds_adjacency == 4] = 1
                
                src_graph_for_smiles = {
                    'elements': batch['elements'].cpu(),
                    'masks': batch['product_masks'].cpu(),
                    'bonds': product_bonds_adjacency,
                    'aromas': batch['product_aromas'].cpu(),
                    'charges': batch['product_charges'].cpu(),
                }
            
            # Expand src_graph for meanflow_sample if k > 1
            if k > 1:
                src_graph_expanded = {}
                for key, val in src_graph.items():
                    if isinstance(val, torch.Tensor):
                        src_graph_expanded[key] = val.repeat_interleave(k, dim=0)
                    else:
                        src_graph_expanded[key] = val
                src_graph = src_graph_expanded

            # Run sampling
            results = model.diffusion.meanflow_sample(src_graph, mode=mode)
            
            # Process results
            B = batch['elements'].shape[0]
            
            # Move results to CPU once
            for key in results:
                if isinstance(results[key], torch.Tensor):
                    results[key] = results[key].cpu()
            
           
            ground_truth_cpu = {
                key: val.cpu() if isinstance(val, torch.Tensor) else val
                for key, val in ground_truth.items()
            }
            
           
            for b in range(B):
                start = b * k
                end = (b + 1) * k
                
                
                src_flags_val = results['flags'][start:end]
                    
                    
                sample_dict = {
                    'pred_bonds': results['bonds'][start:end],
                    'pred_aromas': results['aromas'][start:end],
                    'pred_charges': results['charges'][start:end],
                    'pred_elements': results['elements'][start:end], 
                    # 'elements': results['elements'][start:end],
                    'src_masks': results['masks'][start:end],
                    'src_flags': src_flags_val,
                    
                # True values (replicated K times)
                'true_elements': ground_truth_cpu['elements'][b].unsqueeze(0).repeat(k, 1),
                'true_bonds': ground_truth_cpu['bonds'][b].unsqueeze(0).repeat(k, 1, 1),
                'true_aromas': ground_truth_cpu['aromas'][b].unsqueeze(0).repeat(k, 1),
                'true_charges': ground_truth_cpu['charges'][b].unsqueeze(0).repeat(k, 1), 
                'tgt_masks': ground_truth_cpu['masks'][b].unsqueeze(0).repeat(k, 1),
                }
                    
                
                if save_smiles and smiles_file is not None:
                    process_and_save_smiles(
                        i, b, k, mode, 
                        sample_dict, src_graph_for_smiles, 
                        smiles_file
                    )
                
              
                acc, valid = evaluate_single_sample((sample_dict, mode, eval_topk))
                
                if acc is not None:
                    all_accuracies.append(acc)
                    all_validities.append(valid)
                    total_samples += 1
                
                
                del sample_dict
            
           
            del results, ground_truth_cpu
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    
    if save_smiles and smiles_file is not None:
        smiles_file.close()
        print(f"SMILES saved to: {smiles_path}")
    
    print(f"\nEvaluation completed for {total_samples} samples.")
    
    if total_samples == 0:
        print("No valid results found.")
        return

    
    acc_all = np.array(all_accuracies)  
    valid_all = np.array(all_validities)  
    
    results_all = list(zip(all_accuracies, all_validities))
    
    print("\nResults:")
    save_dir = os.path.dirname(ckpt_path)
    result_file = os.path.join(save_dir, "topk_results.txt")
    
    with open(result_file, "w") as f:
        for k_val in k_list:
            if k_val > acc_all.shape[1]:
                continue

            acc_k = acc_all[:, :k_val]
            
            acc_k_clean = acc_k[~np.all(np.isnan(acc_k), axis=1)]
            if len(acc_k_clean) == 0:
                print(f"Top {k_val}: No valid samples.")
                continue
                
            hits = np.any(acc_k_clean == 1.0, axis=1).astype(int)
            accuracy = np.mean(hits)
            
            # Validity (Top-K validity)
            valid_k = valid_all[:, :k_val]
            validity = np.mean(valid_k)
            
            msg = f"Top {k_val} accuracy: {accuracy:.4f}, validity: {validity:.4f}"
            print(msg)
            f.write(msg + "\n")
    
    print(f"\nResults saved to {result_file}")

if __name__ == "__main__":
    main()
