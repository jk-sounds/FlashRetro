import torch
from flashretro.chem.utils import result2mol, get_canonical_smiles
from collections import Counter
import numpy as np
import traceback

def canonicalize_smiles(smi):

    if smi is None:
        return None
    try:
        canon_s = get_canonical_smiles(smi)
        if canon_s is not None:
            return canon_s
    except:
        pass
    return smi

def get_unique_smiles(candidate_smiles):

    if not candidate_smiles:
        return []
        
    smiles_counts = Counter(candidate_smiles)
    
    unique_smiles = list(dict.fromkeys(candidate_smiles))

    sorted_candidates = sorted(unique_smiles, key=lambda s: smiles_counts[s], reverse=True)
    
    return sorted_candidates

def evaluate_single_sample(args):
    """
    Args:
        args: (sample_dict, mode, topk)
    """
    torch.set_num_threads(1)

    sample, mode, topk = args
    sample_num = len(sample["src_masks"])
    

    true_elements = sample["true_elements"][0]
    tgt_masks = sample["tgt_masks"][0]
    true_bonds = sample["true_bonds"][0]
    true_aromas = sample["true_aromas"][0]
    true_charges = sample["true_charges"][0]
    src_flags = sample["src_flags"][0] 
    

    _, tgt_s, _ = result2mol(
        (true_elements, tgt_masks, true_bonds, true_aromas, true_charges, src_flags)
    )
    

    if tgt_s is None:
        return None, None
    

    tgt_s = canonicalize_smiles(tgt_s)

    candidate_smiles = []
    
    for i in range(sample_num):
        pred_bonds = sample["pred_bonds"][i]
        pred_aromas = sample["pred_aromas"][i]
        pred_charges = sample["pred_charges"][i]
        src_flags_i = sample["src_flags"][i]
        src_masks_i = sample["src_masks"][i]
        pred_elements = sample["pred_elements"][i]
            
        _, pred_s, _ = result2mol(
            (pred_elements, src_masks_i, pred_bonds, pred_aromas, pred_charges, src_flags_i)
        )
        
        if pred_s is None:
            continue
        canon_s = canonicalize_smiles(pred_s)
        candidate_smiles.append(canon_s)

    sorted_candidates = get_unique_smiles(candidate_smiles)
            
    acc_smiles = []
    valid_smiles = [] 
    
    eval_limit = min(len(sorted_candidates), topk) 
    
    if eval_limit == 0:
        return [0.0] * topk, [0.0] * topk

    for i in range(eval_limit):
        pred_s = sorted_candidates[i]

        valid_smiles.append(1.0)

        is_match = False
        # Only retro logic
        tgt_split = sorted(tgt_s.split("."))
        pred_split = sorted(pred_s.split("."))
        if tgt_split == pred_split:
            is_match = True
        
        acc_smiles.append(1.0 if is_match else 0.0)
    

    remaining = topk - len(acc_smiles)
    if remaining > 0:
        acc_smiles.extend([0.0] * remaining)
        valid_smiles.extend([0.0] * remaining)  
        
    del sample
    return acc_smiles, valid_smiles

def process_and_save_smiles(
    batch_idx, sample_idx, k, mode, 
    sample_dict, src_graph_for_smiles, 
    smiles_file
):

    product_s = None
    if mode == 'retro' and src_graph_for_smiles is not None:
        try:
            product_elements = src_graph_for_smiles['elements'][sample_idx]
            product_masks = src_graph_for_smiles['masks'][sample_idx]
            product_bonds = src_graph_for_smiles['bonds'][sample_idx]
            product_aromas = src_graph_for_smiles['aromas'][sample_idx]
            product_charges = src_graph_for_smiles['charges'][sample_idx]

            product_flags = (~product_masks).long()
            
            _, product_s, _ = result2mol(
                (product_elements, product_masks, product_bonds, 
                 product_aromas, product_charges, product_flags)
            )

            product_s = canonicalize_smiles(product_s)
            
        except Exception as e:
            print(f"Warning: Failed to extract product SMILES for batch {batch_idx}, sample {sample_idx}: {e}")
            traceback.print_exc()
            product_s = None
    

    true_elements = sample_dict["true_elements"][0]
    tgt_masks = sample_dict["tgt_masks"][0]
    true_bonds = sample_dict["true_bonds"][0]
    true_aromas = sample_dict["true_aromas"][0]
    true_charges = sample_dict["true_charges"][0]
    src_flags_true = sample_dict["src_flags"][0]
    
    _, tgt_s, _ = result2mol(
        (true_elements, tgt_masks, true_bonds, true_aromas, true_charges, src_flags_true)
    )
    
    if tgt_s is not None:

        tgt_s = canonicalize_smiles(tgt_s)

        candidate_smiles = []
        for k_idx in range(k):
            pred_bonds = sample_dict["pred_bonds"][k_idx]
            pred_aromas = sample_dict["pred_aromas"][k_idx]
            pred_charges = sample_dict["pred_charges"][k_idx]
            pred_elements = sample_dict["pred_elements"][k_idx]
            src_masks_k = sample_dict["src_masks"][k_idx]
            src_flags_k = sample_dict["src_flags"][k_idx]
            
            _, pred_s, _ = result2mol(
                (pred_elements, src_masks_k, pred_bonds, pred_aromas, pred_charges, src_flags_k)
            )

            if pred_s is None:
                continue

            pred_s = canonicalize_smiles(pred_s)
            
            candidate_smiles.append(pred_s)

        sorted_candidates = get_unique_smiles(candidate_smiles)

        if sorted_candidates and smiles_file is not None:
            for rank_idx, pred_s in enumerate(sorted_candidates[:10]):
                smiles_file.write(
                    f"{batch_idx},{sample_idx},{rank_idx},{mode},{product_s or ''},{tgt_s},{pred_s}\n"
                )
