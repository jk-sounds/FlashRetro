from torch import nn
import torch
import torch.nn.functional as F
from flashretro.data.deltagraph import DeltaGraphBatch
from flashretro.models.misc import shape_back, append_dims
from flashretro.models.transformers import (
    get_encoder, 
    get_variational_decoder,
    get_merge_encoder
)
from flashretro.models.transformers.layers import LatentVelocityNetwork
from .loss import MeanFlowLoss
from flashretro.chem.constants import element_id, get_element_id_batch, MAX_DIFF
from typing import Dict, Tuple
import torch.distributions as D
from tqdm import tqdm
import copy
from .utils import *

Code = torch.Tensor
ProbParam = torch.Tensor
Prob = D.Distribution
Order = torch.Tensor
Time = torch.Tensor

    
class VAEDIFMUniform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.flow_sampler = None
        self.interm_encoder = None

        self.reactant_encoder = get_encoder(config.mol_encoder)
        self.product_encoder = get_encoder(config.mol_encoder)
        self.merge_encoder = get_merge_encoder(config.merge_encoder)
        self.product_decoder = get_variational_decoder(config.vae_decoder)
        self.decoder_only = config.decoder_only
        self.force_symbond = config.get('force_symbond', True)

        # MeanFlow components
        assert config.get('meanflow', None) is not None, "必须使用 meanflow"
        self.velocity_network = LatentVelocityNetwork(
            d_model=config.mol_encoder.dim,
            nhead=config.mol_encoder.nhead,
            num_layers=config.meanflow.num_layers,
            dropout=config.mol_encoder.dropout
        )
        self.meanflow_loss = MeanFlowLoss(
            **config.meanflow.loss
        )
        self.use_meanflow = True

    
    def get_meanflow_loss(
            self,
            data: DeltaGraphBatch,
            mode: str = 'retro'
        ):
        """MeanFlow loss computation in latent space
        
        Following plan.md:
        1. Encode reactant/product to latent space (z0, z1)
        2. Apply MeanFlow loss in latent space
        3. Decode z1 to compute CE loss for discrete graph reconstruction
        """
        translate_graph = prepare_translate_graph(data, mode=mode)
        
        # Construct src and tgt graphs for encoder input
        # Note: charges are NOT +6 here because AtomEncoder adds +6 internally
        src = {
            'element_types': translate_graph['src_element_types'],
            'elements': translate_graph['src_elements'],
            'bonds': translate_graph['src_bonds'],
            'aromas': translate_graph['src_aromas'],
            'charges': translate_graph['src_charges'],
            'masks': translate_graph['src_masks'],
            'padding_masks': translate_graph['padding_masks'],
            'segments': translate_graph['src_segments'],
            'flags': translate_graph['src_flags']
        }
        
        tgt = {
            'element_types': translate_graph['tgt_element_types'],
            'elements': translate_graph['tgt_elements'],
            'bonds': translate_graph['tgt_bonds'],
            'aromas': translate_graph['tgt_aromas'],
            'charges': translate_graph['tgt_charges'],
            'masks': translate_graph['tgt_masks'],
            'padding_masks': translate_graph['padding_masks'],
            'segments': translate_graph['tgt_segments'],
            'flags': translate_graph['tgt_flags']
        }
        
        # Encode to latent space: z0 = src latent, z1 = tgt latent
        src_emb = self.product_encoder(
            src['elements'],
            src['bonds'],
            src['aromas'],
            src['charges'],
            src['padding_masks'],
            src['segments'],
            src['flags']
        )  # Condition embedding
        
        tgt_emb = self.reactant_encoder(
            tgt['elements'],
            tgt['bonds'],
            tgt['aromas'],
            tgt['charges'],
            tgt['padding_masks'],
            tgt['segments'],
            tgt['flags']
        )  # Target latent (z1)
        
        # Compute MeanFlow loss in latent space
        loss_mf = self.meanflow_loss(
            self.velocity_network,
            tgt_emb,  # z1 (target latent)
            src_emb,  # conditioning (source latent)
            mask=translate_graph['padding_masks'],
            task_token=translate_graph['task_token']
        )
        
        # Decode tgt_emb for CE loss (to ensure latent is meaningful)
        # Following plan.md 3.1: Use Reactant Graph (src) as structure template
        merge_emb = self.merge_encoder(
            translate_graph['task_token'], 
            tgt_emb + src_emb, 
            translate_graph['padding_masks']
        )
        pred = self.product_decoder(
            merge_emb if not self.decoder_only else tgt_emb,
            translate_graph['src_bonds'],  # Use src_bonds as template (not tgt_bonds)
            translate_graph['padding_masks'],
            decoder_only=self.decoder_only
        )
        
        # Compute CE loss
        loss_ce_dict = self.compute_loss(pred, translate_graph, mode=mode)
        
        # Combine losses
        loss_ce_dict['meanflow'] = loss_mf
        return loss_ce_dict
    
    def compute_loss(self, pred: Dict[str, torch.Tensor], data: TranslateGraph, mode='retro'):
        loss_bond = self.ce_bond_loss(
            pred['bond_outputs'], 
            data['tgt_bonds'], 
            data['tgt_masks'], 
            data['padding_masks']
        )
        loss_aroma = self.aroma_loss(
            pred['aroma_logits'], 
            data['tgt_aromas'], 
            data['tgt_masks']
        )
        loss_charge = self.charge_loss(
            pred['charge_logits'], 
            data['tgt_charges'], 
            data['tgt_masks']
        )
        loss_element = self.element_loss(
            pred['element_type_logits'], 
            data['tgt_element_types'], 
            data['tgt_masks']
        )

        return {
            'bond': loss_bond.mean(),
            'aroma': loss_aroma.mean(),
            'charge': loss_charge.mean(),
            'element': loss_element.mean(),
        }
    
    def sample(self, data, sample_steps=100, t_min=0.001, mode='retro'):
        src_graph = prepare_source_graph(data, mode=mode)
        if self.use_meanflow and sample_steps == 1:
            preds = self.meanflow_sample(src_graph, mode=mode)
        else:
            pass
        # else:
        #     preds = self.ode_sample(src_graph, sample_steps=sample_steps, t_min=t_min, mode=mode)
        return preds
    
    @torch.no_grad()
    def meanflow_sample(self, src_graph: OneSideGraph, mode='retro'):
        """1-NFE sampling using MeanFlow velocity network"""
        B = src_graph['padding_masks'].shape[0]
        
        # Task token
        if mode == 'retro':
            task_token = torch.zeros((B, 1), device=src_graph['elements'].device).long()
        else:
            task_token = torch.ones((B, 1), device=src_graph['elements'].device).long()
        
        # Encode source to latent space (pass 7 individual parameters)
        src_emb = self.product_encoder(
            src_graph['elements'],
            src_graph['bonds'],
            src_graph['aromas'],
            src_graph['charges'],
            src_graph['padding_masks'],
            src_graph['segments'],
            src_graph['flags']
        )
        
        # Sample noise as z_t (t=1)
        z_t = torch.randn_like(src_emb)
        
        # Predict velocity: u(z_t, r, t) where r=0, t=1
        t = torch.ones(B, device=src_emb.device)
        r = torch.zeros(B, device=src_emb.device)
        u = self.velocity_network(z_t, src_emb, t, r, src_mask=src_graph['padding_masks'])
        
        # One-step: z_0 = z_1 - u(z_1, 0, 1)
        z_0 = z_t - u
        
        # Merge z_0 with src_emb (following the same pattern as get_meanflow_loss)
        merge_emb = self.merge_encoder(
            task_token, 
            z_0 + src_emb, 
            src_graph['padding_masks']
        )
        
        # Decode from merged embedding
        # Use src_graph['bonds'] as template (structure basis)
        pred = self.product_decoder(
            merge_emb if not self.decoder_only else z_0,
            src_graph['bonds'],
            src_graph['padding_masks'],
            decoder_only=self.decoder_only
        )
        
        # Parse predictions (following ode_sample format for consistency)
        # bonds: convert to one-hot format (same as ode_sample line 568)
        pred_bonds = F.one_hot(pred['bond_outputs'].argmax(dim=-1).long(), num_classes=MAX_DIFF)
        pred_aromas = (torch.sigmoid(pred['aroma_logits']).squeeze(-1) > 0.5).long()
        pred_charges = pred['charge_logits'].argmax(dim=-1) - 6
        
        # element_types: predict from logits
        pred_element_types = pred['element_type_logits'].argmax(dim=-1)
        
        # For retro: product elements unchanged, only dummy elements predicted
        if mode == 'retro':
            gen_masks_element = torch.logical_xor(src_graph['masks'], src_graph['padding_masks'])
            element_types = torch.where(
                gen_masks_element,
                pred_element_types,
                src_graph['element_types']
            )
        else:
            element_types = src_graph['element_types']
        
        # Get element IDs from element types
        elements = get_element_id_batch(element_types)

        return {
            'bonds': pred_bonds,
            'aromas': pred_aromas,
            'charges': pred_charges,
            'elements': elements,
            'element_types': element_types,
            'masks': src_graph['padding_masks'],
            'flags': src_graph['flags']
        }

    
        
    
    
    def element_loss(self, logits, tgt_element_types, tgt_masks):
        tgt_masks = tgt_masks.bool().float()
        B, L = logits.shape[:2]
        CE = nn.CrossEntropyLoss(reduction='none')
        element_loss = CE(logits.permute(0, 2, 1), tgt_element_types)
        element_loss = element_loss * (1-tgt_masks)
        element_loss = element_loss.sum(dim=1) / (tgt_masks.sum(dim=1)+1e-6)
        return element_loss
    
    def aroma_loss(self, logits, tgt_aromas, tgt_masks):
        tgt_masks = tgt_masks.bool().float()
        B, L = tgt_aromas.shape[:2]
        BCE = nn.BCEWithLogitsLoss(reduction='none')
        tgt_aromas = tgt_aromas.bool().float()
        logits = logits.view(B, L)
        aroma_loss = BCE(logits, tgt_aromas.float()) #[B, L]
        aroma_loss = aroma_loss * (1-tgt_masks)
        aroma_loss = aroma_loss.sum(dim=1) / (tgt_masks.sum(dim=1)+1e-6)
        return aroma_loss
    
    def charge_loss(self, logits, tgt_charges, tgt_masks):
        tgt_masks = tgt_masks.bool().float()
        CE = nn.CrossEntropyLoss(reduction='none')
        # assumes [B, C, L] (input, target)        
        tgt_charges = tgt_charges.long() + 6
        charge_loss = CE(logits.permute(0, 2, 1), tgt_charges)
        charge_loss = charge_loss * (1-tgt_masks)
        charge_loss = charge_loss.sum(dim=1) / (tgt_masks.sum(dim=1)+1e-6)
        return charge_loss

    def ce_bond_loss(self, bond_logits, tgt_bonds, tgt_masks, padding_masks):
        B, L = tgt_bonds.shape[:2]
        tgt_masks = tgt_masks.float() # 1 iff masked
        or_masks = 1 - torch.einsum("bl,bk->blk", tgt_masks, tgt_masks) # notice that this doesn't mask the edges between target and side products
        and_masks = torch.einsum("bl,bk->blk", 1-tgt_masks, 1-tgt_masks)
        tgt_bonds = tgt_bonds * and_masks

        pad_masks = 1 - padding_masks.float()
        # [B, L], 0 if padding
        pad_masks = torch.einsum("bl,bk->blk", pad_masks, pad_masks)
        # Set the lower triangular part (including diagonal) to -100
        
        if self.force_symbond:
            lower_tri_mask = torch.tril(
                torch.ones(L, L, device=tgt_bonds.device)
            ).unsqueeze(0).expand(B, -1, -1)
            tgt_bonds = tgt_bonds * (1 - lower_tri_mask) + lower_tri_mask * (-100)
        else:
            eye_mask = torch.eye(
                L, device=tgt_bonds.device
            ).unsqueeze(0).expand(B, -1, -1)
            tgt_bonds = tgt_bonds * (1 - eye_mask) + eye_mask * (-100)
        CE = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        ignor_masks = (tgt_bonds != -100)
        loss = CE(bond_logits.permute(0, 3, 1, 2), tgt_bonds.long())
        loss = loss*pad_masks*or_masks*ignor_masks
        loss = loss.sum(dim=(1, 2)) / (pad_masks*or_masks*ignor_masks).sum(dim=(1, 2))
        return loss