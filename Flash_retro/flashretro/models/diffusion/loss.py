import torch
import torch.nn as nn
import numpy as np
from torch.func import jvp

class MeanFlowLoss:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            time_sampler="logit_normal",
            time_mu=-0.4,
            time_sigma=1.0,
            ratio_r_not_equal_t=0.75,
            adaptive_p=1.0,
            # CFG related
            cfg_omega=1.0,
            cfg_min_t=0.0,
            cfg_max_t=0.8,
            ):
        self.weighting = weighting
        self.path_type = path_type
        
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.adaptive_p = adaptive_p
        
        self.cfg_omega = cfg_omega
        self.cfg_min_t = cfg_min_t
        self.cfg_max_t = cfg_max_t

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device):
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        fraction_equal = 1.0 - self.ratio_r_not_equal_t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        r = torch.where(equal_mask, t, r)
        
        return r, t 
    
    def __call__(self, velocity_network, z1, cond, mask=None, task_token=None):
        """
        z1: [L, B, D] Target (Clean) Latent
        cond: [L, B, D] Condition Latent
        mask: [B, L] padding mask
        """
        L, B, D = z1.shape
        device = z1.device
        
        # Sample time
        r, t = self.sample_time_steps(B, device) # [B]
        
        z0 = torch.randn_like(z1) # Noise [L, B, D]
        

        t_b = t.view(1, -1, 1)
        
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_b) # scalar or [1, B, 1]
        


        
        z_target = z1 # Data
        z_noise = z0  # Noise
        
        z_t = (1 - t_b) * z_target + t_b * z_noise
        v_t = -z_target + z_noise # velocity d/dt z_t
        

        time_diff = (t - r).view(1, -1, 1) # [1, B, 1]
        

        
        def fn_current(z, cur_r, cur_t):
            # z: [L, B, D]
            # cur_r, cur_t: [B]
            # Disable scaled_dot_product optimizations for JVP compatibility
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, 
                enable_math=True, 
                enable_mem_efficient=False
            ):
                return velocity_network(z, cond, cur_t, cur_r, src_mask=mask)
            
        primals = (z_t, r, t)
        tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
        

        
        u = velocity_network(z_t, cond, t, r, src_mask=mask)
        

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, 
            enable_math=True, 
            enable_mem_efficient=False
        ):
            _, dudt = jvp(fn_current, primals, tangents)
        
        u_target = v_t - time_diff * dudt
        
        error = u - u_target.detach()
        loss_per_element = error**2  # [L, B, D]
        
        # Apply mask and normalize
        if mask is not None:

             mask_t = mask.transpose(0, 1).unsqueeze(-1) # [L, B, 1]
             loss_per_element = loss_per_element * (~mask_t)
             
             # Calculate valid elements per sample
             valid_lengths = (~mask).sum(dim=1).float()  # [B]
             # Normalize by (valid_length * D) to get average loss per element
             L, B, D = loss_per_element.shape
             loss_per_sample = loss_per_element.sum(dim=(0, 2)) / (valid_lengths * D + 1e-6)  # [B]
        else:
             # No mask: average over all elements
             loss_per_sample = loss_per_element.mean(dim=(0, 2))  # [B]
        
        # Adaptive weighting (optional)
        if self.weighting == "adaptive":
             weights = 1.0 / (loss_per_sample.detach() + 1e-3).pow(self.adaptive_p)
             loss = weights * loss_per_sample
        else:
             loss = loss_per_sample
             
        return loss.mean()  # Mean over batch


