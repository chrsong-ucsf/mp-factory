import torch
import torch.nn as nn
from monai.networks.nets import SPADEDiffusionModelUNet

class AnatomyConditionalSynthesisNet(nn.Module):
    """
    Anatomy-Conditional Synthesis Network for Project 10 Phase 3 Stage 2.
    
    This model synthesizes a target phase latent representation from a source NCCT latent 
    representation, conditioned on an organ mask, using a SPADE-based Diffusion Model.
    
    Expected Inputs:
        x (torch.Tensor): The noisy latent tensor of the target phase being denoised.
                          Shape: [B, C, D, H, W] for 3D latents (e.g., [B, 4, 16, 16, 16]).
        timesteps (torch.Tensor): Diffusion timesteps. Shape: [B].
        source_latent (torch.Tensor): Source NCCT latent representation.
                                      Shape: [B, C, D, H, W] (e.g., [B, 4, 16, 16, 16]).
        anatomy_mask (torch.Tensor): The anatomical organ mask used for SPADE conditioning.
                                     Shape: [B, num_classes, D, H, W].
                            
    Returns:
        torch.Tensor: The predicted noise or denoised latent.
                      Shape: [B, C, D, H, W] (same as input `x` and `source_latent`).
    """
    def __init__(
        self,
        spatial_dims: int = 3,
        latent_channels: int = 4,
        label_nc: int = 5,
        channels: tuple = (128, 256, 256),
        attention_levels: tuple = (False, True, True),
        num_res_blocks: int = 2,
        num_head_channels: int = 64,
        with_conditioning: bool = False,
    ):
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.latent_channels = latent_channels
        self.label_nc = label_nc
        
        # The input to the UNet will be the concatenation of the noisy target latent 
        # and the source NCCT latent.
        self.in_channels = latent_channels * 2 
        self.out_channels = latent_channels
        
        self.model = SPADEDiffusionModelUNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            label_nc=self.label_nc,
            channels=channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
            with_conditioning=with_conditioning,
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        source_latent: torch.Tensor, 
        anatomy_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the anatomy conditional diffusion network.
        
        Args:
            x (torch.Tensor): Noisy target phase latent [B, latent_channels, D, H, W].
            timesteps (torch.Tensor): Diffusion timesteps [B].
            source_latent (torch.Tensor): Source NCCT latent [B, latent_channels, D, H, W].
            anatomy_mask (torch.Tensor): SPADE condition mask (anatomy) [B, label_nc, D, H, W].
            
        Returns:
            torch.Tensor: Denoised representation or predicted noise [B, latent_channels, D, H, W].
        """
        # Concatenate the noisy target and the source NCCT latent
        net_input = torch.cat([x, source_latent], dim=1)
        
        # Forward through SPADE Diffusion Model UNet
        # 'seg' parameter in SPADEDiffusionModelUNet is used for the spatial condition mask
        return self.model(x=net_input, timesteps=timesteps, seg=anatomy_mask)

if __name__ == "__main__":
    print("Running CLI Smoke Test for AnatomyConditionalSynthesisNet...")
    
    # Dummy shapes
    batch_size = 2
    latent_channels = 4
    spatial_size = 16  # e.g., 16x16x16 3D latents
    num_classes = 5
    
    # Initialize Model
    model = AnatomyConditionalSynthesisNet(
        spatial_dims=3,
        latent_channels=latent_channels,
        label_nc=num_classes,
        channels=(64, 128, 128),  # smaller channels for quick test
        attention_levels=(False, True, True),
        num_res_blocks=1,
    )
    
    print(f"Model instantiated successfully.")
    
    # Dummy tensors
    x = torch.randn(batch_size, latent_channels, spatial_size, spatial_size, spatial_size)
    source_latent = torch.randn(batch_size, latent_channels, spatial_size, spatial_size, spatial_size)
    timesteps = torch.randint(0, 1000, (batch_size,))
    anatomy_mask = torch.randint(0, 2, (batch_size, num_classes, spatial_size, spatial_size, spatial_size)).float()
    
    print(f"Inputs:")
    print(f"  - Noisy Target (x): {x.shape}")
    print(f"  - Timesteps: {timesteps.shape}")
    print(f"  - Source Latent: {source_latent.shape}")
    print(f"  - Anatomy Mask: {anatomy_mask.shape}")
    
    # Forward Pass
    with torch.no_grad():
        output = model(x, timesteps, source_latent, anatomy_mask)
        
    print(f"\nOutput:")
    print(f"  - Predicted Noise / Denoised: {output.shape}")
    
    # Verify shape
    assert output.shape == x.shape, f"Expected output shape {x.shape}, got {output.shape}"
    print("Smoke test passed: Output shape matches expected shape.")
