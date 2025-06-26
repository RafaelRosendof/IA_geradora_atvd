import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from typing import Optional, Tuple, List
import numpy as np
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Basic building block with GroupNorm, SiLU activation, and Conv2d."""
    
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]  # Add spatial dimensions
        h = h + time_emb
        
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        # Downsample or Upsample
        return self.transform(h)


class UNet(nn.Module):
    """U-Net architecture for diffusion model."""
    
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(c_in, 64, 3, padding=1)
        
        # Downsample
        self.downs = nn.ModuleList([
            Block(64, 128, time_dim),      # 64 -> 32
            Block(128, 256, time_dim),     # 32 -> 16  
            Block(256, 256, time_dim),     # 16 -> 8
            Block(256, 512, time_dim),     # 8 -> 4
        ])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU()
        )
        
        # Upsample
        self.ups = nn.ModuleList([
            Block(512, 256, time_dim, up=True),   # 4 -> 8
            Block(256, 256, time_dim, up=True),   # 8 -> 16
            Block(256, 128, time_dim, up=True),   # 16 -> 32
            Block(128, 64, time_dim, up=True),    # 32 -> 64
        ])
        
        # Final projection
        self.output = nn.Conv2d(64, c_out, 1)

    def forward(self, x, timestep):
        # Time embedding
        t = self.time_mlp(timestep)
        
        # Initial conv
        x = self.conv0(x)
        
        # U-Net
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        
        x = self.bottleneck(x)
        
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        
        return self.output(x)


class DDPMScheduler:
    """DDPM noise scheduler."""
    
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        
        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)  
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def add_noise(self, x_start, t, noise=None):
        """Add noise to the input tensor."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, n):
        """Sample random timesteps."""
        return torch.randint(low=1, high=self.timesteps, size=(n,), device=self.device)
    
    def _extract(self, a, t, x_shape):
        """Extract values from tensor a at timesteps t."""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionModel(L.LightningModule):
    """Complete Diffusion Model with PyTorch Lightning."""
    
    def __init__(
        self,
        lr=2e-4,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        image_size=64,
        sample_steps=50,
        save_samples_every=10
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.timesteps = timesteps
        self.image_size = image_size
        self.sample_steps = sample_steps
        self.save_samples_every = save_samples_every
        
        # Model and scheduler
        self.model = UNet(device=self.device)
        self.scheduler = DDPMScheduler(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=self.device
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def forward(self, x, t):
        return self.model(x, t)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        batch_size = images.shape[0]
        
        # Sample noise and timesteps
        noise = torch.randn_like(images)
        timesteps = self.scheduler.sample_timesteps(batch_size)
        
        # Add noise to images
        x_t = self.scheduler.add_noise(images, timesteps, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, timesteps)
        
        # Calculate loss
        loss = self.criterion(predicted_noise, noise)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        batch_size = images.shape[0]
        
        # Sample noise and timesteps
        noise = torch.randn_like(images)
        timesteps = self.scheduler.sample_timesteps(batch_size)
        
        # Add noise to images
        x_t = self.scheduler.add_noise(images, timesteps, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, timesteps)
        
        # Calculate loss
        loss = self.criterion(predicted_noise, noise)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        """Generate and save samples at the end of validation."""
        if self.current_epoch % self.save_samples_every == 0:
            self.sample_and_save()
    
    @torch.no_grad()
    def sample(self, num_samples=16, save_path=None):
        """Generate samples using DDPM sampling."""
        self.model.eval()
        
        # Start with random noise
        x = torch.randn(num_samples, 3, self.image_size, self.image_size).to(self.device)
        
        # Sample with progress bar
        timesteps = list(range(self.timesteps))[::-1]
        
        for i in tqdm(timesteps, desc="Sampling"):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            
            # Get model prediction
            predicted_noise = self.model(x, t)
            
            # Get scheduler parameters
            alpha = self.scheduler.alphas[i]
            alpha_cumprod = self.scheduler.alphas_cumprod[i]
            beta = self.scheduler.betas[i]
            
            # DDPM sampling formula
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
        
        # Denormalize images from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        x = torch.clamp(x, 0, 1)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(x, save_path, nrow=4)
        
        return x
    
    def sample_and_save(self):
        """Sample and save images during training."""
        save_dir = f"samples/epoch_{self.current_epoch}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "generated_samples.png")
        self.sample(num_samples=16, save_path=save_path)
        print(f"Samples saved to {save_path}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }


# Training script and utilities
def train_diffusion_model(
    data_dir: str,
    attr_file: str,
    max_epochs: int = 100,
    batch_size: int = 32,
    image_size: int = 64,
    lr: float = 2e-4,
    accelerator: str = "gpu",
    devices: int = 1
):
    """Train the diffusion model."""
    
    # Create data module using your existing code
    from your_datamodule import create_celeba_datamodule  # Import your datamodule
    
    datamodule = create_celeba_datamodule(
        data_dir=data_dir,
        attr_file=attr_file,
        model_type='diffusion',
        image_size=image_size,
        batch_size=batch_size
    )
    
    # Create model
    model = DiffusionModel(
        lr=lr,
        image_size=image_size,
        timesteps=1000,
        sample_steps=50,
        save_samples_every=5
    )
    
    # Setup trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed",  # Use mixed precision for faster training
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        check_val_every_n_epoch=5,
        callbacks=[
            L.pytorch_lightning.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                filename="diffusion-{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3
            ),
            L.pytorch_lightning.callbacks.LearningRateMonitor(logging_interval="step")
        ]
    )
    
    # Train model
    trainer.fit(model, datamodule)
    
    return model, trainer


def generate_samples_from_checkpoint(checkpoint_path: str, num_samples: int = 64, save_path: str = "final_samples.png"):
    """Generate samples from a trained model checkpoint."""
    
    # Load model from checkpoint
    model = DiffusionModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    # Generate samples
    samples = model.sample(num_samples=num_samples, save_path=save_path)
    
    print(f"Generated {num_samples} samples and saved to {save_path}")
    return samples


# Example usage
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "/path/to/celeba/images"  # Update this path
    ATTR_FILE = "/path/to/celeba/attributes.csv"  # Update this path
    
    # Train the model
    print("Starting diffusion model training...")
    model, trainer = train_diffusion_model(
        data_dir=DATA_DIR,
        attr_file=ATTR_FILE,
        max_epochs=100,
        batch_size=32,
        image_size=64,
        lr=2e-4,
        accelerator="gpu",
        devices=1
    )
    
    # Generate final samples
    print("Generating final samples...")
    model.sample_and_save()
    
    print("Training completed!")


# Additional utility functions
def interpolate_samples(model, num_steps=10, save_path="interpolation.png"):
    """Generate interpolation between two random noise vectors."""
    model.eval()
    
    with torch.no_grad():
        # Generate two random starting points
        z1 = torch.randn(1, 3, model.image_size, model.image_size).to(model.device)
        z2 = torch.randn(1, 3, model.image_size, model.image_size).to(model.device)
        
        interpolated_samples = []
        
        for i in range(num_steps):
            # Linear interpolation
            alpha = i / (num_steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Generate sample from interpolated noise
            sample = model.sample_from_noise(z_interp)
            interpolated_samples.append(sample)
        
        # Concatenate and save
        grid = torch.cat(interpolated_samples, dim=0)
        grid = (grid + 1) / 2  # Denormalize
        save_image(grid, save_path, nrow=num_steps)
    
    print(f"Interpolation saved to {save_path}")


def calculate_fid_score(model, dataloader, num_samples=5000):
    """Calculate FID score (requires pytorch-fid package)."""
    try:
        from pytorch_fid import fid_score
        import tempfile
        
        model.eval()
        
        # Generate samples
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_dir = os.path.join(temp_dir, "fake")
            real_dir = os.path.join(temp_dir, "real")
            os.makedirs(fake_dir)
            os.makedirs(real_dir)
            
            # Save generated samples
            num_batches = num_samples // 16
            for i in range(num_batches):
                samples = model.sample(num_samples=16)
                for j, sample in enumerate(samples):
                    save_image(sample, os.path.join(fake_dir, f"fake_{i*16+j}.png"))
            
            # Save real samples
            count = 0
            for batch in dataloader:
                if count >= num_samples:
                    break
                images = batch['image']
                images = (images + 1) / 2  # Denormalize
                for j, image in enumerate(images):
                    if count >= num_samples:
                        break
                    save_image(image, os.path.join(real_dir, f"real_{count}.png"))
                    count += 1
            
            # Calculate FID
            fid_value = fid_score.calculate_fid_given_paths([real_dir, fake_dir], 
                                                          batch_size=50, device='cuda', dims=2048)
            return fid_value
            
    except ImportError:
        print("pytorch-fid package not installed. Install with: pip install pytorch-fid")
        return None