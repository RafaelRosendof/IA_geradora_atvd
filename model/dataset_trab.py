import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict, Any
import numpy as np


class CelebADataset(Dataset):
    """
    CelebA Dataset for loading images and attributes.
    Can be used for VAE, GAN, and Diffusion models.
    """
    
    def __init__(
        self,
        root_dir: str,
        attr_file: str,
        image_size: int = 64,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        return_attributes: bool = False
    ):
        """
        Args:
            root_dir: Path to the directory containing images
            attr_file: Path to the attributes CSV file
            image_size: Size to resize images to (square)
            split: 'train', 'val', or 'test'
            transform: Optional custom transforms
            return_attributes: Whether to return attributes along with images
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.return_attributes = return_attributes
        
        # Load attributes
        self.attr_df = pd.read_csv(attr_file)
        
        # Define default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
            
        # Split the dataset (you can modify this logic based on your needs)
        total_samples = len(self.attr_df)
        if split == 'train':
            self.indices = list(range(0, int(0.8 * total_samples)))
        elif split == 'val':
            self.indices = list(range(int(0.8 * total_samples), int(0.9 * total_samples)))
        elif split == 'test':
            self.indices = list(range(int(0.9 * total_samples), total_samples))
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        self.data = self.attr_df.iloc[self.indices].reset_index(drop=True)
        
        # Get attribute names (excluding image_id)
        self.attr_names = [col for col in self.attr_df.columns if col != 'image_id']
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get image path and load image
        img_name = self.data.iloc[idx]['image_id']
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = torch.zeros(3, self.image_size, self.image_size)
        
        result = {'image': image}
        
        if self.return_attributes:
            # Get attributes (convert from {-1, 1} to {0, 1})
            attributes = self.data.iloc[idx][self.attr_names].values.astype(np.float32)
            attributes = (attributes + 1) / 2  # Convert from [-1, 1] to [0, 1]
            result['attributes'] = torch.tensor(attributes)
            result['image_id'] = img_name
            
        return result


class CelebADataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for CelebA dataset.
    Works for VAE, GAN, and Diffusion models.
    """
    
    def __init__(
        self,
        data_dir: str,
        attr_file: str,
        image_size: int = 64,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        return_attributes: bool = False,
        train_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None
    ):
        """
        Args:
            data_dir: Path to directory containing CelebA images
            attr_file: Path to the attributes CSV file
            image_size: Size to resize images to
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory in data loaders
            return_attributes: Whether to return attributes (useful for conditional models)
            train_transforms: Custom transforms for training
            val_transforms: Custom transforms for validation/test
        """
        super().__init__()
        self.data_dir = data_dir
        self.attr_file = attr_file
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.return_attributes = return_attributes
        
        # Define transforms
        self.train_transforms = train_transforms or transforms.Compose([
            transforms.Resize((image_size + 4, image_size + 4)),  # Slightly larger for random crop
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.val_transforms = val_transforms or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Save hyperparameters
        self.save_hyperparameters()
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = CelebADataset(
                root_dir=self.data_dir,
                attr_file=self.attr_file,
                image_size=self.image_size,
                split='train',
                transform=self.train_transforms,
                return_attributes=self.return_attributes
            )
            
            self.val_dataset = CelebADataset(
                root_dir=self.data_dir,
                attr_file=self.attr_file,
                image_size=self.image_size,
                split='val',
                transform=self.val_transforms,
                return_attributes=self.return_attributes
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = CelebADataset(
                root_dir=self.data_dir,
                attr_file=self.attr_file,
                image_size=self.image_size,
                split='test',
                transform=self.val_transforms,
                return_attributes=self.return_attributes
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def get_attribute_names(self) -> list:
        """Get list of attribute names."""
        # Create a dummy dataset to get attribute names
        dummy_dataset = CelebADataset(
            root_dir=self.data_dir,
            attr_file=self.attr_file,
            split='train'
        )
        return dummy_dataset.attr_names


# Example usage and helper functions
def create_celeba_datamodule(
    data_dir: str,
    attr_file: str,
    model_type: str = 'vae',
    image_size: int = 64,
    batch_size: int = 32
) -> CelebADataModule:
    """
    Factory function to create CelebA DataModule configured for different model types.
    
    Args:
        data_dir: Path to CelebA images
        attr_file: Path to attributes CSV
        model_type: 'vae', 'gan', or 'diffusion'
        image_size: Image size
        batch_size: Batch size
    """
    
    # Model-specific configurations
    if model_type.lower() == 'gan':
        # GANs often benefit from more aggressive augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((image_size + 8, image_size + 8)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return_attributes = False  # Unconditional GAN
        
    elif model_type.lower() == 'vae':
        # VAEs can use moderate augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((image_size + 4, image_size + 4)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return_attributes = True  # Can use attributes for conditional VAE
        
    elif model_type.lower() == 'diffusion':
        # Diffusion models often use minimal augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return_attributes = False  # Unconditional diffusion
        
    else:
        raise ValueError("model_type must be 'vae', 'gan', or 'diffusion'")
    
    return CelebADataModule(
        data_dir=data_dir,
        attr_file=attr_file,
        image_size=image_size,
        batch_size=batch_size,
        return_attributes=return_attributes,
        train_transforms=train_transforms,
        num_workers=4
    )
