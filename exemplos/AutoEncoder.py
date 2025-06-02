# import pandas as pd 
#import re
from tqdm import tqdm
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import lightning as L
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score 
import torchvision.transforms as transforms

def conv2d(in_chans , out_chan , kernel = 4 , stride = 2 , padding = 1):
    
    return nn.Sequential(
        nn.Conv2d(
            in_chans,
            out_chan,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_chan),
        nn.ReLU()
    ) 
 
# This is a model for the fashion mnist 

class Encoder(nn.Module):
    def __init__(self, channels=1): #only gray chan
        super().__init__() 
        
        self.conv1 = conv2d(channels, 128)
        self.conv2 = conv2d(128, 256)
        self.conv3 = conv2d(256, 512)
        self.conv4 = conv2d(512, 1024)
        self.linear = nn.Linear(1024, 16)

    def forward(self, x):
        x = self.conv1(x)  # (batch size, 128, 14, 14)
        x = self.conv2(x)  # (bs, 256, 7, 7)
        x = self.conv3(x)  # (bs, 512, 3, 3)
        x = self.conv4(x)  # (bs, 1024, 1, 1)
        # Keep batch dimension when flattening
        x = self.linear(x.flatten(start_dim=1))  # (bs, 16)
        return x
    
        # (bs , 128 , 14 , 14)
        # (bs , 256 , 7 , 7)
        # (bs , 512 , 3 , 3)
        # (bs , 1024 , 1 , 1)
        
        # (bs , 32)
        #output should be a vector of size 32 from the encoder 

    
    
    
# for the decoder we need to use the transpose layer because ...

def conv_transpose_2d(
    in_chan,
    out_chan,
    kernel=3,
    stride=2,
    padding=1,
    output_padding=0,
    with_act=True,
):
    #with act because ...
    module = [
        nn.ConvTranspose2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
    ]

    if with_act:
        module.append(nn.BatchNorm2d(out_chan))
        
        module.append(nn.ReLU())
        
    return nn.Sequential(*module) #include all parameters 


class Decoder(nn.Module):
    
    def __init__(self , out_chan=1):
        super().__init__()
        
        self.linear = nn.Linear(
            16, 1024 * 4 * 4
        )  # note it's reshaped in forward
        self.t_conv1 = conv_transpose_2d(1024, 512)
        self.t_conv2 = conv_transpose_2d(512, 256, output_padding=1)
        self.t_conv3 = conv_transpose_2d(256, out_chan, output_padding=1)

    def forward(self, x):
        bs = x.shape[0]
        x = self.linear(x)  # (bs, 1024*4*4)
        x = x.reshape((bs, 1024, 4, 4))  # (bs, 1024, 4, 4)
        x = self.t_conv1(x)  # (bs, 512, 7, 7)
        x = self.t_conv2(x)  # (bs, 256, 14, 14)
        x = self.t_conv3(x)  # (bs, 1, 28, 28)
        return x


class AutoEncoderPL(L.LightningModule):
    def __init__(self , learning_rate=1e-3):
        #super.__init__()
        super(AutoEncoderPL, self).__init__()
        #load the 2 models 
        
        self.encoder = Encoder(channels=1)
        self.decoder = Decoder(out_chan=1)
        
        self.save_hyperparameters()
        self.validation_step_outputs = []
        
        
    def forward(self , x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def training_step(self, batch , batch_idx):
        
        x , _ = batch 
        
        reconstructed = self(x)
        
        loss = F.mse_loss(reconstructed , x)
        
        self.log('train_loss' , loss , prog_bar=True)
        
        return loss 
    
    def validation_step(self, batch , batch_idx):
        x , _ = batch 
        
        reconstructed = self(x)
        
        val_loss = F.mse_loss(reconstructed , x)
        self.log('val_loss' , val_loss , prog_bar=True)
        
        if batch_idx == 0:
            self.validation_step_outputs.append((x, reconstructed))
            
        return val_loss
    
    def on_validation_batch_end(self):
        self.validation_step_outputs=[]
        
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    # This signature matches what Lightning expects
        pass  
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters() , lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim , mode='min' , factor=0.5 , patience=2
        )
        
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
        
        ################################ ENTENDER O MOTIVO DO ERRO DO KERNEL = 4 E KERNEL = 2 E KERNEL = 3 DA PARTE TRANSPOSE 
        
class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        
        self.batch_size = batch_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def prepare_data(self):
        # Download data if needed
        torchvision.datasets.FashionMNIST('./data', train=True, download=True)
        torchvision.datasets.FashionMNIST('./data', train=False, download=True)
        #torchvision.datasets.MNIST('./data', train=True, download=True)
        #torchvision.datasets.MNIST('./data', train=True, download=True)
        
        
    def setup(self, stage=None):
        # Load datasets
        self.train_dataset = torchvision.datasets.FashionMNIST(
            './data', train=True, transform=self.transform, download=False
        )
        
        self.val_dataset = torchvision.datasets.FashionMNIST(
            './data', train=False, transform=self.transform, download=False
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        

def main():
    
    data_module = FashionMNISTDataModule(batch_size=64)
    
    # Create model
    model = AutoEncoderPL(learning_rate=1e-4)
    
    # Configure trainer with callbacks
    trainer = L.Trainer(
        max_epochs=20,
        accelerator='auto',  # Use GPU if available, otherwise CPU
        devices=1,
        precision=32,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="val_loss",  # Monitor validation loss instead of F1
                mode="min",  # Lower loss is better
                save_top_k=1,
                filename="{epoch}-{val_loss:.4f}"
            ),
            L.pytorch.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                mode="min"  # Lower loss is better
            )
        ]
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Save the final model
    trainer.save_checkpoint("AutoEncoder.ckpt")

    
    #inference function bellow 
    
def inference(image_path, model_path="AutoEncoder.ckpt"):

    # Load the model
    model = AutoEncoderPL.load_from_checkpoint(model_path)
    model.eval()
    
    # Load and transform the image
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure grayscale
        transforms.Resize((28, 28)),  # Fashion MNIST size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        image = transforms.Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension
    except:
        
        try:
            idx = int(image_path)
            test_set = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
            image, _ = test_set[idx]
            image = transform(image).unsqueeze(0)
        except:
            raise ValueError("Invalid image path or index")
    
    # Perform reconstruction
    with torch.no_grad():
        reconstructed = model(image)
    
    return image, reconstructed



if __name__ == "__main__":
    main()