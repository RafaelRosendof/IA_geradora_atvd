


'''

https://www.kaggle.com/code/rannierymaia/generation-of-mnist-images-using-gan
https://www.kaggle.com/code/rannierymaia/vae-celeba-image-generation


Example for the use of the datamodule for training a VAE or GAN or diffusion model on the CelebA dataset.

if __name__ == "__main__":
    # Example usage
    data_dir = "path/to/celeba/images"
    attr_file = "path/to/celeba/attributes.csv"
    
    # Create data module for VAE
    dm_vae = create_celeba_datamodule(
        data_dir=data_dir,
        attr_file=attr_file,
        model_type='vae',
        image_size=64,
        batch_size=32
    )
    
    # Setup and test
    dm_vae.setup('fit')
    train_loader = dm_vae.train_dataloader()
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"Image batch shape: {batch['image'].shape}")
    if 'attributes' in batch:
        print(f"Attributes shape: {batch['attributes'].shape}")
        print(f"Number of attributes: {len(dm_vae.get_attribute_names())}")

'''


##### Exemple for the use of LightningModule for training a VAE or GAN or diffusion model on the CelebA dataset.

''' 

this exemple is for the AutoEncoder 


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
        
'''