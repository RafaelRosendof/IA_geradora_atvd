import torch 
import torch.nn as nn
import torch.nn.functional as F





class VAE(nn.Module):

    
    def __init__(self):
        super(VAE, self).__init__()

        
 
        self.batch_size = 128
        self.epochs = 50
        self.latent_dim = 256
        self.learning_rate = 1e-3
        self.image_size = 128  # Using 64x64 for faster training, FFHQ is typically 1024x1024
        self.channels = 3

        # Encoder
        self.encoder = nn.Sequential(
            # first layer: [B, 3, 128, 128] -> [B, 16, 64, 64]
            nn.Conv2d(self.channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # second layer: [B, 16, 64, 64] -> [B, 32, 32, 32]
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # third layer: [B, 32, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # fourth layer: [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # fifth layer: [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # flatten: [B, 256, 4, 4] -> [B, 4096]
            nn.Flatten()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256*4*4, self.latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, self.latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(self.latent_dim, 256*4*4) # [B,latent_dim] -> [B,4096]
        
        self.decoder = nn.Sequential(
            # first layer: [B, 256, 4, 4] -> [B, 128, 8, 8]            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            # second layer: [B, 128, 8, 8] -> [B, 64, 16, 16]            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            # third layer: [B, 64, 16, 16] -> [B, 32, 32, 32]            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            # fourth layer: [B, 32, 32, 32] -> [B, 16, 64, 64]            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            # fifth layer: [B, 16, 64, 64] -> [B, 3, 128, 128]            
            nn.ConvTranspose2d(16, self.channels, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Tanh()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD, MSE, KLD


#Adjust this according to the celeba dataset 
class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim, cond_dim):
        super(Generator, self).__init__()

        # embedding layer for class
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # generator backbone
        self.model = nn.Sequential(
            # Input is latent_dim (noise)
            nn.Linear(latent_dim + cond_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, input_dim),
            nn.Tanh()  # Output between -1 and 1
        )

    def forward(self, x, c):
        # c -> embedding
        c = self.label_emb(c.int())
        # concatenate noise with conditioning
        x = torch.cat((x, c), dim=-1)
        # pass it throught G
        x = self.model(x)
        # reshape outout into image format
        x = x.view(x.size(0), img_channels, img_size, img_size)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Discriminator, self).__init__()

        # embedding layer: conditional GAN
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(input_dim + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1 (real or fake)
        )

    def forward(self, x, c):
        # flatten image: 2D -> 1D
        x = x.view(x.size(0), -1)
        # embedding of the conditioning
        c = self.label_emb(c.int())
        # concatenate input with conditioning
        x = torch.cat((x, c), dim=-1)
        # pass it through D
        y = self.model(x)
        return y
    
    
    
    
    
class DiffusionModel(nn.Module):
    """
    Placeholder for a diffusion model.
    This class should be implemented with the specific architecture and training logic for a diffusion model.
    """
    
    def __init__(self):
        super(DiffusionModel, self).__init__()
        # Define the architecture of the diffusion model here
        pass
    
    def forward(self, x):
        # Implement the forward pass for the diffusion model
        pass