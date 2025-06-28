import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import time as time
import torchvision.transforms as T
import torchvision.datasets as datasets
import tempfile
import shutil
import argparse

# Importar a função calculate_fid_given_paths do pytorch_fid
from pytorch_fid.fid_score import calculate_fid_given_paths

# Função para verificar imagens corrompidas no dataset
def scan_corrupted_images(img_dir, attr_path, max_check=None):
    """
    Scan the dataset for corrupted images that cannot be loaded.
    
    Args:
        img_dir: Directory containing images
        attr_path: Path to attributes CSV file
        max_check: Maximum number of images to check (None for all)
    
    Returns:
        List of corrupted image paths
    """
    print("Scanning dataset for corrupted images...")
    attributes_df = pd.read_csv(attr_path)
    corrupted_files = []
    
    total_files = len(attributes_df) if max_check is None else min(max_check, len(attributes_df))
    
    for idx in range(total_files):
        img_name = attributes_df.iloc[idx, 0]
        img_path = os.path.join(img_dir, img_name)
        
        try:
            with Image.open(img_path) as img:
                img.convert('RGB')
                img.load()  # Force load to catch any corruption
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            print(f"Corrupted image found: {img_name} - {e}")
            corrupted_files.append(img_path)
        
        if idx % 1000 == 0:
            print(f"Checked {idx}/{total_files} images... Found {len(corrupted_files)} corrupted so far.")
    
    print(f"Scan complete. Found {len(corrupted_files)} corrupted images out of {total_files} checked.")
    return corrupted_files

# Acesso ao diretório de dados do CelebA.
# Por favor, ajuste este caminho para onde o dataset CelebA foi descompactado.
# Ele deve apontar para a pasta que contém 'img_align_celeba' e 'list_attr_celeba.csv'.
celeba_dataset_path = './data/face-vae' # Exemplo de caminho local. AJUSTE O SEU CAMINHO AQUI!

if not os.path.exists(celeba_dataset_path):
    print(f"ATENÇÃO: O diretório de dados '{celeba_dataset_path}' não foi encontrado.")
    print("Por favor, baixe o dataset CelebA e ajuste 'celeba_dataset_path' para o local correto.")
    print("O script tentará continuar, mas falhará sem os dados de imagem/atributos.")

# Hyperparameters (Ajustados para GANs, WGAN-GP e otimização de GPU)
batch_size = 32 
epochs = 50 
latent_dim = 100 
# Taxas de aprendizado REDUZIDAS para estabilidade máxima na depuração de NaNs.
# Aumente gradualmente APÓS a estabilização.
learning_rate_g = 0.000005 # Reduzido ainda mais para tentar evitar NaNs.
learning_rate_d = 0.000005 # Reduzido ainda mais para tentar evitar NaNs.
image_size = 128 
channels = 3 
num_attributes = 5 
lambda_gp = 10.0 # Valor padrão. Pode ser reduzido para 1.0 ou 5.0 se NaNs persistirem.
d_steps = 5 

# Definir os atributos-alvo
target_attrs = ['Smiling', 'Male', 'Blond_Hair', 'Eyeglasses', 'Wearing_Hat']

# Weight initialization function for GANs (DCGAN paper guidelines)
def weights_init(m):
    """Initialize weights for Conv and BatchNorm layers following DCGAN paper"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, latent_dim, num_attributes, img_channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_attributes = num_attributes

        self.attr_embedding = nn.Sequential(
            nn.Linear(self.num_attributes, 128),
            nn.ReLU(True)
        )

        self.projection_dim = 512 * 4 * 4
        self.fc_projection = nn.Linear(self.latent_dim + 128, self.projection_dim)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, img_channels, 4, 2, 1, bias=False), # 64x64 -> 128x128
            nn.Tanh() # Tanh para saída de imagem normalizada para [-1, 1]
        )
        self.apply(weights_init)

    def forward(self, z, attrs):
        attr_emb = self.attr_embedding(attrs)
        z_conditioned = torch.cat([z, attr_emb], 1)

        h = self.fc_projection(z_conditioned)
        h = F.relu(h)
        h = h.view(-1, 512, 4, 4)

        return self.main(h)

class Discriminator(nn.Module):
    def __init__(self, num_attributes, img_channels):
        super(Discriminator, self).__init__()
        self.num_attributes = num_attributes

        self.attr_embedding = nn.Sequential(
            nn.Linear(self.num_attributes, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4 + 128, 1),
            # nn.Sigmoid() # Removido para WGAN-GP
        )
        self.apply(weights_init)

    def forward(self, img, attrs):
        attr_emb = self.attr_embedding(attrs)
        features = self.feature_extractor(img)
        features = features.view(features.size(0), -1)
        conditioned_features = torch.cat([features, attr_emb], 1)
        return self.fc_final(conditioned_features)

# Inicializa modelos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device being used: {device}')

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.enabled = True 
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    torch.cuda.empty_cache()
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')

netG = Generator(latent_dim, num_attributes, channels).to(device)
netD = Discriminator(num_attributes, channels).to(device)

print("⚠️ torch.compile desabilitado para compatibilidade. Usando modelos padrão.")

# ***** AQUI: SCALER COMENTADO PARA FORÇAR FP32 E DEPURAR NaNs *****
# Usa mixed precision (AMP) para melhor utilização da GPU e menor consumo de VRAM
# Para depurar NaNs, é recomendado desabilitar temporariamente a mixed precision.
scaler = None # Comentei 'torch.amp.GradScaler('cuda')' para forçar FP32.

def print_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Max: {max_allocated:.2f}GB, Total: {total:.2f}GB")
        return allocated, cached, max_allocated, total
    return 0, 0, 0, 0

# Otimizadores para WGAN-GP: Adam com beta1=0.0 para o Discriminador é CRUCIAL para estabilidade
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate_d, betas=(0.0, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))

class CelebAConditionalDataset(Dataset):
    def __init__(self, img_dir, attr_path, transform=None, target_attrs=None):
        self.img_dir = img_dir
        self.attributes_df = pd.read_csv(attr_path)
        self.transform = transform
        self.target_attrs = target_attrs

    def __len__(self):
        return len(self.attributes_df)

    def __getitem__(self, idx):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                current_idx = (idx + attempt) % len(self.attributes_df)
                img_name = self.attributes_df.iloc[current_idx, 0]
                img_path = os.path.join(self.img_dir, img_name)
                
                # Try to open and convert the image
                image = Image.open(img_path).convert('RGB')
                
                # Verify the image can be loaded properly
                image.load()
                
                if self.transform:
                    image = self.transform(image)

                attrs = []
                if self.target_attrs:
                    for attr in self.target_attrs:
                        attrs.append(1 if self.attributes_df.iloc[current_idx][attr] == 1 else 0)

                attrs_tensor = torch.tensor(attrs, dtype=torch.float32)

                return image, attrs_tensor
                
            except (OSError, IOError, Image.UnidentifiedImageError, Exception) as e:
                print(f"Warning: Corrupted image at index {current_idx}, file: {img_name if 'img_name' in locals() else 'unknown'}. Error: {e}")
                print(f"Skipping to next image... (attempt {attempt + 1}/{max_retries})")
                continue
        
        # If we've exhausted all retries, raise an error
        raise RuntimeError(f"Could not load a valid image after {max_retries} attempts starting from index {idx}")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normaliza para [-1, 1]
])

img_dir = os.path.join(celeba_dataset_path, 'img_align_celeba', 'img_align_celeba')
attr_path = os.path.join(celeba_dataset_path, 'list_attr_celeba.csv')

# Optional: Scan for corrupted images (set to True to enable)
scan_for_corrupted = False
if scan_for_corrupted:
    corrupted_files = scan_corrupted_images(img_dir, attr_path, max_check=1000)  # Check first 1000 images
    if corrupted_files:
        print(f"Warning: Found {len(corrupted_files)} corrupted images. Training will skip these automatically.")

dataset = CelebAConditionalDataset(
    img_dir=img_dir,
    attr_path=attr_path,
    transform=transform,
    target_attrs=target_attrs
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)

os.makedirs("results/cgan_optimized", exist_ok=True)
os.makedirs("models/cgan_optimized", exist_ok=True)

log_file_path = "results/cgan_optimized/training_log.txt"
log_file = open(log_file_path, "w")
log_file.write("Iniciando o treinamento de CGAN com WGAN-GP OTIMIZADO\n")
log_file.write(f"Hyperparameters: Batch Size={batch_size}, Epochs={epochs}, Latent Dim={latent_dim}\n")
log_file.write(f"Learning Rate G={learning_rate_g}, Learning Rate D={learning_rate_d}\n")
log_file.write(f"Lambda GP={lambda_gp}, D Steps={d_steps}\n")
log_file.write("-" * 50 + "\n")

# Funções para checkpoint
def save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, g_losses, d_losses, best_fid, checkpoint_dir='checkpoints'):
    """
    Salva um checkpoint completo do treinamento.
    
    Args:
        epoch: Época atual
        netG: Modelo Generator
        netD: Modelo Discriminator  
        optimizerG: Otimizador do Generator
        optimizerD: Otimizador do Discriminator
        g_losses: Lista de perdas do Generator
        d_losses: Lista de perdas do Discriminator
        best_fid: Melhor FID score até agora
        checkpoint_dir: Diretório para salvar checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'g_losses': g_losses,
        'd_losses': d_losses,
        'best_fid': best_fid,
        'hyperparameters': {
            'batch_size': batch_size,
            'latent_dim': latent_dim,
            'learning_rate_g': learning_rate_g,
            'learning_rate_d': learning_rate_d,
            'image_size': image_size,
            'channels': channels,
            'num_attributes': num_attributes
        }
    }
    
    # Salva o checkpoint mais recente
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Salva checkpoint a cada 10 épocas
    if epoch % 10 == 0:
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
        torch.save(checkpoint, epoch_checkpoint_path)
    
    print(f"Checkpoint salvo: {checkpoint_path}")

def load_checkpoint(checkpoint_path, netG, netD, optimizerG, optimizerD, device):
    """
    Carrega um checkpoint e restaura o estado do treinamento.
    
    Args:
        checkpoint_path: Caminho para o arquivo de checkpoint
        netG: Modelo Generator
        netD: Modelo Discriminator
        optimizerG: Otimizador do Generator
        optimizerD: Otimizador do Discriminator
        device: Device (CPU/GPU)
    
    Returns:
        Tuple com (start_epoch, g_losses, d_losses, best_fid)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint não encontrado: {checkpoint_path}")
        return 1, [], [], float('inf')
    
    print(f"Carregando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restaura os modelos
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netD.load_state_dict(checkpoint['netD_state_dict'])
    
    # Restaura os otimizadores
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    
    # Restaura as variáveis de treinamento
    start_epoch = checkpoint['epoch'] + 1
    g_losses = checkpoint.get('g_losses', [])
    d_losses = checkpoint.get('d_losses', [])
    best_fid = checkpoint.get('best_fid', float('inf'))
    
    print(f"Checkpoint carregado! Retomando do epoch {start_epoch}")
    print(f"Melhor FID até agora: {best_fid:.4f}")
    
    return start_epoch, g_losses, d_losses, best_fid

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """
    Encontra o checkpoint mais recente no diretório.
    
    Args:
        checkpoint_dir: Diretório de checkpoints
    
    Returns:
        Caminho para o checkpoint mais recente ou None se não encontrar
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_path):
        return latest_path
    
    # Procura por checkpoints de época
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if checkpoint_files:
        # Ordena por número da época e retorna o mais recente
        checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        return os.path.join(checkpoint_dir, checkpoint_files[-1])
    
    return None

def list_available_checkpoints(checkpoint_dir='checkpoints'):
    """
    Lista todos os checkpoints disponíveis no diretório.
    
    Args:
        checkpoint_dir: Diretório de checkpoints
    
    Returns:
        Lista de dicionários com informações dos checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Diretório de checkpoints não encontrado: {checkpoint_dir}")
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pth'):
            file_path = os.path.join(checkpoint_dir, file)
            try:
                checkpoint_info = torch.load(file_path, map_location='cpu')
                if 'epoch' in checkpoint_info:
                    checkpoints.append({
                        'file': file,
                        'path': file_path,
                        'epoch': checkpoint_info['epoch'],
                        'best_fid': checkpoint_info.get('best_fid', 'N/A'),
                        'size_mb': os.path.getsize(file_path) / 1024 / 1024,
                        'modified': time.ctime(os.path.getmtime(file_path))
                    })
            except Exception as e:
                print(f"Erro ao ler checkpoint {file}: {e}")
    
    # Ordena por época
    checkpoints.sort(key=lambda x: x['epoch'])
    return checkpoints

def print_checkpoint_info(checkpoint_dir='checkpoints'):
    """
    Imprime informações detalhadas sobre checkpoints disponíveis.
    """
    checkpoints = list_available_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print("Nenhum checkpoint encontrado.")
        return
    
    print(f"\n=== Checkpoints Disponíveis em '{checkpoint_dir}' ===")
    print(f"{'Arquivo':<25} {'Época':<8} {'Melhor FID':<12} {'Tamanho':<10} {'Modificado'}")
    print("-" * 80)
    
    for cp in checkpoints:
        fid_str = f"{cp['best_fid']:.2f}" if isinstance(cp['best_fid'], float) else str(cp['best_fid'])
        print(f"{cp['file']:<25} {cp['epoch']:<8} {fid_str:<12} {cp['size_mb']:.1f}MB {cp['modified']}")
    
    print("-" * 80)
    print(f"Total: {len(checkpoints)} checkpoints")

if __name__ == '__main__':
    # Argumentos de linha de comando para controle de checkpoints
    parser = argparse.ArgumentParser(description='GAN Training with Checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--checkpoint-path', type=str, help='Path to specific checkpoint file to load')
    parser.add_argument('--no-checkpoint', action='store_true', help='Start training from scratch, ignoring any existing checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='Save checkpoint every N epochs (default: 1)')
    parser.add_argument('--list-checkpoints', action='store_true', help='List available checkpoints and exit')
    
    args = parser.parse_args()
    
    # Cria o diretório de checkpoints se não existir
    os.makedirs('checkpoints', exist_ok=True)
    
    # Se solicitado, lista checkpoints e sai
    if args.list_checkpoints:
        print_checkpoint_info('checkpoints')
        exit(0)
    
    # Configuração do sistema de checkpoints
    if args.no_checkpoint:
        start_epoch = 1
        g_losses_per_epoch = []
        d_losses_per_epoch = []
        best_fid_score = float('inf')
        print("Iniciando treinamento do zero (ignorando checkpoints existentes).")
    elif args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            start_epoch, g_losses_per_epoch, d_losses_per_epoch, best_fid_score = load_checkpoint(
                args.checkpoint_path, netG, netD, optimizerG, optimizerD, device)
            print(f"Carregando checkpoint específico: {args.checkpoint_path}")
        else:
            print(f"Checkpoint especificado não encontrado: {args.checkpoint_path}")
            start_epoch = 1
            g_losses_per_epoch = []
            d_losses_per_epoch = []
            best_fid_score = float('inf')
    elif args.resume or not args.no_checkpoint:
        # Tenta carregar o último checkpoint automaticamente
        latest_checkpoint = find_latest_checkpoint('checkpoints')
        if latest_checkpoint:
            start_epoch, g_losses_per_epoch, d_losses_per_epoch, best_fid_score = load_checkpoint(
                latest_checkpoint, netG, netD, optimizerG, optimizerD, device)
            print(f"Retomando treinamento a partir do epoch {start_epoch}...")
        else:
            start_epoch = 1
            g_losses_per_epoch = []
            d_losses_per_epoch = []
            best_fid_score = float('inf')
            print("Nenhum checkpoint encontrado. Iniciando treinamento do zero.")
    else:
        start_epoch = 1
        g_losses_per_epoch = []
        d_losses_per_epoch = []
        best_fid_score = float('inf')
        print("Iniciando treinamento do zero.")
        
    print(f"Configuração de checkpoints: Salvar a cada {args.checkpoint_interval} épocas")

    # Teste inicial do dataloader e modelos
    if len(dataloader) > 0:
        batch_images, batch_attrs = next(iter(dataloader))
        print(f"Batch de Imagens shape: {batch_images.shape}")
        print(f"Batch de Atributos shape: {batch_attrs.shape}")
        
        print(f"Batch de imagens device: {batch_images.device}")
        test_batch = batch_images.to(device)
        test_attrs = batch_attrs.to(device)
        print(f"Após mover para GPU - device: {test_batch.device}")

        print("\n=== Status Inicial da GPU ===")
        print_gpu_memory_usage()
        print("==============================\n")
        
        print("Testando forward pass do Generator...")
        test_noise = torch.randn(4, latent_dim, device=device)
        test_attrs_small = test_attrs[:4]
        with torch.no_grad():
            test_output = netG(test_noise, test_attrs_small)
        print(f"Output do Generator shape: {test_output.shape}")
        print("✅ Forward pass funcionando!")
        
        print("\n=== Status da GPU após teste ===")
        print_gpu_memory_usage()
        print("==============================\n")

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            if i < batch_images.shape[0]:
                img = batch_images[i]
                img = img * 0.5 + 0.5
                img = img.permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                attr_labels = " ".join([f"{target_attrs[j]}: {'Y' if batch_attrs[i,j] == 1 else 'N'}" for j in range(num_attributes)])
                ax.set_title(attr_labels, fontsize=8)
                ax.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("DataLoader está vazio. Verifique o caminho dos dados e a configuração do dataset.")    # Funções auxiliares para o treinamento
    def calculate_gradient_penalty(discriminator, real_samples, fake_samples, attrs, lambda_gp, device):
        """Calcula o gradient penalty para WGAN-GP."""
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # É CRÍTICO que interpolates tenha requires_grad=True para o cálculo do GP
        # real_samples e fake_samples devem ser tensores com histórico de gradientes (não .data)
        # AQUI: real_samples e fake_samples JÁ DEVEM TER requires_grad=True
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
        
        # O .requires_grad_(True) é aplicado APÓS a operação para garantir que o histórico seja rastreado.
        # No entanto, a forma mais robusta é clonar e aplicar.
        interpolates.requires_grad_(True) 
        
        # discriminator é chamado aqui. d_interpolates PRECISA de um grad_fn
        d_interpolates = discriminator(interpolates, attrs)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates, device=device),
            create_graph=True, 
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1) 
        gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
        return gradient_penalty

    # Configurações para FID
    real_data_root = os.path.join(celeba_dataset_path, 'img_align_celeba', 'img_align_celeba')
    num_fid_samples = 2000 

    fid_image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])

    def generate_and_save_images_for_fid(generator, latent_dim, num_attributes, num_images, device, output_path):
        """Gera e salva imagens para o cálculo do FID."""
        generator.eval() 
        imgs_saved = 0
        fid_batch_size = min(256, num_images)
        
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        pbar_gen = tqdm(total=num_images, desc="Gerando imagens para FID")
        while imgs_saved < num_images:
            current_batch_size = min(num_images - imgs_saved, fid_batch_size)
            batch_noise = torch.randn(current_batch_size, latent_dim, device=device)
            batch_attrs = torch.randint(0, 2, (current_batch_size, num_attributes), dtype=torch.float32).to(device)

            with torch.no_grad():
                fake_images = generator(batch_noise, batch_attrs).cpu()

            for j in range(fake_images.size(0)):
                img = fake_images[j]
                img = img * 0.5 + 0.5
                save_image(img, os.path.join(output_path, f'generated_fid_{imgs_saved:05d}.png'))
                imgs_saved += 1
                pbar_gen.update(1)
                if imgs_saved >= num_images:
                    break
        pbar_gen.close()
        generator.train()
        return output_path

    def generate_image(generator, noise, attrs, device):
        """Gera uma única imagem dado ruído e atributos."""
        generator.eval()
        with torch.no_grad():
            noise = noise.to(device)
            attrs = attrs.to(device)
            generated_image = generator(noise, attrs)
            generated_image = generated_image * 0.5 + 0.5
        generator.train()
        return generated_image.cpu()

    # Inicialização das variáveis de treinamento
    fid_scores_per_epoch = []

    print("Iniciando o treinamento...")

    fixed_noise = torch.randn(96, latent_dim, device=device)
    fixed_attrs_base = torch.tensor([
        [1, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0], 
        [1, 0, 1, 0, 0], 
        [0, 1, 1, 0, 0], 
        [1, 0, 0, 1, 0], 
        [0, 1, 0, 1, 0], 
        [1, 0, 0, 0, 1], 
        [0, 1, 0, 0, 1], 
        [0, 0, 0, 0, 0], 
        [1, 1, 0, 0, 0], 
        [0, 0, 1, 0, 0], 
        [1, 0, 0, 1, 1]  
    ], dtype=torch.float32).to(device)

    fixed_attrs = fixed_attrs_base.repeat(8, 1) 

    start_time = time.time()

    best_g_loss = float('-inf')
    best_epoch = 0

    temp_gen_fid_dir = tempfile.mkdtemp(prefix='fid_gen_')

    # Função para formatar tempo
    def format_time(seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"

    # Loop principal de treinamento
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs}")

        current_d_loss_sum = 0
        current_g_loss_sum = 0
        g_steps_count = 0

        log_frequency = max(1, len(dataloader) // 5)

        for i, (real_images, attrs) in pbar:
            if i % log_frequency == 0:
                log_file.write(f"Epoch {epoch}/{epochs}, Batch {i}: real_images min: {real_images.min().item():.4f}, max: {real_images.max().item():.4f}\n")

            real_images = real_images.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            netD.zero_grad()
            
            # Generate fake images for discriminator training
            noise = torch.randn(batch_size, latent_dim, device=device)
            
            # ***** AQUI: Autocast para D_loss (FP16) - Desabilitado por scaler=None *****
            # O uso de 'torch.amp.autocast' é condicional a 'scaler is not None'.
            # Como 'scaler' está definido como 'None' para depuração, tudo está rodando em FP32.
            if torch.cuda.is_available() and scaler is not None:
                with torch.amp.autocast('cuda'):
                    fake_images_for_d = netG(noise, attrs)
                    real_output = netD(real_images, attrs).view(-1)
                    fake_output = netD(fake_images_for_d.detach(), attrs).view(-1)
            else:
                # FP32 training without autocast
                fake_images_for_d = netG(noise, attrs)
                real_output = netD(real_images, attrs).view(-1)
                fake_output = netD(fake_images_for_d.detach(), attrs).view(-1)

            # Para o GP, precisamos que real_images e fake_images_for_d tenham requires_grad=True
            # Clonar e aplicar requires_grad_(True) é a forma mais segura.
            real_images_for_gp = real_images.clone().detach().requires_grad_(True)
            fake_images_for_gp = fake_images_for_d.clone().detach().requires_grad_(True)

            # calculate_gradient_penalty é chamado fora de autocast para garantir FP32 para GP
            gp = calculate_gradient_penalty(netD, real_images_for_gp, fake_images_for_gp, attrs, lambda_gp, device)
            
            if i % log_frequency == 0:
                log_file.write(f"Epoch {epoch}/{epochs}, Batch {i}: Gradient Penalty (GP): {gp.item():.4f}\n")

            d_loss = -torch.mean(real_output) + torch.mean(fake_output) + gp
            
            # Backward pass e otimização
            if scaler: # Este bloco não será executado por enquanto, já que scaler é None
                scaler.scale(d_loss).backward()
                scaler.unscale_(optimizerD)
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                scaler.step(optimizerD)
                scaler.update()
            else: # Este bloco será executado (FP32)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                optimizerD.step()
                
            current_d_loss_sum += d_loss.item()

            if i % d_steps == 0: # Treinar o Gerador
                netG.zero_grad()
                noise = torch.randn(batch_size, latent_dim, device=device)
                
                # ***** AQUI: Autocast para G_loss (FP16) - Desabilitado por scaler=None *****
                if torch.cuda.is_available() and scaler is not None:
                    with torch.amp.autocast('cuda'):
                        fake_images = netG(noise, attrs) # Gerador produz novas fake_images
                        g_output = netD(fake_images, attrs).view(-1)
                        g_loss = -torch.mean(g_output) # Gerador tenta maximizar a saída do D para fake_images
                else:
                    # FP32 training without autocast
                    fake_images = netG(noise, attrs) # Gerador produz novas fake_images
                    g_output = netD(fake_images, attrs).view(-1)
                    g_loss = -torch.mean(g_output) # Gerador tenta maximizar a saída do D para fake_images

                if torch.isnan(d_loss) or torch.isnan(g_loss):
                    log_file.write(f"ATENÇÃO: NaN detectado na perda do D ou G na Época {epoch}, Batch {i}. Interrompendo treinamento.\n")
                    print(f"ATENÇÃO: NaN detectado na perda do D ou G na Época {epoch}, Batch {i}. Interrompendo treinamento.")
                    break 
                
                # Print detalhado das perdas para depuração de NaNs
                if i % 10 == 0: 
                    # Note: real_output e fake_output aqui são da última iteração do D, não deste G
                    print(f"DEBUG Batch {i}: D_Loss={d_loss.item():.4f}, G_Loss={g_loss.item():.4f}, Real_Output_Mean={torch.mean(real_output).item():.4f}, Fake_Output_Mean_D_Side={torch.mean(fake_output).item():.4f}, GP={gp.item():.4f}, G_Output_G_Side={torch.mean(g_output).item():.4f}")
                    
                if i % log_frequency == 0:
                    log_file.write(f"Epoch {epoch}/{epochs}, Batch {i}: G_Output (Discriminator on Fakes) Sample: {g_output[0].item():.4f}, Min: {g_output.min().item():.4f}, Max: {g_output.max().item():.4f}, Mean: {g_output.mean().item():.4f}\n")
                    log_file.write(f"Epoch {epoch}/{epochs}, Batch {i}: G_Loss calculated: {g_loss.item():.4f}\n")

                if scaler: # Este bloco não será executado por enquanto
                    scaler.scale(g_loss).backward()
                    scaler.unscale_(optimizerG)
                    torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
                    scaler.step(optimizerG)
                    scaler.update()
                else: # Este bloco será executado (FP32)
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
                    optimizerG.step()
                
                current_g_loss_sum += g_loss.item()
                g_steps_count += 1

                if i % (log_frequency * 2) == 0:
                    log_file.write(f"Epoch {epoch}/{epochs}, Batch {i}: Generator Gradients (Norms):\n")
                    for name, param in netG.named_parameters():
                        if param.grad is not None:
                            norm = param.grad.norm().item()
                            log_file.write(f"  {name}: {norm:.6f}\n")
                            if norm > 1e6:
                                log_file.write(f"  WARNING: Exploding gradient detected for {name} in Generator!\n")
                            elif norm < 1e-8 and i > 100:
                                log_file.write(f"  WARNING: Vanishing gradient detected for {name} in Generator!\n")
            
            pbar.set_postfix({'Loss D': f'{d_loss.item():.4f}', 'Loss G': f'{g_loss.item():.4f}'})

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        avg_d_loss = current_d_loss_sum / len(dataloader)
        avg_g_loss = current_g_loss_sum / g_steps_count if g_steps_count > 0 else 0

        g_losses_per_epoch.append(avg_g_loss)
        d_losses_per_epoch.append(avg_d_loss)

        total_time_elapsed = epoch_end_time - start_time
        avg_time_per_epoch = total_time_elapsed / epoch
        estimated_time_remaining = avg_time_per_epoch * (epochs - epoch)

        print(f"\n--- Epoch {epoch}/{epochs} Concluída ---")
        print(f"  Avg Loss D: {avg_d_loss:.4f} | Avg Loss G: {avg_g_loss:.4f}")
        print(f"  Duração da Época: {format_time(epoch_duration)}")
        print(f"  Tempo Total Decorrido: {format_time(total_time_elapsed)}")
        print(f"  Tempo Estimado Restante: {format_time(estimated_time_remaining)}")
        print(f"  Progresso Total: {(epoch/epochs)*100:.2f}%")
        
        if epoch % 5 == 0:
            print_gpu_memory_usage()

        log_file.write(f"\n--- Epoch {epoch}/{epochs} Concluída ---\n")
        log_file.write(f"  Avg Loss D: {avg_d_loss:.4f} | Avg Loss G: {avg_g_loss:.4f}\n")
        log_file.write(f"  Duração da Época: {format_time(epoch_duration)}\n")
        log_file.write(f"  Tempo Total Decorrido: {format_time(total_time_elapsed)}\n")
        log_file.write(f"  Tempo Estimado Restante: {format_time(estimated_time_remaining)}\n")
        log_file.write(f"  Progresso Total: {(epoch/epochs)*100:.2f}%\n")

        with torch.no_grad():
            generated_samples = netG(fixed_noise, fixed_attrs).cpu()
            save_image(generated_samples, f'results/cgan_optimized/epoch_{epoch:03d}_generated_samples.png', nrow=12, normalize=True)

        if epoch % 5 == 0 or epoch == epochs:
            print("\nCalculando FID...")
            log_file.write("\nCalculando FID...\n")

            shutil.rmtree(temp_gen_fid_dir, ignore_errors=True)
            temp_gen_fid_dir = tempfile.mkdtemp(prefix='fid_gen_')

            gen_path = generate_and_save_images_for_fid(netG, latent_dim, num_attributes, num_fid_samples, device, temp_gen_fid_dir)

            try:
                fid_value = calculate_fid_given_paths([real_data_root, gen_path],
                                                    batch_size,
                                                    device,
                                                    dims=2048)
                fid_scores_per_epoch.append(fid_value)
                print(f"  FID Score na Época {epoch}: {fid_value:.2f}")
                log_file.write(f"  FID Score na Época {epoch}: {fid_value:.2f}\n")
                
                # Atualiza o melhor FID score
                if fid_value < best_fid_score:
                    best_fid_score = fid_value
                    print(f"  >>> Novo melhor FID Score: {best_fid_score:.2f} na Época {epoch} <<<")
                    log_file.write(f"  >>> Novo melhor FID Score: {best_fid_score:.2f} na Época {epoch} <<<\n")
                    
            except Exception as e:
                print(f"Erro ao calcular FID na Época {epoch}: {e}")
                log_file.write(f"Erro ao calcular FID na Época {epoch}: {e}\n")
                fid_scores_per_epoch.append(float('nan'))

            shutil.rmtree(temp_gen_fid_dir)
            temp_gen_fid_dir = tempfile.mkdtemp(prefix='fid_gen_')

        if avg_g_loss > best_g_loss:
            best_g_loss = avg_g_loss
            best_epoch = epoch
            torch.save(netG.state_dict(), 'models/cgan_optimized/best_generator.pth')
            torch.save(netD.state_dict(), 'models/cgan_optimized/best_discriminator.pth')
            print(f"  >>> Modelo salvo! Nova melhor G Loss: {best_g_loss:.4f} na Época {best_epoch} <<<")
            log_file.write(f"  >>> Modelo salvo! Nova melhor G Loss: {best_g_loss:.4f} na Época {best_epoch} <<<\n")

        if epoch % 10 == 0:
            torch.save(netG.state_dict(), f'models/cgan_optimized/netG_epoch_{epoch:03d}.pth')
            torch.save(netD.state_dict(), f'models/cgan_optimized/netD_epoch_{epoch:03d}.pth')
        
        # Salva checkpoint conforme o intervalo configurado
        if epoch % args.checkpoint_interval == 0:
            save_checkpoint(epoch, netG, netD, optimizerG, optimizerD, g_losses_per_epoch, d_losses_per_epoch, best_fid_score, checkpoint_dir='checkpoints')

    print("\nTreinamento concluído.")
    log_file.write("\nTreinamento concluído.\n")
    log_file.close()

    total_training_duration = time.time() - start_time
    print(f"Duração total do treinamento: {format_time(total_training_duration)}")

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses_per_epoch, label='Discriminator Loss')
    plt.plot(g_losses_per_epoch, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/cgan_optimized/training_losses.png')
    plt.show()

    if len(fid_scores_per_epoch) > 0:
        plt.figure(figsize=(10, 5))
        epochs_fid = [e for e in range(1, epochs + 1) if e % 5 == 0 or e == epochs] 
        plt.plot(epochs_fid, fid_scores_per_epoch, marker='o', linestyle='-', color='red', label='FID Score')
        plt.xlabel('Epoch')
        plt.ylabel('FID Score')
        plt.title('FID Score Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/cgan_optimized/fid_scores.png')
        plt.show()

    # Exemplo de uso da função generate_image (descomente para testar)
    # single_noise = torch.randn(1, latent_dim, device=device)
    # desired_attrs = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32).to(device) # Smiling, Male, Blond Hair
    # generated_single_image = generate_image(netG, single_noise, desired_attrs, device)
    # save_image(generated_single_image, 'results/cgan_optimized/single_generated_image.png')

    # plt.imshow(generated_single_image.squeeze().permute(1, 2, 0).numpy())
    # plt.title("Generated Image with Attributes: Smiling, Male, Blond Hair")
    # plt.axis('off')
    # plt.show()