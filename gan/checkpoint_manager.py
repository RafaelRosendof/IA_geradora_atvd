#!/usr/bin/env python3
"""
Gerenciador de Checkpoints para GAN Training
"""

import os
import sys
import argparse
import torch
import time
from datetime import datetime

def list_checkpoints(checkpoint_dir='checkpoints'):
    """Lista todos os checkpoints disponíveis."""
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
                        'g_losses': len(checkpoint_info.get('g_losses', [])),
                        'd_losses': len(checkpoint_info.get('d_losses', [])),
                        'size_mb': os.path.getsize(file_path) / 1024 / 1024,
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                    })
            except Exception as e:
                print(f"Erro ao ler checkpoint {file}: {e}")
    
    # Ordena por época
    checkpoints.sort(key=lambda x: x['epoch'])
    return checkpoints

def print_checkpoint_info(checkpoint_dir='checkpoints'):
    """Imprime informações detalhadas sobre checkpoints."""
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print("Nenhum checkpoint encontrado.")
        return
    
    print(f"\n=== Checkpoints Disponíveis em '{checkpoint_dir}' ===")
    print(f"{'Arquivo':<30} {'Época':<8} {'Melhor FID':<12} {'Épocas':<8} {'Tamanho':<10} {'Modificado'}")
    print("-" * 95)
    
    for cp in checkpoints:
        fid_str = f"{cp['best_fid']:.2f}" if isinstance(cp['best_fid'], float) else str(cp['best_fid'])
        print(f"{cp['file']:<30} {cp['epoch']:<8} {fid_str:<12} {cp['g_losses']:<8} {cp['size_mb']:.1f}MB {cp['modified']}")
    
    print("-" * 95)
    print(f"Total: {len(checkpoints)} checkpoints")

def delete_checkpoint(checkpoint_path):
    """Deleta um checkpoint específico."""
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print(f"Checkpoint deletado: {checkpoint_path}")
        except Exception as e:
            print(f"Erro ao deletar checkpoint: {e}")
    else:
        print(f"Checkpoint não encontrado: {checkpoint_path}")

def cleanup_old_checkpoints(checkpoint_dir='checkpoints', keep_latest=5, keep_every_n=10):
    """
    Remove checkpoints antigos mantendo apenas os mais recentes e alguns marcos.
    
    Args:
        checkpoint_dir: Diretório de checkpoints
        keep_latest: Quantos checkpoints mais recentes manter
        keep_every_n: Manter um checkpoint a cada N épocas (marcos)
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if len(checkpoints) <= keep_latest:
        print(f"Apenas {len(checkpoints)} checkpoints encontrados. Nenhum será removido.")
        return
    
    # Determinar quais manter
    keep_files = set()
    
    # Manter os mais recentes
    for cp in checkpoints[-keep_latest:]:
        keep_files.add(cp['file'])
    
    # Manter marcos (a cada N épocas)
    for cp in checkpoints:
        if cp['epoch'] % keep_every_n == 0:
            keep_files.add(cp['file'])
    
    # Deletar os outros
    deleted_count = 0
    for cp in checkpoints:
        if cp['file'] not in keep_files:
            delete_checkpoint(cp['path'])
            deleted_count += 1
    
    print(f"Limpeza concluída: {deleted_count} checkpoints removidos, {len(keep_files)} mantidos.")

def backup_checkpoint(checkpoint_path, backup_dir='checkpoint_backups'):
    """Cria um backup de um checkpoint específico."""
    import shutil
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint não encontrado: {checkpoint_path}")
        return
    
    os.makedirs(backup_dir, exist_ok=True)
    
    # Nome do backup com timestamp
    base_name = os.path.basename(checkpoint_path)
    name, ext = os.path.splitext(base_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"{name}_backup_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_name)
    
    try:
        shutil.copy2(checkpoint_path, backup_path)
        print(f"Backup criado: {backup_path}")
    except Exception as e:
        print(f"Erro ao criar backup: {e}")

def main():
    parser = argparse.ArgumentParser(description='Gerenciador de Checkpoints para GAN Training')
    parser.add_argument('--list', action='store_true', help='Lista checkpoints disponíveis')
    parser.add_argument('--delete', type=str, help='Deleta um checkpoint específico')
    parser.add_argument('--cleanup', action='store_true', help='Remove checkpoints antigos')
    parser.add_argument('--backup', type=str, help='Cria backup de um checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Diretório de checkpoints')
    parser.add_argument('--keep-latest', type=int, default=5, help='Quantos checkpoints recentes manter na limpeza')
    parser.add_argument('--keep-every', type=int, default=10, help='Manter um checkpoint a cada N épocas')
    
    args = parser.parse_args()
    
    if args.list:
        print_checkpoint_info(args.checkpoint_dir)
    elif args.delete:
        delete_checkpoint(args.delete)
    elif args.cleanup:
        cleanup_old_checkpoints(args.checkpoint_dir, args.keep_latest, args.keep_every)
    elif args.backup:
        backup_checkpoint(args.backup)
    else:
        print("Use --help para ver as opções disponíveis")
        print_checkpoint_info(args.checkpoint_dir)

if __name__ == '__main__':
    main()
