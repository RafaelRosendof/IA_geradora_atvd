#!/usr/bin/env python3
"""
Script para analisar os resultados do treinamento da GAN e identificar problemas.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """
    Analisa um checkpoint e extrai informações úteis.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint não encontrado: {checkpoint_path}")
        return None
    
    print(f"Analisando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'best_fid': checkpoint.get('best_fid', 'N/A'),
        'g_losses': checkpoint.get('g_losses', []),
        'd_losses': checkpoint.get('d_losses', []),
        'hyperparameters': checkpoint.get('hyperparameters', {})
    }
    
    print(f"  Época: {info['epoch']}")
    print(f"  Melhor FID: {info['best_fid']}")
    print(f"  Número de épocas com dados: {len(info['g_losses'])}")
    
    if info['g_losses']:
        g_losses = np.array(info['g_losses'])
        d_losses = np.array(info['d_losses'])
        
        print(f"  Generator Loss - Mean: {g_losses.mean():.4f}, Std: {g_losses.std():.4f}")
        print(f"  Generator Loss - Min: {g_losses.min():.4f}, Max: {g_losses.max():.4f}")
        print(f"  Discriminator Loss - Mean: {d_losses.mean():.4f}, Std: {d_losses.std():.4f}")
        print(f"  Discriminator Loss - Min: {d_losses.min():.4f}, Max: {d_losses.max():.4f}")
        
        # Detectar valores extremos
        if np.abs(d_losses).max() > 10000:
            print("  ⚠️  ALERTA: Discriminator Loss tem valores extremos!")
        if np.abs(g_losses).max() > 5000:
            print("  ⚠️  ALERTA: Generator Loss tem valores extremos!")
        
        # Analisar tendências
        if len(g_losses) >= 10:
            recent_g = g_losses[-10:]
            recent_d = d_losses[-10:]
            
            g_trend = "crescente" if recent_g[-1] > recent_g[0] else "decrescente"
            d_trend = "crescente" if recent_d[-1] > recent_d[0] else "decrescente"
            
            print(f"  Tendência G Loss (últimas 10 épocas): {g_trend}")
            print(f"  Tendência D Loss (últimas 10 épocas): {d_trend}")
    
    return info

def analyze_training_log(log_path):
    """
    Analisa o arquivo de log do treinamento.
    """
    if not os.path.exists(log_path):
        print(f"Log não encontrado: {log_path}")
        return
    
    print(f"\nAnalisando log: {log_path}")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Procurar por warnings e erros
    warnings = []
    errors = []
    nan_detections = []
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if 'warning' in line_lower:
            warnings.append((i+1, line.strip()))
        if 'error' in line_lower or 'erro' in line_lower:
            errors.append((i+1, line.strip()))
        if 'nan' in line_lower:
            nan_detections.append((i+1, line.strip()))
    
    print(f"  Total de linhas: {len(lines)}")
    print(f"  Warnings encontrados: {len(warnings)}")
    print(f"  Errors encontrados: {len(errors)}")
    print(f"  Detecções de NaN: {len(nan_detections)}")
    
    if warnings:
        print("\n  Últimos 5 warnings:")
        for line_num, warning in warnings[-5:]:
            print(f"    Linha {line_num}: {warning[:100]}...")
    
    if errors:
        print("\n  Últimos 5 errors:")
        for line_num, error in errors[-5:]:
            print(f"    Linha {line_num}: {error[:100]}...")
    
    if nan_detections:
        print("\n  Detecções de NaN:")
        for line_num, nan_line in nan_detections[-3:]:
            print(f"    Linha {line_num}: {nan_line[:100]}...")

def plot_training_analysis(checkpoint_path, save_path=None):
    """
    Cria gráficos detalhados de análise do treinamento.
    """
    info = analyze_checkpoint(checkpoint_path)
    if not info or not info['g_losses']:
        return
    
    g_losses = np.array(info['g_losses'])
    d_losses = np.array(info['d_losses'])
    epochs = np.arange(1, len(g_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, g_losses, label='Generator Loss', color='blue', alpha=0.7)
    axes[0, 0].plot(epochs, d_losses, label='Discriminator Loss', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss curves (log scale)
    axes[0, 1].semilogy(epochs, np.abs(g_losses), label='|Generator Loss|', color='blue', alpha=0.7)
    axes[0, 1].semilogy(epochs, np.abs(d_losses), label='|Discriminator Loss|', color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('|Loss| (log scale)')
    axes[0, 1].set_title('Training Losses (Log Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss histograms
    axes[1, 0].hist(g_losses, bins=50, alpha=0.7, label='Generator Loss', color='blue', density=True)
    axes[1, 0].hist(d_losses, bins=50, alpha=0.7, label='Discriminator Loss', color='red', density=True)
    axes[1, 0].set_xlabel('Loss Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Loss Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Moving averages
    window = min(10, len(g_losses) // 5)
    if window > 1:
        g_ma = pd.Series(g_losses).rolling(window=window).mean()
        d_ma = pd.Series(d_losses).rolling(window=window).mean()
        
        axes[1, 1].plot(epochs, g_ma, label=f'G Loss (MA-{window})', color='blue', linewidth=2)
        axes[1, 1].plot(epochs, d_ma, label=f'D Loss (MA-{window})', color='red', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (Moving Average)')
        axes[1, 1].set_title(f'Smoothed Losses (Moving Average)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Gráfico salvo em: {save_path}")
    
    plt.show()

def diagnose_training_issues(checkpoint_path):
    """
    Diagnóstica problemas comuns no treinamento.
    """
    print("\n" + "="*50)
    print("DIAGNÓSTICO DE PROBLEMAS DE TREINAMENTO")
    print("="*50)
    
    info = analyze_checkpoint(checkpoint_path)
    if not info or not info['g_losses']:
        print("Não foi possível analisar o checkpoint.")
        return
    
    g_losses = np.array(info['g_losses'])
    d_losses = np.array(info['d_losses'])
    
    issues = []
    recommendations = []
    
    # 1. Verificar valores extremos
    if np.abs(d_losses).max() > 50000:
        issues.append("Discriminator Loss extremamente alto")
        recommendations.append("Reduzir learning rate do discriminador para 1e-6 ou menor")
        recommendations.append("Verificar gradient penalty - pode estar muito alto")
        recommendations.append("Considerar usar spectral normalization")
    
    if np.abs(g_losses).max() > 10000:
        issues.append("Generator Loss extremamente alto")
        recommendations.append("Reduzir learning rate do gerador")
        recommendations.append("Verificar se o discriminador não está muito forte")
    
    # 2. Verificar instabilidade
    if len(g_losses) > 10:
        g_var = np.var(g_losses[-10:])
        d_var = np.var(d_losses[-10:])
        
        if g_var > 1000000:  # High variance in recent epochs
            issues.append("Generator Loss muito instável nas últimas épocas")
            recommendations.append("Reduzir learning rates")
            recommendations.append("Aumentar batch size se possível")
        
        if d_var > 1000000000:  # Very high variance
            issues.append("Discriminator Loss extremamente instável")
            recommendations.append("Reduzir lambda_gp para 5.0 ou 1.0")
            recommendations.append("Verificar se não há problemas com gradient penalty")
    
    # 3. Verificar tendências
    if len(g_losses) >= 20:
        recent_g = g_losses[-20:]
        recent_d = d_losses[-20:]
        
        # Tendência crescente consistente pode indicar problemas
        g_increasing = np.all(np.diff(recent_g[-10:]) > 0)
        d_magnitude_increasing = np.all(np.diff(np.abs(recent_d[-10:])) > 0)
        
        if g_increasing:
            issues.append("Generator Loss aumentando consistentemente")
            recommendations.append("Discriminador pode estar muito forte")
            recommendations.append("Considerar treinar G mais vezes por iteração")
        
        if d_magnitude_increasing:
            issues.append("Magnitude do Discriminator Loss aumentando")
            recommendations.append("Possível instabilidade numérica")
            recommendations.append("Verificar gradient clipping")
    
    # 4. Relatório
    if issues:
        print("\n🚨 PROBLEMAS IDENTIFICADOS:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
        
        print("\n💡 RECOMENDAÇÕES:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("\n✅ Nenhum problema crítico identificado nos dados disponíveis.")
    
    # 5. Sugestões de hyperparameters
    print(f"\n🔧 SUGESTÕES DE HYPERPARAMETERS:")
    current_hp = info.get('hyperparameters', {})
    
    print(f"Configuração atual:")
    for key, value in current_hp.items():
        print(f"  {key}: {value}")
    
    print(f"\nConfiguração sugerida para maior estabilidade:")
    print(f"  learning_rate_g: 1e-6  # Muito mais baixo")
    print(f"  learning_rate_d: 5e-7  # Ainda mais baixo")
    print(f"  lambda_gp: 1.0         # Reduzido")
    print(f"  d_steps: 3             # Menos steps do D")
    print(f"  batch_size: 16         # Menor se possível")

def main():
    """
    Função principal para análise.
    """
    print("Analisador de Treinamento GAN")
    print("=" * 40)
    
    # Caminhos dos arquivos
    checkpoint_dir = "checkpoints"
    log_file = "results/cgan_optimized/training_log.txt"
    
    # Encontrar o último checkpoint
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            # Priorizar latest_checkpoint.pth
            if 'latest_checkpoint.pth' in checkpoints:
                latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            else:
                # Ou o mais recente por nome
                checkpoints.sort()
                latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    
    if latest_checkpoint:
        print(f"\nAnalisando checkpoint: {latest_checkpoint}")
        
        # Análise básica
        analyze_checkpoint(latest_checkpoint)
        
        # Análise do log
        analyze_training_log(log_file)
        
        # Diagnóstico
        diagnose_training_issues(latest_checkpoint)
        
        # Criar gráficos
        try:
            plot_training_analysis(latest_checkpoint, 'results/cgan_optimized/training_analysis.png')
        except Exception as e:
            print(f"Erro ao criar gráficos: {e}")
    
    else:
        print("Nenhum checkpoint encontrado.")
        print("Verifique se o treinamento foi executado e gerou checkpoints.")

if __name__ == "__main__":
    main()
