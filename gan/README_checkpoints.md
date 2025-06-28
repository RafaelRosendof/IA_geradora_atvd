# Sistema de Checkpoints para Treinamento GAN

Este guia explica como usar o sistema de checkpoints implementado no script de treinamento da GAN.

## 🚀 Características Principais

- **Salvamento Automático**: Checkpoints são salvos automaticamente durante o treinamento
- **Retomada Inteligente**: O treinamento pode ser retomado exatamente onde parou
- **Múltiplos Checkpoints**: Mantém histórico de checkpoints para diferentes épocas
- **Recuperação de Estado Completo**: Restaura modelos, otimizadores, métricas e configurações

## 📁 Estrutura de Checkpoints

```
checkpoints/
├── latest_checkpoint.pth          # Checkpoint mais recente
├── checkpoint_epoch_010.pth       # Checkpoint da época 10
├── checkpoint_epoch_020.pth       # Checkpoint da época 20
└── checkpoint_epoch_030.pth       # Checkpoint da época 30
```

## 🛠️ Como Usar

### 1. Iniciar Treinamento do Zero
```bash
python gan.py --no-checkpoint
```

### 2. Retomar Treinamento Automaticamente
```bash
python gan.py --resume
```
O script procurará automaticamente pelo checkpoint mais recente e retomará o treinamento.

### 3. Carregar Checkpoint Específico
```bash
python gan.py --checkpoint-path checkpoints/checkpoint_epoch_020.pth
```

### 4. Configurar Intervalo de Salvamento
```bash
python gan.py --checkpoint-interval 5
```
Salva checkpoint a cada 5 épocas (padrão é a cada 1 época).

### 5. Listar Checkpoints Disponíveis
```bash
python gan.py --list-checkpoints
```

## 📊 Informações Salvas no Checkpoint
- Estado dos modelos (Generator e Discriminator)
- Estado dos otimizadores (incluindo momentum, etc.)
- Época atual
- Histórico de perdas (Generator e Discriminator)
- Melhor FID score até o momento
- Hiperparâmetros utilizados

### Arquivos de checkpoint:
- `checkpoints/latest_checkpoint.pth` - Checkpoint mais recente
- `checkpoints/checkpoint_epoch_XXX.pth` - Checkpoints de marcos (a cada 10 épocas)

## Como usar os checkpoints

### 1. Treinamento normal (com checkpoints automáticos)
```bash
python gan.py
```
- Automaticamente carrega o último checkpoint se existir
- Salva checkpoints a cada época por padrão

### 2. Iniciar do zero (ignorando checkpoints)
```bash
python gan.py --no-checkpoint
```

### 3. Retomar de um checkpoint específico
```bash
python gan.py --checkpoint-path checkpoints/checkpoint_epoch_020.pth
```

### 4. Configurar intervalo de salvamento
```bash
python gan.py --checkpoint-interval 5
```
- Salva checkpoint a cada 5 épocas em vez de a cada época

### 5. Listar checkpoints disponíveis
```bash
python gan.py --list-checkpoints
```

## Gerenciador de Checkpoints

Use o script `checkpoint_manager.py` para operações avançadas:

### Listar checkpoints
```bash
python checkpoint_manager.py --list
```

### Fazer backup de um checkpoint
```bash
python checkpoint_manager.py --backup checkpoints/latest_checkpoint.pth
```

### Limpar checkpoints antigos
```bash
python checkpoint_manager.py --cleanup
```
- Mantém os 5 checkpoints mais recentes
- Mantém checkpoints de marcos (a cada 10 épocas)

### Limpar com configurações personalizadas
```bash
python checkpoint_manager.py --cleanup --keep-latest 3 --keep-every 20
```

### Deletar um checkpoint específico
```bash
python checkpoint_manager.py --delete checkpoints/checkpoint_epoch_010.pth
```

## Exemplo de uso típico

### Cenário 1: Primeira execução
```bash
# Primeira vez rodando o treinamento
python gan.py

# Se der problema na época 15, você pode retomar:
python gan.py
# Automaticamente retoma do checkpoint mais recente
```

### Cenário 2: Experimentação
```bash
# Treinar com configuração A
python gan.py --checkpoint-interval 10

# Fazer backup do melhor checkpoint
python checkpoint_manager.py --backup checkpoints/latest_checkpoint.pth

# Experimentar com configuração B do zero
python gan.py --no-checkpoint

# Se não funcionou bem, voltar para o backup
python gan.py --checkpoint-path checkpoint_backups/latest_checkpoint_backup_20250628_120000.pth
```

### Cenário 3: Manutenção
```bash
# Ver todos os checkpoints
python checkpoint_manager.py --list

# Limpar checkpoints antigos para economizar espaço
python checkpoint_manager.py --cleanup
```

## Vantagens do sistema de checkpoints

1. **Recuperação rápida**: Se o treinamento for interrompido, você não perde horas de progresso
2. **Experimentação segura**: Pode testar diferentes configurações sem medo
3. **Comparação de modelos**: Pode comparar modelos de diferentes épocas
4. **Backup automático**: Histórico completo de treinamento preservado
5. **Economia de tempo**: Não precisa treinar do zero toda vez

## Estrutura dos arquivos

```
checkpoints/
├── latest_checkpoint.pth              # Checkpoint mais recente
├── checkpoint_epoch_010.pth           # Marcos de 10 em 10 épocas
├── checkpoint_epoch_020.pth
└── ...

checkpoint_backups/                    # Backups manuais
├── latest_checkpoint_backup_20250628_120000.pth
└── ...

models/cgan_optimized/                 # Modelos salvos (formato antigo)
├── best_generator.pth
├── best_discriminator.pth
└── ...
```

## Troubleshooting

### Checkpoint corrompido
Se um checkpoint estiver corrompido:
```bash
# Listar checkpoints disponíveis
python checkpoint_manager.py --list

# Usar um checkpoint anterior
python gan.py --checkpoint-path checkpoints/checkpoint_epoch_020.pth
```

### Erro de compatibilidade
Se houver mudanças no modelo e o checkpoint não for compatível:
```bash
# Iniciar do zero
python gan.py --no-checkpoint
```

### Espaço em disco
Para economizar espaço:
```bash
# Limpeza automática (mantém apenas os importantes)
python checkpoint_manager.py --cleanup

# Limpeza mais agressiva
python checkpoint_manager.py --cleanup --keep-latest 2 --keep-every 20
```
