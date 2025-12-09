"""
Experiment configurations for Information Gain ablation study
"""

# Base configuration (shared across all experiments)
BASE_CONFIG = {
    'env': 'WordleEnv100-v0',
    'batch_size': 64,
    'lr': 0.0002,
    'gamma': 0.99,
    'accelerator': 'gpu',
    'max_steps': 781,  # 50K episodes
    'log_every_n_steps': 50,
}

# Random seeds for reproducibility
SEEDS = [42, 123, 456]

# Information gain weight ablations
INFO_GAIN_WEIGHTS = [0.0, 0.05, 0.10, 0.20]  # 0.0 is baseline

# Generate all experiment configurations
EXPERIMENTS = []

for weight in INFO_GAIN_WEIGHTS:
    for seed in SEEDS:
        exp_name = f"{'baseline' if weight == 0.0 else f'infogain_{weight:.2f}'}_seed{seed}"
        
        config = BASE_CONFIG.copy()
        config.update({
            'seed': seed,
            'info_gain_weight': weight,
            'run_name': exp_name,
            'checkpoint_dir': f'checkpoints/{exp_name}',
        })
        
        EXPERIMENTS.append(config)

print(f"Total experiments: {len(EXPERIMENTS)}")
print("\nExperiment list:")
for i, exp in enumerate(EXPERIMENTS):
    print(f"{i+1:2d}. {exp['run_name']}")