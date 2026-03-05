"""
Hybrid Training: Supervised Pre-training + PPO Fine-tuning

This script combines the best of both approaches:
1. Phase 1: Fast supervised pre-training with ranking losses
2. Phase 2: PPO fine-tuning for better score discrimination

Usage:
    python train_hybrid_ranking.py --dataset ds_critique --num_pretrain_epochs 30 --num_ppo_epochs 10
"""

import argparse
import torch
import os
from datetime import datetime

from train_ranking_model import train_ranking_reward_model
from train_ppo_ranking import train_ppo_ranking_model, PPORankingPolicy, PPOValueNetwork
from ranking_models import RankingRewardModel


def train_hybrid_ranking_model(args):
    """
    Train a hybrid ranking model with two phases:
    1. Supervised pre-training (fast, stable)
    2. PPO fine-tuning (better discrimination)
    """

    print("="*80)
    print("HYBRID RANKING MODEL TRAINING")
    print("="*80)
    print(f"Phase 1: Supervised pre-training ({args.num_pretrain_epochs} epochs)")
    print(f"Phase 2: PPO fine-tuning ({args.num_ppo_epochs} epochs)")
    print(f"Loss function: {args.loss_function}")
    print("="*80 + "\n")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(args.output_dir, f"hybrid_{timestamp}")
    pretrain_dir = os.path.join(base_output_dir, "phase1_supervised")
    ppo_dir = os.path.join(base_output_dir, "phase2_ppo")

    os.makedirs(pretrain_dir, exist_ok=True)
    os.makedirs(ppo_dir, exist_ok=True)

    # =========================================================================
    # PHASE 1: Supervised Pre-training
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 1: SUPERVISED PRE-TRAINING")
    print("="*80 + "\n")

    # Create args for supervised training
    pretrain_args = argparse.Namespace(
        base_model=args.base_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_pretrain_epochs,
        dropout=args.dropout,
        output_dir=pretrain_dir,
        use_cuda=args.use_cuda,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataset=args.dataset,
        num_workers=args.num_workers,
        val_frequency=args.val_frequency,
        val_subset_size=args.val_subset_size,
        use_quantization=args.use_quantization,
        loss_function=args.loss_function
    )

    # Train supervised model
    supervised_model = train_ranking_reward_model(pretrain_args)

    print("\n✅ Phase 1 complete! Supervised model saved to:", pretrain_dir)

    # Save pretrained model path
    pretrained_path = os.path.join(pretrain_dir, 'best_model.pt')
    if not os.path.exists(pretrained_path):
        pretrained_path = os.path.join(pretrain_dir, 'final_model.pt')

    # =========================================================================
    # PHASE 2: PPO Fine-tuning
    # =========================================================================
    print("\n" + "="*80)
    print("PHASE 2: PPO FINE-TUNING")
    print("="*80 + "\n")

    # Load pretrained weights into PPO policy
    print(f"Loading pretrained weights from {pretrained_path}...")

    # Create PPO args
    ppo_args = argparse.Namespace(
        base_model=args.base_model,
        batch_size=args.batch_size,
        learning_rate=args.ppo_learning_rate,
        num_epochs=args.num_ppo_epochs,
        updates_per_epoch=args.updates_per_epoch,
        dropout=args.dropout,
        output_dir=ppo_dir,
        use_cuda=args.use_cuda,
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        dataset=args.dataset,
        num_workers=args.num_workers,
        val_frequency=args.val_frequency,
        save_frequency=args.save_frequency,
        use_quantization=args.use_quantization,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs
    )

    # Initialize PPO policy with pretrained weights
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available()
                         else 'mps' if torch.backends.mps.is_available()
                         else 'cpu')

    # Load pretrained checkpoint
    checkpoint = torch.load(pretrained_path, map_location=device)

    # Initialize PPO policy
    policy = PPORankingPolicy(
        base_model=args.base_model,
        dropout=args.dropout,
        use_quantization=args.use_quantization
    ).to(device)

    # Transfer encoder weights from supervised model to PPO policy
    print("Transferring encoder weights to PPO policy...")
    policy_state_dict = policy.state_dict()
    pretrained_state_dict = checkpoint['model_state_dict']

    # Map supervised model weights to PPO policy encoder
    for key in pretrained_state_dict:
        if key.startswith('encoder'):
            ppo_key = f'encoder_model.{key}'
            if ppo_key in policy_state_dict:
                policy_state_dict[ppo_key] = pretrained_state_dict[key]

    policy.load_state_dict(policy_state_dict, strict=False)
    print("✓ Encoder weights transferred successfully")

    # Initialize value network (from scratch or pretrained)
    value_net = PPOValueNetwork(
        base_model=args.base_model,
        dropout=args.dropout,
        use_quantization=args.use_quantization
    ).to(device)

    # Optionally transfer encoder weights to value network too
    if args.init_value_from_pretrain:
        print("Transferring encoder weights to value network...")
        value_state_dict = value_net.state_dict()
        for key in pretrained_state_dict:
            if key.startswith('encoder'):
                value_key = f'encoder_model.{key}'
                if value_key in value_state_dict:
                    value_state_dict[value_key] = pretrained_state_dict[key]
        value_net.load_state_dict(value_state_dict, strict=False)
        print("✓ Value network encoder weights transferred")

    # Save initialized PPO models
    init_path = os.path.join(ppo_dir, 'ppo_initialized_from_pretrain.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'pretrained_from': pretrained_path,
        'config': {
            'base_model': args.base_model,
            'dropout': args.dropout
        }
    }, init_path)
    print(f"✓ Initialized PPO models saved to {init_path}")

    # Train PPO with pretrained initialization
    print("\nStarting PPO fine-tuning with pretrained initialization...")
    ppo_policy, ppo_value = train_ppo_ranking_model(ppo_args)

    print("\n✅ Phase 2 complete! PPO fine-tuned model saved to:", ppo_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("HYBRID TRAINING COMPLETE")
    print("="*80)
    print(f"\nPhase 1 (Supervised): {pretrain_dir}")
    print(f"  - best_model.pt or final_model.pt")
    print(f"\nPhase 2 (PPO Fine-tuned): {ppo_dir}")
    print(f"  - ppo_initialized_from_pretrain.pt (initial)")
    print(f"  - ppo_final_model.pt (fine-tuned)")
    print(f"\nTo compare models:")
    print(f"  python compare_supervised_vs_ppo.py \\")
    print(f"      --supervised_model {pretrained_path} \\")
    print(f"      --ppo_model {os.path.join(ppo_dir, 'ppo_final_model.pt')}")
    print("="*80 + "\n")

    return supervised_model, ppo_policy, ppo_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid ranking model (supervised + PPO)")

    # Model architecture
    parser.add_argument('--base_model', type=str, default='bert-base-uncased')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_quantization', action='store_true')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate for supervised pre-training')
    parser.add_argument('--ppo_learning_rate', type=float, default=1e-5,
                       help='Learning rate for PPO fine-tuning (usually lower)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    # Phase 1: Supervised pre-training
    parser.add_argument('--num_pretrain_epochs', type=int, default=30,
                       help='Number of epochs for supervised pre-training')
    parser.add_argument('--loss_function', type=str, default='listnet',
                       choices=['mse', 'listnet', 'ranknet', 'listmle', 'approxndcg'],
                       help='Loss function for supervised training (listnet recommended)')

    # Phase 2: PPO fine-tuning
    parser.add_argument('--num_ppo_epochs', type=int, default=10,
                       help='Number of epochs for PPO fine-tuning')
    parser.add_argument('--updates_per_epoch', type=int, default=100)
    parser.add_argument('--init_value_from_pretrain', action='store_true',
                       help='Initialize value network encoder from pretrained model')

    # PPO hyperparameters
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--ppo_epochs', type=int, default=4)

    # Dataset and data loading
    parser.add_argument('--dataset', type=str, default='ds_critique')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_frequency', type=int, default=5)
    parser.add_argument('--val_subset_size', type=int, default=0)
    parser.add_argument('--save_frequency', type=int, default=5)

    # Output and logging
    parser.add_argument('--output_dir', type=str, default='models/hybrid_ranking')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true')

    args = parser.parse_args()

    train_hybrid_ranking_model(args)

    print("\n🎉 Hybrid training complete!")
