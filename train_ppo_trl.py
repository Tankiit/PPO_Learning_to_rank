"""
TRL-based PPO Training for Ranking (Digital Socrates Style)

This implements PPO using the TRL (Transformer Reinforcement Learning) library,
similar to the approach used in Digital Socrates paper.

Digital Socrates Results (Llama-2-7b-chat-hf):
- Baseline: BLEU=0.0051, METEOR=0.1965, ROUGE=0.1285, BERTScore=0.8588
- PPO TRL (10k): BLEU=0.0060, METEOR=0.2020, ROUGE=0.1384, BERTScore=0.8618
- PPO TRL (20k): BLEU=0.0072, METEOR=0.1998, ROUGE=0.1528, BERTScore=0.8677

Our implementation aims to replicate these results with ranking reward model.
"""

import argparse
import torch
import os
from typing import Dict, List
from tqdm import tqdm
import json

try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl import create_reference_model
    from trl.core import LengthSampler
    TRL_AVAILABLE = True
except ImportError:
    print("⚠️  TRL library not installed. Install with: pip install trl")
    TRL_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import numpy as np

from ranking_models import RankingRewardModel


class TRLPPORankingTrainer:
    """
    PPO trainer using TRL library for ranking explanations
    Similar to Digital Socrates approach
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available()
                                   else 'mps' if torch.backends.mps.is_available()
                                   else 'cpu')

        print(f"Using device: {self.device}")

        # Load reward model (your trained ranking model)
        print("\nLoading reward model...")
        self.reward_model = self.load_reward_model(args.reward_model_path)

        # Load base LLM for generation
        print(f"\nLoading base model: {args.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with value head for PPO
        if TRL_AVAILABLE:
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
                device_map="auto" if args.use_cuda else None
            )

            # Create reference model (frozen copy)
            self.ref_model = create_reference_model(self.model)

            print("Model and reference model loaded successfully")
        else:
            raise ImportError("TRL library required. Install with: pip install trl")

        # PPO configuration (matching Digital Socrates)
        self.ppo_config = PPOConfig(
            model_name=args.base_model,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            mini_batch_size=args.mini_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optimize_cuda_cache=True,
            early_stopping=args.early_stopping,
            target_kl=args.target_kl,
            ppo_epochs=args.ppo_epochs,
            seed=args.seed,
            init_kl_coef=args.init_kl_coef,
            adap_kl_ctrl=True,
        )

        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        print(f"\n✅ TRL PPO Trainer initialized")

    def load_reward_model(self, path: str) -> RankingRewardModel:
        """Load trained ranking reward model"""
        checkpoint = torch.load(path, map_location=self.device)

        if 'config' in checkpoint:
            config = checkpoint['config']
            base_model = config.get('base_model', 'bert-base-uncased')
            dropout = config.get('dropout', 0.1)
        else:
            base_model = 'bert-base-uncased'
            dropout = 0.1

        model = RankingRewardModel(
            base_model=base_model,
            output_mode="regression",
            dropout=dropout
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"✅ Reward model loaded from {path}")
        return model

    def load_dataset(self):
        """Load dataset for training"""
        print(f"\nLoading dataset: {self.args.dataset}")

        if self.args.dataset == 'ds_critique':
            data_dir = '/Users/tanmoy/research/PPO_learning_to_rank/PPO_Learning_to_rank/data/processed/comprehensive_ranking_dataset'
            if not os.path.exists(data_dir):
                data_dir = 'data/processed/comprehensive_ranking_dataset'

            full_dataset = load_from_disk(data_dir)
            train_ds = full_dataset['train'].filter(lambda x: x['source'] == 'ds-critique')

            # Convert to prompt format
            def format_prompt(example):
                return {
                    'query': f"Question: {example['premise']}\nExplain why the answer is correct.",
                    'reference': example['candidate'],
                    'quality_score': example['quality_score']
                }

            train_prompts = train_ds.map(format_prompt)

            # Sample for training
            if self.args.max_train_samples > 0:
                train_prompts = train_prompts.select(range(min(self.args.max_train_samples, len(train_prompts))))

            return train_prompts

        else:
            raise ValueError(f"Dataset {self.args.dataset} not supported")

    def compute_reward(self, query: str, generated_text: str) -> float:
        """
        Compute reward using the trained ranking model

        Args:
            query: The question/prompt
            generated_text: Model's generated explanation

        Returns:
            reward: Scalar reward (0-1)
        """
        # Use ranking model to score the generated explanation
        with torch.no_grad():
            scores = self.reward_model.rank_explanations(
                query,
                [generated_text],
                return_scores=True
            )
            reward = scores[0]  # Single score

        # Scale reward (Digital Socrates uses reward shaping)
        if self.args.reward_scaling == 'linear':
            reward = reward
        elif self.args.reward_scaling == 'exponential':
            # Exponential scaling: emphasize high-quality explanations
            reward = np.exp(8.0 * (reward - 0.5))  # ES 8.0 from Digital Socrates

        return float(reward)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate explanations for a batch of prompts"""
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_prompt_length
        ).to(self.device)

        # Generate
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_generation_length,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode (remove prompt)
        prompt_lengths = inputs['input_ids'].shape[1]
        generated_texts = [
            self.tokenizer.decode(output[prompt_lengths:], skip_special_tokens=True)
            for output in gen_outputs
        ]

        return generated_texts

    def train(self):
        """Train with PPO using TRL"""
        print("\n" + "="*80)
        print("STARTING TRL PPO TRAINING (Digital Socrates Style)")
        print("="*80)

        # Load dataset
        dataset = self.load_dataset()
        print(f"Training on {len(dataset)} examples")

        # Training metrics
        all_rewards = []
        all_losses = []
        epoch_metrics = []

        # Training loop
        for epoch in range(self.args.num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            print(f"{'='*80}")

            epoch_rewards = []
            epoch_losses = []

            # Process in batches
            for i in tqdm(range(0, len(dataset), self.args.batch_size), desc=f"Epoch {epoch+1}"):
                batch_data = dataset[i:i + self.args.batch_size]
                queries = batch_data['query']

                # Generate responses
                response_texts = self.generate_batch(queries)

                # Tokenize for PPO
                query_tensors = [
                    self.tokenizer.encode(q, return_tensors="pt").squeeze().to(self.device)
                    for q in queries
                ]
                response_tensors = [
                    self.tokenizer.encode(r, return_tensors="pt").squeeze().to(self.device)
                    for r in response_texts
                ]

                # Compute rewards
                rewards = [
                    torch.tensor(self.compute_reward(q, r))
                    for q, r in zip(queries, response_texts)
                ]

                # PPO step
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)

                # Track metrics
                epoch_rewards.extend([r.item() for r in rewards])
                if 'ppo/loss/total' in stats:
                    epoch_losses.append(stats['ppo/loss/total'])

                # Log batch stats
                if i % (self.args.batch_size * 10) == 0:
                    mean_reward = np.mean(epoch_rewards[-len(queries):])
                    print(f"  Batch {i//self.args.batch_size}: Mean Reward = {mean_reward:.4f}")

            # Epoch summary
            mean_epoch_reward = np.mean(epoch_rewards)
            mean_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0

            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Mean Reward: {mean_epoch_reward:.4f}")
            print(f"  Mean Loss: {mean_epoch_loss:.4f}")

            # Save metrics
            epoch_metrics.append({
                'epoch': epoch + 1,
                'mean_reward': mean_epoch_reward,
                'mean_loss': mean_epoch_loss,
                'std_reward': np.std(epoch_rewards)
            })

            all_rewards.extend(epoch_rewards)
            all_losses.extend(epoch_losses)

            # Save checkpoint
            if (epoch + 1) % self.args.save_frequency == 0:
                self.save_checkpoint(epoch + 1, epoch_metrics)

        # Final save
        self.save_checkpoint(self.args.num_epochs, epoch_metrics, final=True)

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Final Mean Reward: {np.mean(all_rewards):.4f}")
        print(f"Output: {self.args.output_dir}")

    def save_checkpoint(self, epoch: int, metrics: List[Dict], final: bool = False):
        """Save model checkpoint and metrics"""
        os.makedirs(self.args.output_dir, exist_ok=True)

        # Save model
        if final:
            save_path = os.path.join(self.args.output_dir, 'final_model')
        else:
            save_path = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch}')

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save metrics
        metrics_path = os.path.join(self.args.output_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="TRL PPO Training for Ranking (Digital Socrates Style)")

    # Model
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                       help='Base LLM for generation (Llama, Mistral, etc)')
    parser.add_argument('--reward_model_path', type=str, required=True,
                       help='Path to trained ranking reward model')

    # Training
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for PPO')
    parser.add_argument('--mini_batch_size', type=int, default=1,
                       help='Mini batch size for PPO updates')
    parser.add_argument('--learning_rate', type=float, default=1.41e-5,
                       help='Learning rate (Digital Socrates uses 1.41e-5)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_train_samples', type=int, default=10000,
                       help='Max training samples (10k or 20k like Digital Socrates)')

    # PPO hyperparameters
    parser.add_argument('--ppo_epochs', type=int, default=4,
                       help='Number of PPO epochs per batch')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--target_kl', type=float, default=0.1,
                       help='Target KL divergence')
    parser.add_argument('--init_kl_coef', type=float, default=0.2,
                       help='Initial KL coefficient')

    # Reward shaping (Digital Socrates style)
    parser.add_argument('--reward_scaling', type=str, default='exponential',
                       choices=['linear', 'exponential'],
                       help='Reward scaling: exponential uses ES 8.0 from Digital Socrates')

    # Generation
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_generation_length', type=int, default=128)

    # Dataset
    parser.add_argument('--dataset', type=str, default='ds_critique')

    # Output
    parser.add_argument('--output_dir', type=str, default='models/ppo_trl_ranking')
    parser.add_argument('--save_frequency', type=int, default=1)

    # Hardware
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use FP16 for faster training')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if not TRL_AVAILABLE:
        print("\n❌ TRL library not installed!")
        print("Install with: pip install trl")
        print("Or: pip install 'trl[peft]' for LoRA support")
        return

    # Print configuration
    print("="*80)
    print("TRL PPO RANKING TRAINER (Digital Socrates Style)")
    print("="*80)
    print(f"Base Model: {args.base_model}")
    print(f"Reward Model: {args.reward_model_path}")
    print(f"Training Samples: {args.max_train_samples}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Reward Scaling: {args.reward_scaling}")
    print("="*80 + "\n")

    # Train
    trainer = TRLPPORankingTrainer(args)
    trainer.train()

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
