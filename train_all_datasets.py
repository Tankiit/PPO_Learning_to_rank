"""
Train ranking models across all available datasets

Supports:
- ChaosNLI
- DS Critique Bank
- e-SNLI
- Stack Exchange
- Daily Dialog
"""

import argparse
import os
import sys
from typing import List, Dict
from datasets import DatasetDict

# Import all loaders
from chaosnli_loader import ChaosNLILoader
from ds_critique_loader import DSCritiqueBankLoader
from esnli_loader import ESNLILoader
from stackexchange_loader import StackExchangeLoader
from dialogue_loader import DialogueLoader

# Import training
from train_ranking_model import train_ranking_reward_model


class MultiDatasetTrainer:
    """Train ranking models across multiple datasets"""

    DATASETS = {
        'chaosnli': ChaosNLILoader,
        'ds_critique': DSCritiqueBankLoader,
        'esnli': ESNLILoader,
        'stackexchange': StackExchangeLoader,
        'dialogue': DialogueLoader
    }

    def __init__(self, args):
        self.args = args

    def get_loader(self, dataset_name: str):
        """Get appropriate loader for dataset"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(self.DATASETS.keys())}")

        loader_class = self.DATASETS[dataset_name]

        # Initialize with appropriate arguments
        if dataset_name == 'ds_critique':
            return loader_class(use_local=True)
        elif dataset_name == 'esnli':
            return loader_class(use_local=True)
        else:
            return loader_class()

    def prepare_dataset(self, dataset_name: str) -> Dict:
        """Load and prepare dataset for training"""
        print(f"\n{'='*80}")
        print(f"Preparing {dataset_name.upper()} dataset")
        print(f"{'='*80}\n")

        # Get loader
        loader = self.get_loader(dataset_name)

        # Load dataset
        try:
            dataset = loader.load_dataset()
        except Exception as e:
            print(f"⚠️  Failed to load {dataset_name}: {e}")
            return None

        # Convert to ranking format
        try:
            train_ranking = loader.convert_to_ranking_format('train')

            # Handle validation split (may not exist)
            try:
                val_ranking = loader.convert_to_ranking_format('validation')
            except:
                # Create validation from train split
                split_idx = int(len(train_ranking) * 0.9)
                val_ranking = train_ranking[split_idx:]
                train_ranking = train_ranking[:split_idx]
                print(f"ℹ️  Created validation split from train (90/10 split)")

            # Handle test split (may not exist)
            try:
                test_ranking = loader.convert_to_ranking_format('test')
            except:
                test_ranking = val_ranking[:min(100, len(val_ranking))]

            print(f"✅ Loaded {dataset_name}:")
            print(f"   Train: {len(train_ranking)} examples")
            print(f"   Val: {len(val_ranking)} examples")
            print(f"   Test: {len(test_ranking)} examples")

            return {
                'train': train_ranking,
                'validation': val_ranking,
                'test': test_ranking,
                'name': dataset_name
            }
        except Exception as e:
            print(f"⚠️  Failed to convert {dataset_name} to ranking format: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_on_dataset(self, dataset_name: str):
        """Train model on a specific dataset"""
        # Prepare dataset
        data = self.prepare_dataset(dataset_name)
        if data is None:
            print(f"❌ Skipping {dataset_name}")
            return None

        # Create output directory for this dataset
        output_dir = os.path.join(self.args.output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Prepare args for training
        train_args = argparse.Namespace(
            base_model=self.args.base_model,
            max_length=self.args.max_length,
            batch_size=self.args.batch_size,
            learning_rate=self.args.learning_rate,
            num_epochs=self.args.num_epochs,
            dropout=self.args.dropout,
            output_dir=output_dir,
            loss_function=self.args.loss_function,
            use_cuda=self.args.use_cuda,
            use_fp16=self.args.use_fp16,
            save_every=self.args.save_every,
            eval_every=self.args.eval_every,
            early_stopping_patience=self.args.early_stopping_patience,
            use_wandb=False,
            use_tensorboard=False,
            use_pretokenization=True,
            num_workers=4
        )

        print(f"\n{'='*80}")
        print(f"Training on {dataset_name.upper()}")
        print(f"Output: {output_dir}")
        print(f"Loss function: {self.args.loss_function}")
        print(f"{'='*80}\n")

        # Train
        try:
            # Pass dataset via args
            train_args.train_data = data['train']
            train_args.val_data = data['validation']
            train_args.test_data = data['test']

            model, metrics = train_ranking_reward_model(train_args)
            print(f"✅ Training completed for {dataset_name}")
            print(f"   Best NDCG@5: {metrics.get('best_ndcg@5', 'N/A')}")

            return {
                'dataset': dataset_name,
                'metrics': metrics,
                'output_dir': output_dir
            }
        except Exception as e:
            print(f"❌ Training failed for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_all(self):
        """Train on all specified datasets"""
        datasets = self.args.datasets if self.args.datasets != ['all'] else list(self.DATASETS.keys())

        print(f"\n{'='*80}")
        print(f"MULTI-DATASET TRAINING")
        print(f"{'='*80}")
        print(f"Datasets: {', '.join(datasets)}")
        print(f"Loss function: {self.args.loss_function}")
        print(f"Base model: {self.args.base_model}")
        print(f"Epochs: {self.args.num_epochs}")
        print(f"{'='*80}\n")

        results = []

        for dataset_name in datasets:
            result = self.train_on_dataset(dataset_name)
            if result:
                results.append(result)

        # Summary
        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}\n")

        for result in results:
            dataset = result['dataset']
            metrics = result['metrics']
            output_dir = result['output_dir']

            print(f"{dataset.upper()}:")
            print(f"  Output: {output_dir}")
            print(f"  Best NDCG@5: {metrics.get('best_ndcg@5', 'N/A'):.4f}")
            print(f"  Best Loss: {metrics.get('best_loss', 'N/A'):.4f}")
            print()

        print(f"✅ All training complete! Results saved to {self.args.output_dir}/")

        return results


def main():
    parser = argparse.ArgumentParser(description="Train ranking models across multiple datasets")

    # Dataset selection
    parser.add_argument('--datasets', nargs='+', default=['all'],
                       choices=['all', 'chaosnli', 'ds_critique', 'esnli', 'stackexchange', 'dialogue'],
                       help='Datasets to train on (default: all)')

    # Model config
    parser.add_argument('--base_model', type=str, default='bert-base-uncased',
                       help='Base model for ranking')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    # Training config
    parser.add_argument('--loss_function', type=str, default='listnet',
                       choices=['mse', 'listnet', 'ranknet', 'listmle', 'approxndcg'],
                       help='Loss function for ranking')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of epochs per dataset')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Early stopping patience')

    # Evaluation
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')

    # Hardware
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    parser.add_argument('--use_fp16', action='store_true',
                       help='Use mixed precision training')

    # Output
    parser.add_argument('--output_dir', type=str, default='models/multi_dataset_ranking',
                       help='Base output directory (subdirs created per dataset)')

    args = parser.parse_args()

    # Create trainer
    trainer = MultiDatasetTrainer(args)

    # Train on all datasets
    results = trainer.train_all()

    print("\n✅ Multi-dataset training complete!")


if __name__ == "__main__":
    main()
