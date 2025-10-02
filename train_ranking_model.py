import argparse
import torch
from torch.utils.data import DataLoader
import os
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from ds_critique_loader import DSCritiqueBankLoader
from ranking_models import RankingRewardModel
from ranking_evaluator import RankingEvaluator

def train_ranking_reward_model(args):
    """Train a ranking-based reward model on DS_Critique_Bank"""

    # Initialize wandb if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project="ranking-reward-model", config=vars(args))
        except ImportError:
            print("wandb not installed, skipping logging")
            args.use_wandb = False

    # Load data
    print("Loading data...")
    loader = DSCritiqueBankLoader()
    dataset = loader.load_dataset()

    train_data = loader.convert_to_ranking_format('train')
    val_data = loader.convert_to_ranking_format('validation' if 'validation' in dataset else 'test')

    print(f"Train examples: {len(train_data)}")
    print(f"Val examples: {len(val_data)}")

    # Convert to regression format for training
    train_regression = loader.create_regression_data(train_data)
    val_regression = loader.create_regression_data(val_data)

    print(f"Train regression examples: {len(train_regression)}")
    print(f"Val regression examples: {len(val_regression)}")

    # Initialize model
    print(f"Initializing model: {args.base_model}")
    model = RankingRewardModel(
        base_model=args.base_model,
        output_mode="regression",
        dropout=args.dropout
    )

    if args.use_cuda and torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    # Create datasets and dataloaders
    train_dataset = RankingDataset(train_regression, model.tokenizer)
    val_dataset = RankingDataset(val_regression, model.tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    best_val_score = float('-inf')
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nStarting training for {args.num_epochs} epochs...")

    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch in progress_bar:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            scores = batch['scores']

            if args.use_cuda and torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                scores = scores.cuda()

            # Forward pass
            pred_scores = model(input_ids, attention_mask)

            # MSE loss for regression
            loss = torch.nn.functional.mse_loss(pred_scores, scores)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / num_batches

        # Validation
        print("\nRunning validation...")
        model.eval()
        evaluator = RankingEvaluator()
        val_results = evaluator.evaluate(model, val_data)

        # Log results
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"{'='*60}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val NDCG@1: {val_results.get('ndcg@1', 0):.4f}")
        print(f"Val NDCG@3: {val_results.get('ndcg@3', 0):.4f}")
        print(f"Val NDCG@5: {val_results.get('ndcg@5', 0):.4f}")
        print(f"Val MAP: {val_results.get('map', 0):.4f}")
        print(f"Val Spearman: {val_results.get('spearman', 0):.4f}")
        print(f"{'='*60}\n")

        if args.use_wandb:
            import wandb
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
            }
            log_dict.update({f'val_{k}': v for k, v in val_results.items() if not k.endswith('_std')})
            wandb.log(log_dict)

        # Save best model
        val_score = val_results.get('ndcg@5', val_results.get('spearman', 0))
        if val_score > best_val_score:
            best_val_score = val_score

            save_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': val_score,
                'val_results': val_results
            }, save_path)

            print(f"✅ Saved best model with val score: {val_score:.4f}\n")

    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'base_model': args.base_model,
            'output_mode': 'regression',
            'dropout': args.dropout
        }
    }, final_path)
    print(f"✅ Saved final model to {final_path}")

    return model


class RankingDataset(torch.utils.data.Dataset):
    """Dataset for ranking reward model training"""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Combine query and explanation
        text = f"{item['query']} {self.tokenizer.sep_token} {item['explanation']}"

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'scores': torch.tensor(item['normalized_score'], dtype=torch.float)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ranking reward model")
    parser.add_argument('--base_model', type=str, default='bert-base-uncased',
                       help='Base model to use (default: bert-base-uncased)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='models/ranking_reward',
                       help='Output directory for models (default: models/ranking_reward)')
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')

    args = parser.parse_args()

    print("="*60)
    print("RANKING REWARD MODEL TRAINING")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"Output dir: {args.output_dir}")
    print(f"Use CUDA: {args.use_cuda}")
    print(f"Use wandb: {args.use_wandb}")
    print("="*60 + "\n")

    train_ranking_reward_model(args)

    print("\n✅ Training complete!")
