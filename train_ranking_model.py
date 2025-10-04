import argparse
import torch
from torch.utils.data import DataLoader
import os
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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

    # Initialize TensorBoard writer
    if args.use_tensorboard:
        tb_log_dir = os.path.join(args.output_dir, 'tensorboard_logs')
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")
    else:
        writer = None

    # Load data
    print(f"Loading data for dataset: {args.dataset}")

    if args.dataset == 'chaosnli':
        import json
        import random
        from datasets import Dataset, DatasetDict

        data_path = 'data/raw/chaosnli/chaosNLI_v1.0/chaosNLI_snli.jsonl'
        with open(data_path, 'r') as f:
            lines = f.readlines()
        
        data = [json.loads(line) for line in lines]
        
        random.shuffle(data)
        
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        dataset = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })

        def _calculate_score(label_counter):
            entailment_count = label_counter.get('e')
            neutral_count = label_counter.get('n')
            contradiction_count = label_counter.get('c')

            if entailment_count is None:
                entailment_count = 0
            if neutral_count is None:
                neutral_count = 0
            if contradiction_count is None:
                contradiction_count = 0

            total_annotations = entailment_count + neutral_count + contradiction_count
            if total_annotations == 0:
                return 0.0
            score = (entailment_count * 2 + neutral_count * 1 + contradiction_count * 0) / total_annotations
            return score

        def _format_as_query(premise):
            return f"Premise: {premise}"

        def convert_to_ranking_format(split):
            ranking_examples = []
            for example in dataset[split]:
                premise = example['example']['premise']
                hypothesis = example['example']['hypothesis']
                label_counter = example['label_counter']
                
                score = _calculate_score(label_counter)
                
                ranking_examples.append({
                    'query': _format_as_query(premise),
                    'explanations': [hypothesis],
                    'scores': [score],
                    'num_candidates': 1
                })
            return ranking_examples

        def create_regression_data(ranking_examples):
            regression_examples = []
            for example in ranking_examples:
                query = example['query']
                explanation = example['explanations'][0]
                score = example['scores'][0]
                
                regression_examples.append({
                    'query': query,
                    'explanation': explanation,
                    'score': score,
                    'normalized_score': score / 2.0  # Normalize to [0, 1]
                })
            return regression_examples

        train_data = convert_to_ranking_format('train')
        val_data = convert_to_ranking_format('validation')

    elif args.dataset == 'ds_critique':
        from datasets import load_from_disk

        data_dir = 'data/processed/comprehensive_ranking_dataset'
        full_dataset = load_from_disk(data_dir)

        train_ds = full_dataset['train'].filter(lambda x: x['source'] == 'ds-critique')
        val_ds = full_dataset['validation'].filter(lambda x: x['source'] == 'ds-critique')

        def convert_format(example):
            return {
                'query_id': example['query_id'],
                'question': example['premise'],
                'explanation': example['candidate'],
                'score': example['quality_score'] + 1,  # Convert 0-4 to 1-5
                'critique': f"Quality level: {example['quality_score']}"
            }

        train_converted = train_ds.map(convert_format, remove_columns=train_ds.column_names)
        val_converted = val_ds.map(convert_format, remove_columns=val_ds.column_names)

        dataset = DatasetDict({
            'train': train_converted,
            'validation': val_converted
        })

        def _format_as_query(question):
            if "If:" in question and "why" in question.lower():
                return question
            elif "?" in question:
                return f"Question: {question}\nExplain why the answer is correct."
            else:
                return f"Explain: {question}"

        def convert_to_ranking_format(split):
            ranking_examples = []
            query_groups = {}

            for example in dataset[split]:
                query_id = example.get('query_id', example.get('qid', ''))
                question = example.get('question', example.get('premise', ''))
                explanation = example.get('explanation', example.get('candidate', ''))
                score = example.get('score', 0)

                if not query_id:
                    query_id = question

                if query_id not in query_groups:
                    query_groups[query_id] = {
                        'question': question,
                        'explanations': [],
                        'scores': []
                    }

                query_groups[query_id]['explanations'].append(explanation)
                query_groups[query_id]['scores'].append(score)

            for query_id, group_data in query_groups.items():
                explanations = group_data['explanations']
                scores = group_data['scores']

                if len(explanations) < 2:
                    continue

                ranking_example = {
                    'query': _format_as_query(group_data['question']),
                    'explanations': explanations,
                    'scores': scores,
                    'num_candidates': len(explanations)
                }
                ranking_examples.append(ranking_example)

            return ranking_examples

        def create_regression_data(ranking_examples):
            regression_examples = []
            for example in ranking_examples:
                query = example['query']
                for exp, score in zip(example['explanations'], example['scores']):
                    regression_examples.append({
                        'query': query,
                        'explanation': exp,
                        'score': score,
                        'normalized_score': (score - 1) / 4.0  # Normalize to [0, 1]
                    })
            return regression_examples

        train_data = convert_to_ranking_format('train')
        val_data = convert_to_ranking_format('validation')

    elif args.dataset == 'esnli':
        from datasets import load_from_disk

        data_dir = 'data/raw/e-snli/normalized'
        dataset = load_from_disk(data_dir)

        score_mapping = {
            "entailment": 3,
            "neutral": 2,
            "contradiction": 1
        }

        def _format_as_query(premise):
            return f"Premise: {premise}"

        def convert_to_ranking_format(split):
            ranking_examples = []
            query_groups = {}

            for example in dataset[split]:
                query_id = example.get('query_id')
                premise = example.get('premise')
                explanation = example.get('gold_explanation')
                label = example.get('label')

                if not query_id:
                    query_id = premise

                if query_id not in query_groups:
                    query_groups[query_id] = {
                        'question': premise,
                        'explanations': [],
                        'scores': []
                    }
                
                score = score_mapping.get(label, 0)

                query_groups[query_id]['explanations'].append(explanation)
                query_groups[query_id]['scores'].append(score)

            for query_id, group_data in query_groups.items():
                explanations = group_data['explanations']
                scores = group_data['scores']

                if len(explanations) < 2:
                    continue

                ranking_example = {
                    'query': _format_as_query(group_data['question']),
                    'explanations': explanations,
                    'scores': scores,
                    'num_candidates': len(explanations)
                }
                ranking_examples.append(ranking_example)

            return ranking_examples

        def create_regression_data(ranking_examples):
            regression_examples = []
            for example in ranking_examples:
                query = example['query']
                for exp, score in zip(example['explanations'], example['scores']):
                    regression_examples.append({
                        'query': query,
                        'explanation': exp,
                        'score': score,
                        'normalized_score': (score - 1) / 2.0  # Normalize to [0, 1]
                    })
            return regression_examples

        train_data = convert_to_ranking_format('train')
        val_data = convert_to_ranking_format('validation')

    elif args.dataset == 'stackexchange':
        from datasets import load_dataset

        dataset = load_dataset("lvwerra/stack-exchange-paired")

        def convert_to_ranking_format(split):
            ranking_examples = []
            for example in dataset[split]:
                query = example['question']
                chosen_answer = example['response_j']
                rejected_answer = example['response_k']

                ranking_examples.append({
                    'query': query,
                    'explanations': [chosen_answer, rejected_answer],
                    'scores': [1, 0],
                    'num_candidates': 2
                })
            return ranking_examples

        def create_regression_data(ranking_examples):
            regression_examples = []
            for example in ranking_examples:
                query = example['query']
                chosen_answer = example['explanations'][0]
                rejected_answer = example['explanations'][1]

                regression_examples.append({
                    'query': query,
                    'explanation': chosen_answer,
                    'score': 1,
                    'normalized_score': 1.0
                })
                regression_examples.append({
                    'query': query,
                    'explanation': rejected_answer,
                    'score': 0,
                    'normalized_score': 0.0
                })
            return regression_examples

        train_data = convert_to_ranking_format('train')
        val_data = convert_to_ranking_format('test') # Using test set for validation as there is no validation set

    elif args.dataset == 'dialogue':
        from datasets import load_dataset

        dataset = load_dataset("daily_dialog")

        def _calculate_score(dialogue, topic, emotion, act):
            score = 0
            score += len(dialogue)
            if topic != 0:
                score += 2
            score += len(set(emotion))
            score += len(set(act))
            return float(score)

        def convert_to_ranking_format(split):
            ranking_examples = []
            for example in dataset[split]:
                dialogue = example['dialog']
                topic = example['topic']
                emotion = example['emotion']
                act = example['act']

                score = _calculate_score(dialogue, topic, emotion, act)

                for i in range(1, len(dialogue)):
                    query = "\n".join(dialogue[:i])
                    explanation = dialogue[i]

                    ranking_examples.append({
                        'query': query,
                        'explanations': [explanation],
                        'scores': [score],
                        'num_candidates': 1
                    })
            return ranking_examples

        def create_regression_data(ranking_examples):
            regression_examples = []
            for example in ranking_examples:
                query = example['query']
                explanation = example['explanations'][0]
                score = example['scores'][0]

                regression_examples.append({
                    'query': query,
                    'explanation': explanation,
                    'score': score,
                    'normalized_score': score / 10.0  # Normalize to [0, 1]
                })
            return regression_examples

        train_data = convert_to_ranking_format('train')
        val_data = convert_to_ranking_format('validation')

    else:
        raise ValueError(f"Dataset {args.dataset} not supported in this simplified script.")

    print(f"Train examples: {len(train_data)}")
    print(f"Val examples: {len(val_data)}")

    # Convert to regression format for training
    train_regression = create_regression_data(train_data)
    val_regression = create_regression_data(val_data)

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
    elif torch.backends.mps.is_available():
        model = model.to('mps')
        print("Using MPS (Metal Performance Shaders)")
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

    global_step = 0
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            scores = batch['scores']

            if args.use_cuda and torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                scores = scores.cuda()
            elif torch.backends.mps.is_available():
                input_ids = input_ids.to('mps')
                attention_mask = attention_mask.to('mps')
                scores = scores.to('mps')

            # Forward pass
            pred_scores = model(input_ids, attention_mask)

            # MSE loss for regression
            loss = torch.nn.functional.mse_loss(pred_scores, scores)

            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Only update weights every gradient_accumulation_steps
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Log to TensorBoard
                if writer is not None:
                    writer.add_scalar('Loss/train_batch', loss.item() * args.gradient_accumulation_steps, global_step)
                    writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

            train_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

            progress_bar.set_postfix({'loss': loss.item() * args.gradient_accumulation_steps})

        # Clear any remaining gradients
        if num_batches % args.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = train_loss / num_batches

        # Log epoch average to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch + 1)

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

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Metrics/val_ndcg@1', val_results.get('ndcg@1', 0), epoch + 1)
            writer.add_scalar('Metrics/val_ndcg@3', val_results.get('ndcg@3', 0), epoch + 1)
            writer.add_scalar('Metrics/val_ndcg@5', val_results.get('ndcg@5', 0), epoch + 1)
            writer.add_scalar('Metrics/val_map', val_results.get('map', 0), epoch + 1)
            writer.add_scalar('Metrics/val_mrr', val_results.get('mrr', 0), epoch + 1)
            writer.add_scalar('Metrics/val_kendall_tau', val_results.get('kendall_tau', 0), epoch + 1)
            writer.add_scalar('Metrics/val_spearman', val_results.get('spearman', 0), epoch + 1)

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

            print(f"Saved best model with val score: {val_score:.4f}\n")

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
    print(f"Saved final model to {final_path}")

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to {tb_log_dir}")
        print(f"   View with: tensorboard --logdir={tb_log_dir}")

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
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='models/ranking_reward',
                       help='Output directory for models (default: models/ranking_reward)')
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--use_tensorboard', action='store_true',
                       help='Use TensorBoard for logging')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                           help='Number of gradient accumulation steps (default: 1)')
    parser.add_argument('--dataset', type=str, default='ds_critique',
                           help='Dataset to use for training (ds_critique, esnli, chaosnli)')
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
    print(f"Use TensorBoard: {args.use_tensorboard}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print("="*60 + "\n")

    train_ranking_reward_model(args)

    print("\nTraining complete!")
