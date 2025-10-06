import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import os
import numpy as np
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import json

from ranking_models import RankingRewardModel
from ranking_evaluator import RankingEvaluator


class PPORankingPolicy(nn.Module):
    """
    PPO policy network for ranking explanations
    Uses same encoder as supervised model but outputs action probabilities
    """

    def __init__(self, base_model="bert-base-uncased", dropout=0.1, use_quantization=False):
        super().__init__()

        # Reuse the ranking model encoder
        self.encoder_model = RankingRewardModel(
            base_model=base_model,
            output_mode="regression",
            dropout=dropout,
            use_quantization=use_quantization
        )

        # Get hidden size
        hidden_size = self.encoder_model.encoder.config.hidden_size

        # Policy head outputs logits for ranking actions
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # Score for each explanation
        )

        self.tokenizer = self.encoder_model.tokenizer

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass through encoder
        Returns: logits for action selection
        """
        # Get encoder outputs
        if hasattr(self.encoder_model, 'model_type') and self.encoder_model.model_type == 'decoder':
            outputs = self.encoder_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs = self.encoder_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        # Get pooled representation
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_hidden / sum_mask

        # Get policy logits
        logits = self.policy_head(pooled_output)
        return logits.squeeze(-1)

    def get_action_probs(self, logits):
        """Convert logits to action probabilities using softmax"""
        return F.softmax(logits, dim=-1)

    def evaluate_actions(self, input_ids, attention_mask, actions):
        """
        Evaluate log probs and entropy for given actions
        actions: indices of selected explanations (batch_size,)
        """
        logits = self.forward(input_ids, attention_mask)
        probs = self.get_action_probs(logits)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy


class PPOValueNetwork(nn.Module):
    """
    PPO value network (critic) for estimating state values
    """

    def __init__(self, base_model="bert-base-uncased", dropout=0.1, use_quantization=False):
        super().__init__()

        # Reuse encoder
        self.encoder_model = RankingRewardModel(
            base_model=base_model,
            output_mode="regression",
            dropout=dropout,
            use_quantization=use_quantization
        )

        hidden_size = self.encoder_model.encoder.config.hidden_size

        # Value head outputs single value estimate
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.tokenizer = self.encoder_model.tokenizer

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Returns value estimate for the state"""
        # Get encoder outputs
        if hasattr(self.encoder_model, 'model_type') and self.encoder_model.model_type == 'decoder':
            outputs = self.encoder_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs = self.encoder_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        # Get pooled representation
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_hidden / sum_mask

        value = self.value_head(pooled_output)
        return value.squeeze(-1)


class PPORankingTrainer:
    """
    PPO trainer for ranking tasks
    Implements full PPO algorithm with clipping, GAE, and entropy bonus
    """

    def __init__(self, policy, value_net, args, device):
        self.policy = policy
        self.value_net = value_net
        self.args = args
        self.device = device

        # Optimizers
        self.policy_optimizer = AdamW(policy.parameters(), lr=args.learning_rate)
        self.value_optimizer = AdamW(value_net.parameters(), lr=args.learning_rate)

        # Schedulers
        total_steps = args.num_epochs * args.updates_per_epoch
        self.policy_scheduler = get_linear_schedule_with_warmup(
            self.policy_optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        self.value_scheduler = get_linear_schedule_with_warmup(
            self.value_optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # PPO hyperparameters
        self.clip_epsilon = args.clip_epsilon
        self.entropy_coef = args.entropy_coef
        self.value_coef = args.value_coef
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.ppo_epochs = args.ppo_epochs

    def compute_gae(self, rewards, values, dones, next_values):
        """
        Compute Generalized Advantage Estimation (GAE)
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32)

    def compute_ppo_loss(self, old_log_probs, log_probs, advantages, entropy):
        """
        Compute PPO clipped loss
        """
        # Compute ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Compute clipped surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus
        entropy_loss = -self.entropy_coef * entropy.mean()

        return policy_loss + entropy_loss

    def compute_rewards_from_ranking(self, query, explanations, gold_scores, pred_ranking):
        """
        Compute reward based on ranking quality (NDCG-based)
        pred_ranking: list of indices in ranked order
        """
        # Convert ranking to scores
        pred_scores = np.zeros(len(explanations))
        for rank, idx in enumerate(pred_ranking):
            pred_scores[idx] = 1.0 / (rank + 1)  # Inverse rank as score

        # Compute NDCG as reward
        from sklearn.metrics import ndcg_score
        try:
            ndcg = ndcg_score([gold_scores], [pred_scores])
        except:
            ndcg = 0.0

        return ndcg

    def collect_experience(self, data_loader, num_samples):
        """
        Collect experience by sampling from current policy
        """
        experiences = []
        self.policy.eval()
        self.value_net.eval()

        collected = 0
        with torch.no_grad():
            for batch in data_loader:
                if collected >= num_samples:
                    break

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                gold_scores = batch['scores'].cpu().numpy()

                # Get policy logits and values
                policy_logits = self.policy(input_ids, attention_mask)
                values = self.value_net(input_ids, attention_mask)

                # Sample actions from policy
                probs = self.policy.get_action_probs(policy_logits)
                dist = Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                # Store experience
                for i in range(len(input_ids)):
                    experiences.append({
                        'input_ids': input_ids[i].cpu(),
                        'attention_mask': attention_mask[i].cpu(),
                        'action': actions[i].cpu(),
                        'log_prob': log_probs[i].cpu(),
                        'value': values[i].cpu(),
                        'gold_score': gold_scores[i]
                    })
                    collected += 1

                    if collected >= num_samples:
                        break

        return experiences

    def update_policy(self, experiences):
        """
        Update policy and value networks using PPO
        """
        # Compute rewards and advantages
        rewards = torch.tensor([exp['gold_score'] for exp in experiences], dtype=torch.float32)
        values = torch.stack([exp['value'] for exp in experiences])
        old_log_probs = torch.stack([exp['log_prob'] for exp in experiences])

        # Simple advantage: reward - value (no temporal aspect for single-step)
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Returns for value function
        returns = rewards

        # Move to device
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0

        for _ in range(self.ppo_epochs):
            # Prepare batch
            input_ids = torch.stack([exp['input_ids'] for exp in experiences]).to(self.device)
            attention_mask = torch.stack([exp['attention_mask'] for exp in experiences]).to(self.device)
            actions = torch.stack([exp['action'] for exp in experiences]).to(self.device)

            # Get current log probs and values
            log_probs, entropy = self.policy.evaluate_actions(input_ids, attention_mask, actions)
            current_values = self.value_net(input_ids, attention_mask)

            # Compute losses
            policy_loss = self.compute_ppo_loss(old_log_probs, log_probs, advantages, entropy)
            value_loss = F.mse_loss(current_values, returns)

            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()
            self.policy_scheduler.step()

            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
            self.value_optimizer.step()
            self.value_scheduler.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'mean_reward': rewards.mean().item()
        }


def train_ppo_ranking_model(args):
    """Train a PPO-based ranking reward model"""

    # Initialize wandb if requested
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project="ppo-ranking-reward-model", config=vars(args))
        except ImportError:
            print("wandb not installed, skipping logging")
            args.use_wandb = False

    # Initialize TensorBoard
    if args.use_tensorboard:
        tb_log_dir = os.path.join(args.output_dir, 'tensorboard_logs_ppo')
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")
    else:
        writer = None

    # Load data (reuse data loading from train_ranking_model.py)
    print(f"Loading data for dataset: {args.dataset}")
    from train_ranking_model import RankingDataset, collate_fn_dynamic_padding

    # Import dataset loading logic
    if args.dataset == 'ds_critique':
        from datasets import load_from_disk, DatasetDict

        data_dir = 'data/processed/comprehensive_ranking_dataset'
        full_dataset = load_from_disk(data_dir)

        train_ds = full_dataset['train'].filter(lambda x: x['source'] == 'ds-critique')
        val_ds = full_dataset['validation'].filter(lambda x: x['source'] == 'ds-critique')

        def convert_format(example):
            return {
                'query_id': example['query_id'],
                'question': example['premise'],
                'explanation': example['candidate'],
                'score': example['quality_score'] + 1,
                'critique': f"Quality level: {example['quality_score']}"
            }

        train_converted = train_ds.map(convert_format, remove_columns=train_ds.column_names)
        val_converted = val_ds.map(convert_format, remove_columns=val_ds.column_names)

        dataset = DatasetDict({
            'train': train_converted,
            'validation': val_converted
        })

        # Convert to ranking format (simplified version)
        def convert_to_ranking_format(split):
            ranking_examples = []
            query_groups = {}

            for example in dataset[split]:
                query_id = example.get('query_id', '')
                question = example.get('question', '')
                explanation = example.get('explanation', '')
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
                if len(group_data['explanations']) < 2:
                    continue

                # Normalize scores
                normalized_scores = [(s - 1) / 4.0 for s in group_data['scores']]

                ranking_examples.append({
                    'query': f"Question: {group_data['question']}\\nExplain why the answer is correct.",
                    'explanations': group_data['explanations'],
                    'scores': normalized_scores,
                    'num_candidates': len(group_data['explanations'])
                })

            return ranking_examples

        def create_regression_data(ranking_examples):
            regression_examples = []
            for example in ranking_examples:
                query = example['query']
                for exp, normalized_score in zip(example['explanations'], example['scores']):
                    regression_examples.append({
                        'query': query,
                        'explanation': exp,
                        'score': normalized_score,
                        'normalized_score': normalized_score
                    })
            return regression_examples

        train_data = convert_to_ranking_format('train')
        val_data = convert_to_ranking_format('validation')

        train_regression = create_regression_data(train_data)
        val_regression = create_regression_data(val_data)
    else:
        raise ValueError(f"Dataset {args.dataset} not yet supported for PPO training")

    print(f"Train examples: {len(train_regression)}")
    print(f"Val examples: {len(val_regression)}")

    # Device setup
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Initialize PPO components
    print(f"Initializing PPO policy and value networks: {args.base_model}")
    policy = PPORankingPolicy(
        base_model=args.base_model,
        dropout=args.dropout,
        use_quantization=args.use_quantization
    ).to(device)

    value_net = PPOValueNetwork(
        base_model=args.base_model,
        dropout=args.dropout,
        use_quantization=args.use_quantization
    ).to(device)

    # Create datasets
    train_dataset = RankingDataset(train_regression, policy.tokenizer, pre_tokenize=True)
    val_dataset = RankingDataset(val_regression, value_net.tokenizer, pre_tokenize=True)

    # Data loaders
    use_pin_memory = args.use_cuda and torch.cuda.is_available()
    collate_fn = partial(collate_fn_dynamic_padding, tokenizer=policy.tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn
    )

    # Initialize PPO trainer
    trainer = PPORankingTrainer(policy, value_net, args, device)

    # Training loop
    print(f"\\nStarting PPO training for {args.num_epochs} epochs...")
    best_val_score = float('-inf')
    os.makedirs(args.output_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.num_epochs):
        # Collect experience and update
        for update in range(args.updates_per_epoch):
            # Collect experience
            experiences = trainer.collect_experience(train_loader, args.batch_size)

            # Update policy
            update_info = trainer.update_policy(experiences)

            global_step += 1

            # Log
            if writer is not None:
                writer.add_scalar('Loss/policy', update_info['policy_loss'], global_step)
                writer.add_scalar('Loss/value', update_info['value_loss'], global_step)
                writer.add_scalar('Reward/mean', update_info['mean_reward'], global_step)

            if update % 10 == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs}, Update {update+1}/{args.updates_per_epoch}: "
                      f"Policy Loss: {update_info['policy_loss']:.4f}, "
                      f"Value Loss: {update_info['value_loss']:.4f}, "
                      f"Mean Reward: {update_info['mean_reward']:.4f}")

        # Validation
        if (epoch + 1) % args.val_frequency == 0:
            print("\\nValidating...")
            # Validation would use ranking evaluator on val_data
            # Skipping for now since ds_critique has limited validation queries
            print("Skipping detailed validation for PPO (use comparison script instead)")

        # Save checkpoint
        if (epoch + 1) % args.save_frequency == 0:
            save_path = os.path.join(args.output_dir, f'ppo_checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'value_net_state_dict': value_net.state_dict(),
                'policy_optimizer_state_dict': trainer.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': trainer.value_optimizer.state_dict(),
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, 'ppo_final_model.pt')
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_net_state_dict': value_net.state_dict(),
        'config': {
            'base_model': args.base_model,
            'dropout': args.dropout
        }
    }, final_path)
    print(f"Saved final PPO model to {final_path}")

    if writer is not None:
        writer.close()

    return policy, value_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO-based ranking reward model")
    parser.add_argument('--base_model', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--updates_per_epoch', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default='models/ppo_ranking_reward')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true')
    parser.add_argument('--dataset', type=str, default='ds_critique')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_frequency', type=int, default=5)
    parser.add_argument('--save_frequency', type=int, default=5)
    parser.add_argument('--use_quantization', action='store_true')

    # PPO-specific hyperparameters
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--ppo_epochs', type=int, default=4)

    args = parser.parse_args()

    print("="*60)
    print("PPO RANKING REWARD MODEL TRAINING")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"PPO clip epsilon: {args.clip_epsilon}")
    print(f"PPO entropy coef: {args.entropy_coef}")
    print("="*60 + "\\n")

    train_ppo_ranking_model(args)
    print("\\nPPO training complete!")
