import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional
import numpy as np

class RankingRewardModel(nn.Module):
    """
    Reward model that outputs continuous scores for ranking explanations
    Supports both regression (predicting scores) and ranking (relative ordering)
    """

    def __init__(self,
                 base_model: str = "bert-base-uncased",
                 output_mode: str = "regression",  # "regression" or "ranking"
                 num_labels: int = 1,
                 dropout: float = 0.1):
        super().__init__()

        self.output_mode = output_mode
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)

        hidden_size = self.encoder.config.hidden_size

        # Projection head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )

        if output_mode == "regression":
            # For regression, we want scores in [0, 1]
            self.output_activation = nn.Sigmoid()
        else:
            # For ranking, raw logits are fine
            self.output_activation = nn.Identity()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )

        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        scores = self.output_activation(logits)

        return scores.squeeze(-1) if self.output_mode == "regression" else scores

    def rank_explanations(self,
                         query: str,
                         explanations: List[str],
                         return_scores: bool = True) -> List[float]:
        """
        Rank a list of explanations for a given query

        Args:
            query: The instruction-style query
            explanations: List of candidate explanations
            return_scores: If True, return scores; if False, return ranks

        Returns:
            List of scores or ranks
        """
        self.eval()

        # Tokenize query + explanation pairs
        inputs = []
        for exp in explanations:
            text = f"{query} {self.tokenizer.sep_token} {exp}"
            inputs.append(text)

        # Batch encode
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        # Move to same device as model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Get scores
        with torch.no_grad():
            scores = self.forward(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )

        scores = scores.cpu().numpy()

        if return_scores:
            return scores.tolist()
        else:
            # Return ranks (1 = best)
            ranks = np.argsort(-scores) + 1
            return ranks.tolist()

    def compute_ranking_loss(self,
                            scores: torch.Tensor,
                            labels: torch.Tensor,
                            margin: float = 1.0) -> torch.Tensor:
        """
        Compute pairwise ranking loss

        Args:
            scores: Model scores for batch of explanations
            labels: Ground truth scores/rankings
            margin: Margin for ranking loss
        """
        # Create all pairs where label_i > label_j
        batch_size = scores.size(0)

        # Expand for pairwise comparisons
        scores_i = scores.unsqueeze(1).expand(batch_size, batch_size)
        scores_j = scores.unsqueeze(0).expand(batch_size, batch_size)

        labels_i = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_j = labels.unsqueeze(0).expand(batch_size, batch_size)

        # Mask for valid pairs (where label_i > label_j)
        mask = (labels_i > labels_j).float()

        # Ranking loss: max(0, margin - (score_i - score_j))
        losses = torch.relu(margin - (scores_i - scores_j)) * mask

        # Average over valid pairs
        num_pairs = mask.sum()
        if num_pairs > 0:
            loss = losses.sum() / num_pairs
        else:
            loss = torch.tensor(0.0, device=scores.device)

        return loss


class RankingLosses:
    @staticmethod
    def pairwise_hinge_loss(scores_pos, scores_neg, margin=1.0):
        """Standard pairwise hinge loss for ranking"""
        return torch.relu(margin - (scores_pos - scores_neg)).mean()

    @staticmethod
    def weighted_hinge_loss(scores_pos, scores_neg, weights, margin=1.0):
        """Weighted hinge loss based on quality score differences"""
        losses = torch.relu(margin - (scores_pos - scores_neg))
        return (losses * weights).mean()

    @staticmethod
    def multi_margin_hinge_loss(scores, labels, margins=None):
        """Different margins based on label differences"""
        if margins is None:
            # Default: larger margin for bigger quality differences
            margins = {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0}

        batch_size = scores.size(0)
        losses = []

        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if labels[i] > labels[j]:
                    diff = int(labels[i] - labels[j])
                    margin = margins.get(diff, 1.0)
                    loss = torch.relu(margin - (scores[i] - scores[j]))
                    losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    @staticmethod
    def ranknet_loss(scores_i, scores_j, labels_i, labels_j):
        """RankNet loss (cross-entropy on pairwise preferences)"""
        # Compute target probabilities
        P_ij = torch.sigmoid(labels_i - labels_j)

        # Compute predicted probabilities
        s_ij = scores_i - scores_j

        # Cross entropy loss
        loss = -P_ij * torch.log(torch.sigmoid(s_ij)) - (1 - P_ij) * torch.log(1 - torch.sigmoid(s_ij))
        return loss.mean()

    @staticmethod
    def lambdarank_loss(scores, labels, ndcg_gains=None):
        """LambdaRank loss - RankNet weighted by NDCG changes"""
        if ndcg_gains is None:
            ndcg_gains = 2**labels - 1  # Default DCG gains

        batch_size = scores.size(0)
        device = scores.device

        # Compute pairwise probabilities
        scores_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # [B, B]
        labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # [B, B]

        # Only consider pairs where labels differ
        mask = labels_diff > 0

        # RankNet probabilities
        P_ij = torch.sigmoid(labels_diff)
        s_ij = torch.sigmoid(scores_diff)

        # Compute position discounts
        ranks = scores.argsort(descending=True).argsort() + 1
        discounts = 1.0 / torch.log2(ranks.float() + 1)

        # NDCG change if positions were swapped
        delta_ndcg = torch.abs(
            (ndcg_gains.unsqueeze(1) - ndcg_gains.unsqueeze(0)) *
            (discounts.unsqueeze(1) - discounts.unsqueeze(0))
        )

        # Weighted loss
        loss = -delta_ndcg * (P_ij * torch.log(s_ij + 1e-10) + (1 - P_ij) * torch.log(1 - s_ij + 1e-10))

        return loss[mask].mean() if mask.any() else torch.tensor(0.0, device=device)

    @staticmethod
    def listnet_loss(scores, labels, temp=1.0):
        """ListNet loss - KL divergence between score distributions"""
        # Convert to probabilities
        y_pred = torch.softmax(scores / temp, dim=-1)
        y_true = torch.softmax(labels / temp, dim=-1)

        # KL divergence
        return torch.sum(y_true * torch.log(y_true / (y_pred + 1e-10)), dim=-1).mean()

    @staticmethod
    def listmle_loss(scores, labels, eps=1e-10):
        """ListMLE loss - likelihood of ground truth permutation"""
        # Sort by true labels
        sorted_labels, indices = labels.sort(dim=-1, descending=True)
        sorted_scores = scores.gather(dim=-1, index=indices)

        # Compute log likelihood
        max_scores = sorted_scores.max(dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(sorted_scores - max_scores)

        cumsum_exp = exp_scores.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        log_prob = sorted_scores - max_scores - torch.log(cumsum_exp + eps)

        # Mask out equal labels
        mask = sorted_labels[..., :-1] > sorted_labels[..., 1:]
        log_prob = log_prob[..., :-1] * mask

        return -log_prob.sum(dim=-1).mean()

    @staticmethod
    def approxndcg_loss(scores, labels, alpha=5.0):
        """ApproxNDCG - smooth approximation of NDCG"""
        n = scores.size(1)
        device = scores.device

        # Smooth ranking function
        pairwise_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
        pairwise_prob = torch.sigmoid(alpha * pairwise_diff)

        # Approximate rank
        approx_rank = 1 + pairwise_prob.sum(dim=1)

        # Compute gains and discounts
        gains = 2**labels - 1
        discounts = 1 / torch.log2(approx_rank + 1)

        # Approximate DCG
        approx_dcg = (gains * discounts).sum(dim=1)

        # Ideal DCG
        ideal_gains, _ = gains.sort(dim=1, descending=True)
        ideal_discounts = 1 / torch.log2(torch.arange(2, n + 2, device=device).float())
        ideal_dcg = (ideal_gains * ideal_discounts).sum(dim=1)

        # NDCG loss
        return 1 - (approx_dcg / (ideal_dcg + 1e-10)).mean()

    @staticmethod
    def softrank_loss(scores, labels, temp=0.1):
        """SoftRank - differentiable ranking"""
        # Compute soft permutation matrix
        n = scores.size(1)
        scores_repeated_1 = scores.unsqueeze(2).repeat(1, 1, n)
        scores_repeated_2 = scores.unsqueeze(1).repeat(1, n, 1)

        P = torch.sigmoid((scores_repeated_1 - scores_repeated_2) / temp)
        P = P * (1 - torch.eye(n, device=scores.device))

        # Soft ranks
        soft_ranks = 1 + P.sum(dim=2)

        # Ranking loss
        target_ranks = (-labels).argsort(dim=1).argsort(dim=1).float() + 1
        return torch.nn.functional.mse_loss(soft_ranks, target_ranks)


class ListwiseRankingModel(RankingRewardModel):
    """
    Extension that supports listwise ranking losses
    Better for learning to rank multiple candidates jointly
    """

    def compute_listwise_loss(self,
                             scores: torch.Tensor,
                             labels: torch.Tensor,
                             temperature: float = 1.0) -> torch.Tensor:
        """
        Compute ListNet-style listwise ranking loss
        """
        # Convert labels to probabilities
        label_probs = torch.softmax(labels / temperature, dim=-1)

        # Convert scores to probabilities
        score_probs = torch.softmax(scores / temperature, dim=-1)

        # KL divergence loss
        loss = torch.sum(label_probs * torch.log(label_probs / (score_probs + 1e-10)), dim=-1)

        return loss.mean()


# Simple mock model for testing without GPU
class SimpleMockRankingModel:
    """Mock ranking model for testing evaluation pipeline"""

    def __init__(self):
        pass

    def rank_explanations(self, query: str, explanations: List[str]) -> List[float]:
        """Rank by length (longer = better)"""
        scores = [len(exp.split()) / 50.0 for exp in explanations]
        return scores


if __name__ == "__main__":
    print("Testing Ranking Reward Model...")

    # Test with simple mock model (no GPU needed)
    print("\n1. Testing SimpleMockRankingModel...")
    model = SimpleMockRankingModel()

    query = "Why does ice float on water?"
    explanations = [
        "Ice is lighter.",
        "Ice is less dense than water because of its molecular structure.",
        "When water freezes, the molecules form a crystalline structure with more space, making it less dense."
    ]

    scores = model.rank_explanations(query, explanations)
    print(f"Query: {query}")
    print("Scores:")
    for exp, score in zip(explanations, scores):
        print(f"  {score:.3f}: {exp}")

    # Test with actual model (requires GPU/CPU)
    print("\n2. Testing RankingRewardModel (this may take a moment)...")
    try:
        actual_model = RankingRewardModel(
            base_model="bert-base-uncased",
            output_mode="regression"
        )
        print("✅ Model initialized successfully")

        # Test forward pass
        dummy_input = actual_model.tokenizer(
            ["test query [SEP] test explanation"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        output = actual_model(**dummy_input)
        print(f"✅ Forward pass successful, output shape: {output.shape}")

        # Test ranking
        scores = actual_model.rank_explanations(query, explanations)
        print(f"✅ Ranking successful, scores: {[f'{s:.3f}' for s in scores]}")

    except Exception as e:
        print(f"⚠️  Could not test actual model: {e}")
        print("   This is normal if you don't have the model downloaded yet")

    print("\n✅ Ranking model test complete!")
