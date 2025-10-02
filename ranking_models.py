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
