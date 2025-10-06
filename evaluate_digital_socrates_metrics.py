"""
Evaluate models using Digital Socrates metrics:
- BLEU
- METEOR
- ROUGE
- CIDEr
- BERTScore
- Accuracy (ranking accuracy)

This allows direct comparison with Digital Socrates results.
"""

import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
from datasets import load_from_disk

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️  NLTK not available. Install with: pip install nltk")
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("⚠️  rouge-score not available. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("⚠️  bert-score not available. Install with: pip install bert-score")
    BERTSCORE_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM
from ranking_models import RankingRewardModel


class DigitalSocratesEvaluator:
    """
    Evaluate explanation generation using Digital Socrates metrics
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available()
                                   else 'mps' if torch.backends.mps.is_available()
                                   else 'cpu')

        print(f"Using device: {self.device}")

        # Load model
        print(f"\nLoading model: {args.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if args.use_fp16 else torch.float32,
            device_map="auto" if args.use_cuda else None
        ).to(self.device)
        self.model.eval()

        # Optionally load ranking model for accuracy
        if args.ranking_model_path:
            print(f"Loading ranking model: {args.ranking_model_path}")
            self.ranking_model = self.load_ranking_model(args.ranking_model_path)
        else:
            self.ranking_model = None

        # Initialize scorers
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        print("✅ Evaluator initialized")

    def load_ranking_model(self, path: str) -> RankingRewardModel:
        """Load ranking model"""
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

        return model

    def generate_explanation(self, query: str) -> str:
        """Generate explanation for a query"""
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_length,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated.strip()

    def compute_bleu(self, reference: str, hypothesis: str) -> float:
        """Compute BLEU score"""
        if not NLTK_AVAILABLE:
            return 0.0

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
        return score

    def compute_meteor(self, reference: str, hypothesis: str) -> float:
        """Compute METEOR score"""
        if not NLTK_AVAILABLE:
            return 0.0

        score = meteor_score([reference.lower().split()], hypothesis.lower().split())
        return score

    def compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute ROUGE scores"""
        if not ROUGE_AVAILABLE:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def compute_bertscore(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """Compute BERTScore (batched for efficiency)"""
        if not BERTSCORE_AVAILABLE:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        P, R, F1 = bert_score(hypotheses, references, lang='en', verbose=False)

        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }

    def compute_ranking_accuracy(self, query: str, generated: str, reference: str) -> float:
        """Compute ranking accuracy using ranking model"""
        if self.ranking_model is None:
            return 0.0

        # Score both generated and reference
        scores = self.ranking_model.rank_explanations(
            query,
            [generated, reference],
            return_scores=True
        )

        # Accuracy: 1 if reference scores higher, else 0
        return 1.0 if scores[1] > scores[0] else 0.0

    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate on test data

        Args:
            test_data: List of dicts with 'query', 'reference' keys

        Returns:
            Dictionary of metrics
        """
        print(f"\nEvaluating on {len(test_data)} examples...")

        all_bleu = []
        all_meteor = []
        all_rouge1 = []
        all_rouge2 = []
        all_rougeL = []
        all_accuracy = []

        references = []
        hypotheses = []

        for example in tqdm(test_data, desc="Generating and scoring"):
            query = example['query']
            reference = example['reference']

            # Generate explanation
            hypothesis = self.generate_explanation(query)

            # Compute metrics
            bleu = self.compute_bleu(reference, hypothesis)
            meteor = self.compute_meteor(reference, hypothesis)
            rouge_scores = self.compute_rouge(reference, hypothesis)

            all_bleu.append(bleu)
            all_meteor.append(meteor)
            all_rouge1.append(rouge_scores['rouge1'])
            all_rouge2.append(rouge_scores['rouge2'])
            all_rougeL.append(rouge_scores['rougeL'])

            # Ranking accuracy
            if self.ranking_model:
                acc = self.compute_ranking_accuracy(query, hypothesis, reference)
                all_accuracy.append(acc)

            # Store for BERTScore (batched)
            references.append(reference)
            hypotheses.append(hypothesis)

        # Compute BERTScore (batched)
        print("\nComputing BERTScore...")
        bertscore_results = self.compute_bertscore(references, hypotheses)

        # Aggregate results
        results = {
            'bleu': np.mean(all_bleu),
            'meteor': np.mean(all_meteor),
            'rouge1': np.mean(all_rouge1),
            'rouge2': np.mean(all_rouge2),
            'rougeL': np.mean(all_rougeL),
            'rouge': np.mean(all_rougeL),  # Digital Socrates uses ROUGE-L
            'bertscore_precision': bertscore_results['precision'],
            'bertscore_recall': bertscore_results['recall'],
            'bertscore_f1': bertscore_results['f1'],
            'bertscore': bertscore_results['f1'],  # Digital Socrates uses F1
        }

        if all_accuracy:
            results['accuracy'] = np.mean(all_accuracy)

        return results

    def print_results(self, results: Dict[str, float]):
        """Print results in Digital Socrates format"""
        print("\n" + "="*80)
        print("DIGITAL SOCRATES METRICS")
        print("="*80)
        print(f"{'Metric':<20} {'Score':<15} {'Digital Socrates Baseline'}")
        print("-"*80)
        print(f"{'Accuracy':<20} {results.get('accuracy', 0.0):.4f}       0.5523 (Llama2-7b)")
        print(f"{'BLEU':<20} {results['bleu']:.4f}       0.0051 (Baseline)")
        print(f"{'METEOR':<20} {results['meteor']:.4f}       0.1965 (Baseline)")
        print(f"{'ROUGE':<20} {results['rouge']:.4f}       0.1285 (Baseline)")
        print(f"{'BERTScore':<20} {results['bertscore']:.4f}       0.8588 (Baseline)")
        print("-"*80)
        print("\nDigital Socrates PPO TRL Results for comparison:")
        print("  PPO TRL (10k):  BLEU=0.0060, METEOR=0.2020, ROUGE=0.1384, BERTScore=0.8618")
        print("  PPO TRL (20k):  BLEU=0.0072, METEOR=0.1998, ROUGE=0.1528, BERTScore=0.8677")
        print("="*80)


def load_test_data(dataset_name: str = 'ds_critique', max_samples: int = 100) -> List[Dict]:
    """Load test data"""
    if dataset_name == 'ds_critique':
        data_dir = '/Users/tanmoy/research/PPO_learning_to_rank/PPO_Learning_to_rank/data/processed/comprehensive_ranking_dataset'
        if not os.path.exists(data_dir):
            data_dir = 'data/processed/comprehensive_ranking_dataset'

        dataset = load_from_disk(data_dir)
        test_ds = dataset['test'].filter(lambda x: x['source'] == 'ds-critique')

        test_data = []
        for i, example in enumerate(test_ds):
            if i >= max_samples:
                break

            test_data.append({
                'query': f"Question: {example['premise']}\nExplain why the answer is correct.",
                'reference': example['candidate']
            })

        return test_data
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def main():
    parser = argparse.ArgumentParser(description="Evaluate with Digital Socrates metrics")

    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model to evaluate')
    parser.add_argument('--ranking_model_path', type=str, default=None,
                       help='Path to ranking model for accuracy metric')

    # Evaluation
    parser.add_argument('--dataset', type=str, default='ds_critique')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Max test samples to evaluate')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Max generation length')

    # Output
    parser.add_argument('--output_file', type=str, default='digital_socrates_metrics.json')

    # Hardware
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_fp16', action='store_true')

    args = parser.parse_args()

    # Check dependencies
    missing = []
    if not NLTK_AVAILABLE:
        missing.append("nltk (pip install nltk)")
    if not ROUGE_AVAILABLE:
        missing.append("rouge-score (pip install rouge-score)")
    if not BERTSCORE_AVAILABLE:
        missing.append("bert-score (pip install bert-score)")

    if missing:
        print("\n⚠️  Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall all at once:")
        print("  pip install nltk rouge-score bert-score")
        return

    print("="*80)
    print("DIGITAL SOCRATES EVALUATION")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print("="*80)

    # Load test data
    test_data = load_test_data(args.dataset, args.max_samples)
    print(f"\nLoaded {len(test_data)} test examples")

    # Initialize evaluator
    evaluator = DigitalSocratesEvaluator(args)

    # Evaluate
    results = evaluator.evaluate(test_data)

    # Print results
    evaluator.print_results(results)

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
