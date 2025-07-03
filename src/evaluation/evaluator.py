import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from datasets import load_metric
from transformers import PreTrainedTokenizer
from tqdm import tqdm

class Evaluator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the evaluator.
        
        Args:
            tokenizer: Tokenizer for processing text
            device: Device to run evaluation on
        """
        self.tokenizer = tokenizer
        self.device = device
        
        # Load metrics
        self.rouge = load_metric("rouge")
        self.bleu = load_metric("bleu")
        self.bertscore = load_metric("bertscore")
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader containing evaluation data
        
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        all_predictions = []
        all_references = []
        all_citations = []
        citation_preservation = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Generate summaries
                outputs = model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    citations=batch.get("citations")
                )
                
                # Decode predictions
                predictions = self.tokenizer.batch_decode(
                    outputs["generated_ids"],
                    skip_special_tokens=True
                )
                
                # Decode references
                references = self.tokenizer.batch_decode(
                    batch["target_ids"],
                    skip_special_tokens=True
                )
                
                all_predictions.extend(predictions)
                all_references.extend(references)
                
                # Check citation preservation if available
                if "citations" in batch:
                    batch_citations = self._extract_citations(batch["citations"])
                    all_citations.extend(batch_citations)
                    citation_preservation.extend(
                        self._check_citations(predictions, batch_citations)
                    )
        
        # Calculate metrics
        metrics = {}
        
        # ROUGE scores
        rouge_output = self.rouge.compute(
            predictions=all_predictions,
            references=all_references,
            use_stemmer=True
        )
        metrics.update({
            f"rouge_{key}": value.mid.fmeasure
            for key, value in rouge_output.items()
        })
        
        # BLEU score
        bleu_output = self.bleu.compute(
            predictions=all_predictions,
            references=[[ref] for ref in all_references]
        )
        metrics["bleu"] = bleu_output["bleu"]
        
        # BERTScore
        bertscore_output = self.bertscore.compute(
            predictions=all_predictions,
            references=all_references,
            lang="en"
        )
        metrics["bertscore_f1"] = np.mean(bertscore_output["f1"])
        
        # Citation preservation rate
        if citation_preservation:
            metrics["citation_preservation"] = np.mean(citation_preservation)
        
        return metrics
    
    def _extract_citations(self, citations_tensor: torch.Tensor) -> List[List[str]]:
        """Extract citations from tensor format."""
        # Placeholder for citation extraction
        # This would convert tensor features back to citation text
        return [[] for _ in range(citations_tensor.size(0))]  # Dummy implementation
    
    def _check_citations(
        self,
        summaries: List[str],
        citations: List[List[str]]
    ) -> List[float]:
        """
        Check citation preservation in summaries.
        
        Args:
            summaries: List of generated summaries
            citations: List of citation lists for each summary
        
        Returns:
            List of citation preservation rates
        """
        preservation_rates = []
        
        for summary, paper_citations in zip(summaries, citations):
            if not paper_citations:
                continue
            
            preserved = 0
            for citation in paper_citations:
                if citation in summary:
                    preserved += 1
            
            preservation_rates.append(preserved / len(paper_citations))
        
        return preservation_rates

class HumanEvaluator:
    def __init__(self):
        """Initialize the human evaluation interface."""
        self.criteria = {
            "content_accuracy": "How accurately does the summary represent the main points? (1-5)",
            "readability": "How clear and well-written is the summary? (1-5)",
            "coherence": "How well does the summary flow and maintain logical connections? (1-5)",
            "citation_usage": "How appropriately are citations integrated? (1-5)",
            "technical_accuracy": "How well are technical details preserved? (1-5)"
        }
    
    def evaluate_summary(
        self,
        summary: str,
        original_text: str,
        evaluator_id: str
    ) -> Dict[str, int]:
        """
        Collect human evaluation scores for a summary.
        
        Args:
            summary: Generated summary to evaluate
            original_text: Original paper text
            evaluator_id: Identifier for the human evaluator
        
        Returns:
            Dictionary of scores for each criterion
        """
        print("\nOriginal Text:")
        print("-" * 80)
        print(original_text[:1000] + "...")
        print("\nGenerated Summary:")
        print("-" * 80)
        print(summary)
        print("-" * 80)
        
        scores = {}
        for criterion, description in self.criteria.items():
            while True:
                try:
                    score = int(input(f"\n{description} "))
                    if 1 <= score <= 5:
                        scores[criterion] = score
                        break
                    print("Please enter a score between 1 and 5.")
                except ValueError:
                    print("Please enter a valid number.")
        
        # Add metadata
        scores["evaluator_id"] = evaluator_id
        scores["timestamp"] = torch.cuda.Event().record()
        
        return scores

def calculate_agreement(evaluations: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate inter-rater agreement for human evaluations.
    
    Args:
        evaluations: List of evaluation dictionaries
    
    Returns:
        Dictionary of agreement scores for each criterion
    """
    agreement_scores = {}
    
    # Group evaluations by summary
    by_summary = {}
    for eval_dict in evaluations:
        summary_id = eval_dict["summary_id"]
        if summary_id not in by_summary:
            by_summary[summary_id] = []
        by_summary[summary_id].append(eval_dict)
    
    # Calculate agreement for each criterion
    criteria = [k for k in evaluations[0].keys() if k not in ["evaluator_id", "timestamp", "summary_id"]]
    
    for criterion in criteria:
        agreements = []
        for summary_evals in by_summary.values():
            if len(summary_evals) < 2:
                continue
            
            # Calculate pairwise agreement
            scores = [e[criterion] for e in summary_evals]
            n_agree = sum(1 for i in range(len(scores))
                         for j in range(i + 1, len(scores))
                         if abs(scores[i] - scores[j]) <= 1)
            n_pairs = len(scores) * (len(scores) - 1) / 2
            agreements.append(n_agree / n_pairs)
        
        agreement_scores[criterion] = np.mean(agreements) if agreements else 0.0
    
    return agreement_scores 