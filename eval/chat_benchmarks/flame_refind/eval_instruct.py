"""
REFIND task implementation for Evalchemy.

This task evaluates Relation extraction between entities in financial documents.
Dataset: gtfintechlab/ReFinD
Task type: classification (relation classification)
Metrics: Accuracy, Precision, Recall, F1 Score
"""

import datasets
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class FlameRefindBenchmark(BaseBenchmark):
    """FLaME REFIND benchmark for Evalchemy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "gtfintechlab/ReFinD"
        self.max_new_tokens = kwargs.get('max_new_tokens', 50)
        self.temperature = kwargs.get('temperature', 0.0)
        
        # Common financial relation types (simplified for classification)
        self.relation_types = [
            'no_relation',
            'org-aff',      # organizational affiliation  
            'per-org',      # person-organization
            'org-loc',      # organization-location
            'per-loc',      # person-location
            'other'         # other relations
        ]
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for REFIND task."""
        try:
            # Load dataset
            dataset = datasets.load_dataset(
                self.dataset_name,
                split="test",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load dataset {self.dataset_name}: {e}")
            print("Falling back to mock dataset...")
            from .mock_dataset import create_mock_refind_dataset
            dataset = create_mock_refind_dataset()
            
        instances = []
        
        # Process each example
        for idx, example in enumerate(dataset):
            # Extract entities and context
            tokens = example.get("token", [])
            e1 = example.get("e1", "")
            e2 = example.get("e2", "")
            e1_start = example.get("e1_start", 0)
            e1_end = example.get("e1_end", 0)
            e2_start = example.get("e2_start", 0)
            e2_end = example.get("e2_end", 0)
            
            # Reconstruct sentence from tokens
            if tokens:
                sentence = " ".join(tokens)
            else:
                sentence = f"Entity1: {e1}, Entity2: {e2}"
            
            # Create prompt for relation classification
            prompt = self._create_prompt(sentence, e1, e2)
            
            # Format messages
            messages = [{"role": "user", "content": prompt}]
            formatted = self._prepare_messages(messages, model)
            
            # Create instance
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }
            
            instance = Instance(
                "generate_until",
                example,
                (formatted, gen_kwargs),
                idx
            )
            instances.append(instance)
            
        # Generate outputs
        outputs = self.compute(model, instances)
        
        # Only process on rank 0
        if model.rank != 0:
            return None
            
        # Prepare results
        results = {
            "outputs": outputs,
            "examples": list(dataset),
            "task_name": "flame_refind"
        }
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for REFIND task."""
        if results is None:
            return None
            
        outputs = results["outputs"]
        examples = results["examples"]
        
        # Extract predictions and ground truth
        predictions = []
        ground_truths = []
        
        for output, example in zip(outputs, examples):
            # Parse model output
            pred_relation = self._parse_relation(output)
            predictions.append(pred_relation)
            
            # Get ground truth
            true_relation = example.get("relation", "no_relation")
            ground_truths.append(true_relation)
            
        # Map to binary classification (relation vs no_relation)
        pred_binary = [1 if p != "no_relation" else 0 for p in predictions]
        true_binary = [1 if t != "no_relation" else 0 for t in ground_truths]
        
        # Calculate metrics
        accuracy = accuracy_score(true_binary, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_binary, pred_binary, average='binary', zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "total_examples": len(outputs)
        }
        
    def _create_prompt(self, sentence: str, e1: str, e2: str) -> str:
        """Create prompt for relation extraction."""
        return f"""Discard all previous instructions. Behave like you are an expert in financial relation extraction.

Analyze the relationship between two entities in the following financial text.

Text: {sentence}

Entity 1: {e1}
Entity 2: {e2}

Determine if there is a meaningful relationship between these entities. Consider organizational affiliations, ownership, partnerships, or other business relationships.

Answer with one of:
- "relation" if there is a meaningful business relationship
- "no_relation" if there is no clear relationship

Answer:"""
        
    def _parse_relation(self, output: str) -> str:
        """Parse model output to extract relation type."""
        if not output:
            return "no_relation"
        
        output_lower = output.strip().lower()
        
        # Check for explicit relation indicators
        if "no_relation" in output_lower or "no relation" in output_lower:
            return "no_relation"
        elif "relation" in output_lower:
            return "relation"
        elif any(word in output_lower for word in ["partnership", "owns", "affiliate", "subsidiary", "parent", "member"]):
            return "relation"
        else:
            return "no_relation"