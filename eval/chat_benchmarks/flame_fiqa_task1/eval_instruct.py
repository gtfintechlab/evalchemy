"""
FIQA_TASK1 task implementation for Evalchemy.

This task evaluates Financial sentiment analysis with target aspect identification.
Dataset: gtfintechlab/FiQA_Task1
Task type: regression (sentiment scoring)
Metrics: MSE, MAE, Correlation
"""

import datasets
import numpy as np
from typing import Dict, Any, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class FlameFiqaTask1Benchmark(BaseBenchmark):
    """FLaME FIQA_TASK1 benchmark for Evalchemy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "gtfintechlab/FiQA_Task1"
        self.max_new_tokens = kwargs.get('max_new_tokens', 50)
        self.temperature = kwargs.get('temperature', 0.0)
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for FIQA_TASK1 task."""
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
            from .mock_dataset import create_mock_fiqa_task1_dataset
            dataset = create_mock_fiqa_task1_dataset()
            
        instances = []
        
        # Process each example
        for idx, example in enumerate(dataset):
            sentence = example.get("sentence", "")
            target = example.get("target", "")
            
            # Create prompt for sentiment analysis with target
            prompt = self._create_prompt(sentence, target)
            
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
            "task_name": "flame_fiqa_task1"
        }
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for FIQA_TASK1 task."""
        if results is None:
            return None
            
        outputs = results["outputs"]
        examples = results["examples"]
        
        predicted_scores = []
        true_scores = []
        
        for output, example in zip(outputs, examples):
            # Extract predicted sentiment score
            predicted_score = self._extract_sentiment_score(output)
            true_score = example.get('sentiment_score', 0.0)
            
            if predicted_score is not None:
                predicted_scores.append(predicted_score)
                true_scores.append(true_score)
        
        if not predicted_scores:
            return {"mse": float('inf'), "mae": float('inf'), "correlation": 0.0}
        
        # Calculate regression metrics
        mse = np.mean([(p - t) ** 2 for p, t in zip(predicted_scores, true_scores)])
        mae = np.mean([abs(p - t) for p, t in zip(predicted_scores, true_scores)])
        
        # Calculate correlation
        correlation = np.corrcoef(predicted_scores, true_scores)[0, 1] if len(predicted_scores) > 1 else 0.0
        
        return {
            "mse": mse,
            "mae": mae, 
            "correlation": correlation,
            "valid_predictions": len(predicted_scores),
            "total_examples": len(outputs)
        }
        
    def _create_prompt(self, sentence: str, target: str) -> str:
        """Create prompt for FIQA_TASK1."""
        return f"""Discard all previous instructions. Behave like you are an expert financial sentiment analyst.

Analyze the sentiment of the following financial sentence towards the specific target entity.

Sentence: {sentence}
Target Entity: {target}

Provide a sentiment score between -1 (very negative) and +1 (very positive), where 0 is neutral.
Respond with only the numerical score (e.g., 0.3, -0.7, 0.0)."""
    
    def _extract_sentiment_score(self, output: str) -> Optional[float]:
        """Extract sentiment score from model output."""
        import re
        
        if not output:
            return None
        
        # Look for decimal numbers between -1 and 1
        numbers = re.findall(r'-?\d*\.?\d+', output.strip())
        
        for num_str in numbers:
            try:
                score = float(num_str)
                if -1.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
        
        return None