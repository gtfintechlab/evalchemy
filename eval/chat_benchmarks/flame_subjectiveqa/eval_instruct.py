"""
SUBJECTIVEQA task implementation for Evalchemy.

This task evaluates Multi-aspect subjective analysis of financial responses.
Dataset: gtfintechlab/subjectiveqa
Task type: multi-aspect rating (0-2 scale)
Metrics: Mean Absolute Error, Accuracy per aspect
"""

import datasets
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from typing import Dict, Any, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class FlameSubjectiveqaBenchmark(BaseBenchmark):
    """FLaME SUBJECTIVEQA benchmark for Evalchemy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "gtfintechlab/subjectiveqa"
        self.dataset_config = "5768"
        self.max_new_tokens = kwargs.get('max_new_tokens', 100)
        self.temperature = kwargs.get('temperature', 0.0)
        
        # Aspects to evaluate (0-2 scale)
        self.aspects = ['CLEAR', 'ASSERTIVE', 'CAUTIOUS', 'OPTIMISTIC', 'SPECIFIC', 'RELEVANT']
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for SUBJECTIVEQA task."""
        try:
            # Load dataset
            dataset = datasets.load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                split="test",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load dataset {self.dataset_name}: {e}")
            print("Falling back to mock dataset...")
            from .mock_dataset import create_mock_subjectiveqa_dataset
            dataset = create_mock_subjectiveqa_dataset()
            
        instances = []
        
        # Process each example
        for idx, example in enumerate(dataset):
            question = example.get("QUESTION", "")
            answer = example.get("ANSWER", "")
            
            # Create prompt for subjective quality assessment
            prompt = self._create_prompt(question, answer)
            
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
            "task_name": "flame_subjectiveqa"
        }
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for SUBJECTIVEQA task."""
        if results is None:
            return None
            
        outputs = results["outputs"]
        examples = results["examples"]
        
        # Collect predictions and ground truth for each aspect
        aspect_predictions = {aspect: [] for aspect in self.aspects}
        aspect_ground_truth = {aspect: [] for aspect in self.aspects}
        
        for output, example in zip(outputs, examples):
            # Parse model output to extract ratings for each aspect
            predicted_ratings = self._parse_ratings(output)
            
            for aspect in self.aspects:
                pred_rating = predicted_ratings.get(aspect, 1)  # Default to 1 if not found
                true_rating = example.get(aspect, 1)
                
                aspect_predictions[aspect].append(pred_rating)
                aspect_ground_truth[aspect].append(true_rating)
        
        # Calculate metrics for each aspect
        aspect_metrics = {}
        overall_mae = []
        overall_accuracy = []
        
        for aspect in self.aspects:
            if aspect_predictions[aspect] and aspect_ground_truth[aspect]:
                # Mean Absolute Error
                mae = mean_absolute_error(aspect_ground_truth[aspect], aspect_predictions[aspect])
                
                # Exact match accuracy
                accuracy = accuracy_score(aspect_ground_truth[aspect], aspect_predictions[aspect])
                
                aspect_metrics[f"{aspect.lower()}_mae"] = mae
                aspect_metrics[f"{aspect.lower()}_accuracy"] = accuracy
                
                overall_mae.append(mae)
                overall_accuracy.append(accuracy)
        
        # Overall metrics
        aspect_metrics["overall_mae"] = np.mean(overall_mae) if overall_mae else 0.0
        aspect_metrics["overall_accuracy"] = np.mean(overall_accuracy) if overall_accuracy else 0.0
        aspect_metrics["total_examples"] = len(outputs)
        
        return aspect_metrics
        
    def _create_prompt(self, question: str, answer: str) -> str:
        """Create prompt for subjective quality assessment."""
        return f"""Discard all previous instructions. Behave like you are an expert in evaluating financial communication quality.

Evaluate the following question-answer pair from a financial earnings call on multiple aspects. Rate each aspect on a 0-2 scale:
- 0: Low/Poor
- 1: Medium/Moderate  
- 2: High/Excellent

Question: {question}

Answer: {answer}

Rate the answer on these aspects:
1. CLEAR: How clear and understandable is the answer? (0-2)
2. ASSERTIVE: How confident and assertive is the tone? (0-2)
3. CAUTIOUS: How cautious or conservative is the response? (0-2)
4. OPTIMISTIC: How optimistic or positive is the outlook? (0-2)
5. SPECIFIC: How specific and detailed is the information? (0-2)
6. RELEVANT: How relevant is the answer to the question? (0-2)

Provide your ratings in this format:
CLEAR: [rating]
ASSERTIVE: [rating]
CAUTIOUS: [rating]
OPTIMISTIC: [rating]
SPECIFIC: [rating]
RELEVANT: [rating]"""
        
    def _parse_ratings(self, output: str) -> Dict[str, int]:
        """Parse model output to extract ratings for each aspect."""
        import re
        
        ratings = {}
        
        for aspect in self.aspects:
            # Look for pattern like "CLEAR: 1" or "CLEAR:1"
            pattern = rf"{aspect}:\s*([012])"
            match = re.search(pattern, output, re.IGNORECASE)
            
            if match:
                try:
                    rating = int(match.group(1))
                    if 0 <= rating <= 2:
                        ratings[aspect] = rating
                    else:
                        ratings[aspect] = 1  # Default
                except ValueError:
                    ratings[aspect] = 1  # Default
            else:
                ratings[aspect] = 1  # Default if not found
        
        return ratings