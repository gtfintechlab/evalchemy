"""
BIZBENCH task implementation for Evalchemy.

This task evaluates SEC filing numerical extraction and business understanding.
Dataset: kensho/bizbench
Task type: qa
Metrics: Accuracy (with numeric tolerance), Mean Absolute Error
"""

import datasets
import re
import numpy as np
from typing import Dict, Any, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class FlameBizbenchBenchmark(BaseBenchmark):
    """FLaME BIZBENCH benchmark for Evalchemy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "kensho/bizbench"
        self.max_new_tokens = kwargs.get('max_new_tokens', 20)  # BizBench just needs numbers
        self.temperature = kwargs.get('temperature', 0.0)
        self.tolerance = kwargs.get('tolerance', 0.01)
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for BIZBENCH task."""
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
            from .mock_dataset import create_mock_bizbench_dataset
            dataset = create_mock_bizbench_dataset()
            
        instances = []
        valid_examples = []
        
        # Process each example - filter out examples with no context
        for idx, example in enumerate(dataset):
            # Extract fields based on BizBench dataset structure
            question = example.get("question", "")
            context = example.get("context", "")
            answer = example.get("answer", "")
            
            # Skip instances with no context (as per FLaME implementation)
            if not context:
                continue
                
            valid_examples.append(example)
            
            # Create prompt based on FLaME prompt style
            prompt = self._create_prompt(question, context)
            
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
                len(instances)  # Use current length as index
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
            "examples": valid_examples,
            "task_name": "flame_bizbench"
        }
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for BIZBENCH task."""
        if results is None:
            return None
            
        outputs = results["outputs"]
        examples = results["examples"]
        
        correct = 0
        total = 0
        errors = []
        
        for output, example in zip(outputs, examples):
            # Get predicted and true answers
            pred_answer = self._extract_answer(output)
            true_answer = self._to_float(str(example.get("answer", "")))
            
            if pred_answer is not None and true_answer is not None:
                # Check exact match
                if pred_answer == true_answer:
                    correct += 1
                    
                # Calculate error for MAE
                errors.append(abs(pred_answer - true_answer))
                total += 1
            
        if total == 0:
            return {"accuracy": 0.0, "mae": 0.0, "tolerance_accuracy": 0.0}
            
        accuracy = correct / total
        mae = np.mean(errors)
        
        # Calculate tolerance accuracy
        tolerance_correct = sum(1 for e in errors if e <= self.tolerance)
        tolerance_accuracy = tolerance_correct / total
        
        return {
            "exact_match_accuracy": accuracy,
            "mean_absolute_error": mae,
            "tolerance_accuracy": tolerance_accuracy
        }
        
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for BIZBENCH task based on FLaME implementation."""
        prompt = f"""Discard all previous instructions. You are an expert financial data extractor.
Extract the answer to the following question from the provided SEC filing context.
Provide the answer with just the number without any units or other text.

Question: {question}
Context: {context}"""
        return prompt
        
    def _extract_answer(self, output: str) -> Optional[float]:
        """Extract numeric answer from model output."""
        if not output:
            return None
            
        # Clean the output
        output = output.strip()
        
        # Try to extract numbers (including negative numbers and decimals)
        numbers = re.findall(r'-?\d+\.?\d*', output)
        
        if numbers:
            # Return the first number found (BizBench expects just a number)
            try:
                return float(numbers[0])
            except ValueError:
                pass
                
        return None
        
    def _to_float(self, value: str) -> Optional[float]:
        """Convert string value to float, handling commas."""
        try:
            # Remove commas and convert to float
            return float(str(value).replace(",", ""))
        except ValueError:
            return None