"""
TATQA task implementation for Evalchemy.

This task evaluates Table and text-based financial question answering.
Dataset: gtfintechlab/TATQA
Task type: qa
Metrics: Accuracy (with numeric tolerance)
"""

import datasets
import re
from typing import Dict, Any, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class FlameTatqaBenchmark(BaseBenchmark):
    """FLaME TATQA benchmark for Evalchemy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "gtfintechlab/TATQA"
        self.max_new_tokens = kwargs.get('max_new_tokens', 128)
        self.temperature = kwargs.get('temperature', 0.0)
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for TATQA task."""
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
            from .mock_dataset import create_mock_tatqa_dataset
            dataset = create_mock_tatqa_dataset()
            
        instances = []
        
        # Process each example
        for idx, example in enumerate(dataset):
            question = example.get("question", example.get("Question", ""))
            context = example.get("context", example.get("Context", ""))
            table = example.get("table", example.get("Table", ""))
            
            # Create prompt based on FLaME prompt style
            prompt = self._create_prompt(question, context, table)
            
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
            "task_name": "flame_tatqa"
        }
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for TATQA task."""
        if results is None:
            return None
            
        outputs = results["outputs"]
        examples = results["examples"]
        
        correct = 0
        total = 0
        
        for output, example in zip(outputs, examples):
            # Get predicted and true answers
            pred_answer = self._extract_answer(output)
            true_answer = str(example.get("answer", ""))
            
            # Check if answers match
            if self._answers_match(pred_answer, true_answer):
                correct += 1
            total += 1
            
        accuracy = correct / total if total > 0 else 0.0
        
        return {"accuracy": accuracy}
        
    def _create_prompt(self, question: str, context: str = "", table: str = "") -> str:
        """Create prompt for TATQA task."""
        prompt = "Discard all previous instructions. Behave like you are an expert financial analyst.\n\n"
        
        if context:
            prompt += f"Text: {context}\n\n"
        
        if table:
            prompt += f"Table: {table}\n\n"
            
        prompt += f"Question: {question}\n\n"
        prompt += "Answer the question using the provided text and/or table information. Provide only the final answer."
        
        return prompt
        
    def _extract_answer(self, output: str) -> str:
        """Extract answer from model output using multiple strategies."""
        if not output:
            return ""
            
        output = output.strip()
        
        # Strategy 1: Look for "Final answer:" or similar patterns
        final_patterns = [
            r'(?:final answer|answer|result):\s*([^\n\.]+)',
            r'(?:the answer is|it is)\s+([^\n\.]+)',
            r'(?:equals|total|sum):\s*([^\n]+)'
        ]
        
        for pattern in final_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up common prefixes/suffixes
                answer = re.sub(r'^[^\w\d\-\$]*|[^\w\d\%]*$', '', answer)
                if answer:
                    return answer
        
        # Strategy 2: Extract numeric values (currency, percentages, numbers)
        numeric_patterns = [
            r'\$\s*[\d,]+\.?\d*[MmBbKk]?',  # Currency: $123.45M, $1,000
            r'[\d,]+\.?\d*\s*%',             # Percentages: 45.2%
            r'[\d,]+\.?\d*[MmBbKk]?',        # Numbers: 123.45M, 1,000
        ]
        
        for pattern in numeric_patterns:
            numbers = re.findall(pattern, output)
            if numbers:
                return numbers[-1].strip()  # Return last/most specific number
        
        # Strategy 3: Look for last sentence with numbers
        sentences = output.split('.')
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if re.search(r'[\d,]+', sentence):
                # Extract first significant number from this sentence
                number_match = re.search(r'[\d,]+\.?\d*', sentence)
                if number_match:
                    return number_match.group(0)
        
        # Strategy 4: Return first line if it looks like an answer
        first_line = output.split('\n')[0].strip()
        if len(first_line) < 50 and (re.search(r'[\d]', first_line) or len(first_line.split()) <= 5):
            return first_line
            
        # Fallback: return full output truncated
        return output[:100] if len(output) > 100 else output
        
    def _answers_match(self, pred: str, true: str) -> bool:
        """Check if predicted answer matches true answer."""
        # Normalize answers
        pred = pred.strip().lower()
        true = true.strip().lower()
        
        # Direct match
        if pred == true:
            return True
            
        # Try numeric comparison with tolerance
        try:
            pred_num = float(pred.replace('%', '').replace(',', ''))
            true_num = float(true.replace('%', '').replace(',', ''))
            
            # Check with tolerance
            tolerance = 0.01
            return abs(pred_num - true_num) < tolerance
        except:
            pass
            
        return False
