#!/usr/bin/env python3
"""
Template generator for FLaME tasks in Evalchemy.

This script generates boilerplate code for integrating FLaME evaluation tasks
into the Evalchemy framework.

Usage:
    python scripts/generate_flame_task.py --task-name bizbench --task-type qa --dataset "kensho/bizbench"
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any


TASK_TEMPLATES = {
    "classification": {
        "eval_instruct": '''"""
{task_name_upper} task implementation for Evalchemy.

This task evaluates {task_description}.
Dataset: {dataset_name}
Task type: {task_type}
Metrics: Accuracy, Precision, Recall, F1 Score
"""

import datasets
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class Flame{task_name_camel}Benchmark(BaseBenchmark):
    """FLaME {task_name_upper} benchmark for Evalchemy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "{dataset_name}"
        self.max_new_tokens = kwargs.get('max_new_tokens', 50)
        self.temperature = kwargs.get('temperature', 0.0)
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for {task_name_upper} task."""
        try:
            # Load dataset
            dataset = datasets.load_dataset(
                self.dataset_name,
                split="test",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load dataset {{self.dataset_name}}: {{e}}")
            print("Falling back to mock dataset...")
            from .mock_dataset import create_mock_{task_name_lower}_dataset
            dataset = create_mock_{task_name_lower}_dataset()
            
        instances = []
        
        # Process each example
        for idx, example in enumerate(dataset):
            # TODO: Extract fields from example based on dataset structure
            # This is a template - adjust based on actual dataset fields
            text = example.get("text", example.get("sentence", ""))
            
            # Create prompt based on FLaME prompt style
            prompt = self._create_prompt(text)
            
            # Format messages
            messages = [{{"role": "user", "content": prompt}}]
            formatted = self._prepare_messages(messages, model)
            
            # Create instance
            gen_kwargs = {{
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }}
            
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
        results = {{
            "outputs": outputs,
            "examples": list(dataset),
            "task_name": "flame_{task_name_lower}"
        }}
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for {task_name_upper} task."""
        if results is None:
            return None
            
        outputs = results["outputs"]
        examples = results["examples"]
        
        # Extract predictions and ground truth
        predictions = []
        ground_truths = []
        
        for output, example in zip(outputs, examples):
            # Parse model output
            pred_label = self._parse_output(output)
            predictions.append(pred_label)
            
            # Get ground truth
            # TODO: Adjust field name based on dataset
            true_label = example.get("label", -1)
            ground_truths.append(true_label)
            
        # Convert to numpy arrays
        y_true = np.array(ground_truths)
        y_pred = np.array(predictions)
        
        # Filter out invalid predictions
        valid_mask = y_pred != -1
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            return {{"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}}
            
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {{
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }}
        
    def _create_prompt(self, text: str) -> str:
        """Create prompt for {task_name_upper} task."""
        # TODO: Implement actual prompt from FLaME
        prompt = f"""Discard all previous instructions. Behave like you are an expert.
        
Task: {{text}}

Provide your answer."""
        return prompt
        
    def _parse_output(self, output: str) -> int:
        """Parse model output to extract predicted label."""
        # TODO: Implement parsing logic based on task
        # This is a placeholder
        try:
            # Example: extract first word/line
            first_line = output.strip().split('\\n')[0]
            # Map to label
            return 0  # Placeholder
        except:
            return -1  # Invalid prediction
''',
        "mock_dataset": '''"""Mock dataset for {task_name_upper} task."""

def create_mock_{task_name_lower}_dataset():
    """Create a mock dataset for testing {task_name_upper} task."""
    mock_data = []
    
    # Add 5 mock examples
    for i in range(5):
        example = {{
            "id": i,
            "text": f"This is mock text {{i}} for {task_name_upper} task.",
            "label": i % 3,  # Mock labels 0, 1, 2
        }}
        mock_data.append(example)
    
    return mock_data
'''
    },
    
    "qa": {
        "eval_instruct": '''"""
{task_name_upper} task implementation for Evalchemy.

This task evaluates {task_description}.
Dataset: {dataset_name}
Task type: {task_type}
Metrics: Accuracy (with numeric tolerance)
"""

import datasets
import re
from typing import Dict, Any, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class Flame{task_name_camel}Benchmark(BaseBenchmark):
    """FLaME {task_name_upper} benchmark for Evalchemy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "{dataset_name}"
        self.max_new_tokens = kwargs.get('max_new_tokens', 128)
        self.temperature = kwargs.get('temperature', 0.0)
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for {task_name_upper} task."""
        try:
            # Load dataset
            dataset = datasets.load_dataset(
                self.dataset_name,
                split="test",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load dataset {{self.dataset_name}}: {{e}}")
            print("Falling back to mock dataset...")
            from .mock_dataset import create_mock_{task_name_lower}_dataset
            dataset = create_mock_{task_name_lower}_dataset()
            
        instances = []
        
        # Process each example
        for idx, example in enumerate(dataset):
            # TODO: Extract fields from example based on dataset structure
            question = example.get("question", "")
            context = example.get("context", "")
            
            # Create prompt based on FLaME prompt style
            prompt = self._create_prompt(question, context)
            
            # Format messages
            messages = [{{"role": "user", "content": prompt}}]
            formatted = self._prepare_messages(messages, model)
            
            # Create instance
            gen_kwargs = {{
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }}
            
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
        results = {{
            "outputs": outputs,
            "examples": list(dataset),
            "task_name": "flame_{task_name_lower}"
        }}
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for {task_name_upper} task."""
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
        
        return {{"accuracy": accuracy}}
        
    def _create_prompt(self, question: str, context: str = "") -> str:
        """Create prompt for {task_name_upper} task."""
        # TODO: Implement actual prompt from FLaME
        if context:
            prompt = f"""Discard all previous instructions. Behave like you are an expert.
            
Context: {{context}}
Question: {{question}}

Provide your answer. Repeat your final answer at the end."""
        else:
            prompt = f"""Discard all previous instructions. Behave like you are an expert.
            
Question: {{question}}

Provide your answer."""
        return prompt
        
    def _extract_answer(self, output: str) -> str:
        """Extract answer from model output."""
        # TODO: Implement extraction logic
        # Look for numbers or final answer
        output = output.strip()
        
        # Try to extract numbers
        numbers = re.findall(r'-?\\d+\\.?\\d*', output)
        if numbers:
            return numbers[-1]  # Return last number
            
        # Return full output as fallback
        return output
        
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
''',
        "mock_dataset": '''"""Mock dataset for {task_name_upper} task."""

def create_mock_{task_name_lower}_dataset():
    """Create a mock dataset for testing {task_name_upper} task."""
    mock_data = []
    
    # Add 5 mock QA examples
    questions = [
        "What is the revenue?",
        "What is the profit margin?",
        "How many employees?",
        "What is the growth rate?",
        "What is the market share?"
    ]
    
    answers = ["100", "15%", "5000", "10%", "25%"]
    
    for i, (q, a) in enumerate(zip(questions, answers)):
        example = {{
            "id": i,
            "question": q,
            "context": f"Company X reported strong results in Q{{i+1}}.",
            "answer": a,
        }}
        mock_data.append(example)
    
    return mock_data
'''
    },
    
    "ner": {
        "eval_instruct": '''"""
{task_name_upper} task implementation for Evalchemy.

This task evaluates {task_description}.
Dataset: {dataset_name}
Task type: {task_type}
Metrics: Token-level accuracy, Precision, Recall, F1
"""

import datasets
import numpy as np
from typing import Dict, Any, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class Flame{task_name_camel}Benchmark(BaseBenchmark):
    """FLaME {task_name_upper} benchmark for Evalchemy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "{dataset_name}"
        self.max_new_tokens = kwargs.get('max_new_tokens', 256)
        self.temperature = kwargs.get('temperature', 0.0)
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for {task_name_upper} task."""
        try:
            # Load dataset
            dataset = datasets.load_dataset(
                self.dataset_name,
                split="test",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load dataset {{self.dataset_name}}: {{e}}")
            print("Falling back to mock dataset...")
            from .mock_dataset import create_mock_{task_name_lower}_dataset
            dataset = create_mock_{task_name_lower}_dataset()
            
        instances = []
        
        # Process each example
        for idx, example in enumerate(dataset):
            # TODO: Extract fields from example based on dataset structure
            tokens = example.get("tokens", [])
            
            # Create prompt based on FLaME prompt style
            prompt = self._create_prompt(tokens)
            
            # Format messages
            messages = [{{"role": "user", "content": prompt}}]
            formatted = self._prepare_messages(messages, model)
            
            # Create instance
            gen_kwargs = {{
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            }}
            
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
        results = {{
            "outputs": outputs,
            "examples": list(dataset),
            "task_name": "flame_{task_name_lower}"
        }}
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for {task_name_upper} task."""
        if results is None:
            return None
            
        outputs = results["outputs"]
        examples = results["examples"]
        
        all_predictions = []
        all_ground_truths = []
        
        for output, example in zip(outputs, examples):
            # Parse model output to get predicted tags
            pred_tags = self._parse_tags(output, len(example.get("tokens", [])))
            true_tags = example.get("tags", [])
            
            # Ensure same length
            if len(pred_tags) == len(true_tags):
                all_predictions.extend(pred_tags)
                all_ground_truths.extend(true_tags)
                
        if not all_predictions:
            return {{"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}}
            
        # Calculate token-level metrics
        accuracy = sum(p == t for p, t in zip(all_predictions, all_ground_truths)) / len(all_predictions)
        
        # Calculate precision, recall, F1 for non-O tags
        # TODO: Implement proper NER evaluation metrics
        
        return {{
            "accuracy": accuracy,
            "precision": 0.0,  # TODO: Implement
            "recall": 0.0,     # TODO: Implement
            "f1": 0.0          # TODO: Implement
        }}
        
    def _create_prompt(self, tokens: List[str]) -> str:
        """Create prompt for {task_name_upper} task."""
        # TODO: Implement actual prompt from FLaME
        prompt = f"""Discard all previous instructions. Behave like you are an expert named entity recognizer.
        
Label each token in the following sentence with appropriate tags.
Tokens: {{", ".join(tokens)}}

Return only the list of tags in the same order as the tokens."""
        return prompt
        
    def _parse_tags(self, output: str, expected_length: int) -> List[str]:
        """Parse model output to extract predicted tags."""
        # TODO: Implement parsing logic
        try:
            # Simple approach: split by whitespace
            tags = output.strip().split()
            
            # Pad or truncate to expected length
            if len(tags) < expected_length:
                tags.extend(['O'] * (expected_length - len(tags)))
            elif len(tags) > expected_length:
                tags = tags[:expected_length]
                
            return tags
        except:
            return ['O'] * expected_length
''',
        "mock_dataset": '''"""Mock dataset for {task_name_upper} task."""

def create_mock_{task_name_lower}_dataset():
    """Create a mock dataset for testing {task_name_upper} task."""
    mock_data = []
    
    # Add 5 mock NER examples
    examples = [
        {{
            "tokens": ["Apple", "Inc.", "reported", "revenue", "of", "$100B"],
            "tags": ["B-ORG", "I-ORG", "O", "O", "O", "B-MONEY"]
        }},
        {{
            "tokens": ["Microsoft", "acquired", "GitHub", "in", "2018"],
            "tags": ["B-ORG", "O", "B-ORG", "O", "B-DATE"]
        }},
        {{
            "tokens": ["The", "CEO", "announced", "layoffs"],
            "tags": ["O", "B-PER", "O", "O"]
        }},
        {{
            "tokens": ["Tesla", "stock", "rose", "5%"],
            "tags": ["B-ORG", "O", "O", "B-PERCENT"]
        }},
        {{
            "tokens": ["Amazon", "Web", "Services", "launched"],
            "tags": ["B-ORG", "I-ORG", "I-ORG", "O"]
        }}
    ]
    
    for i, ex in enumerate(examples):
        ex["id"] = i
        mock_data.append(ex)
    
    return mock_data
'''
    }
}


def generate_task(args: argparse.Namespace):
    """Generate task implementation files."""
    task_name_lower = args.task_name.lower()
    task_name_upper = args.task_name.upper()
    task_name_camel = ''.join(word.capitalize() for word in args.task_name.split('_'))
    
    # Get template based on task type
    if args.task_type not in TASK_TEMPLATES:
        print(f"Error: Unknown task type '{args.task_type}'. Choose from: {list(TASK_TEMPLATES.keys())}")
        return
        
    templates = TASK_TEMPLATES[args.task_type]
    
    # Create task directory
    task_dir = Path(f"eval/chat_benchmarks/flame_{task_name_lower}")
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Template variables
    template_vars = {
        "task_name_lower": task_name_lower,
        "task_name_upper": task_name_upper,
        "task_name_camel": task_name_camel,
        "task_description": args.description or f"{task_name_upper} evaluation",
        "task_type": args.task_type,
        "dataset_name": args.dataset,
    }
    
    # Generate eval_instruct.py
    eval_content = templates["eval_instruct"].format(**template_vars)
    eval_file = task_dir / "eval_instruct.py"
    
    if eval_file.exists() and not args.force:
        print(f"Warning: {eval_file} already exists. Use --force to overwrite.")
    else:
        with open(eval_file, "w") as f:
            f.write(eval_content)
        print(f"Created: {eval_file}")
    
    # Generate mock_dataset.py
    mock_content = templates["mock_dataset"].format(**template_vars)
    mock_file = task_dir / "mock_dataset.py"
    
    if mock_file.exists() and not args.force:
        print(f"Warning: {mock_file} already exists. Use --force to overwrite.")
    else:
        with open(mock_file, "w") as f:
            f.write(mock_content)
        print(f"Created: {mock_file}")
    
    # Generate __init__.py
    init_file = task_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()
        print(f"Created: {init_file}")
    
    print(f"\nTask '{task_name_lower}' generated successfully!")
    print(f"Next steps:")
    print(f"1. Edit {eval_file} to implement the actual prompt and parsing logic")
    print(f"2. Update mock_dataset.py with representative examples")
    print(f"3. Test with: python -m eval.eval --model hf --model_args 'pretrained=microsoft/phi-2' --tasks flame_{task_name_lower} --limit 5")


def main():
    parser = argparse.ArgumentParser(
        description="Generate FLaME task implementation templates for Evalchemy"
    )
    parser.add_argument(
        "--task-name",
        required=True,
        help="Name of the task (e.g., bizbench, finred, fiqa_task1)"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["classification", "qa", "ner"],
        help="Type of task (classification, qa, or ner)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="HuggingFace dataset name (e.g., 'kensho/bizbench')"
    )
    parser.add_argument(
        "--description",
        help="Brief description of what the task evaluates"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    generate_task(args)


if __name__ == "__main__":
    main()