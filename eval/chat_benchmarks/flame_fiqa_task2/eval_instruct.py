"""
FIQA_TASK2 task implementation for Evalchemy.

This task evaluates Financial question answering capabilities.
Dataset: gtfintechlab/FiQA_Task2
Task type: qa (question answering)
Metrics: BLEU, ROUGE, BERTScore (simplified to content overlap for now)
"""

import datasets
import numpy as np
from typing import Dict, Any, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class FlameFiqaTask2Benchmark(BaseBenchmark):
    """FLaME FIQA_TASK2 benchmark for Evalchemy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "gtfintechlab/FiQA_Task2"
        self.max_new_tokens = kwargs.get('max_new_tokens', 200)
        self.temperature = kwargs.get('temperature', 0.0)
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for FIQA_TASK2 task."""
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
            from .mock_dataset import create_mock_fiqa_task2_dataset
            dataset = create_mock_fiqa_task2_dataset()
            
        instances = []
        
        # Process each example
        for idx, example in enumerate(dataset):
            question = example.get("question", "")
            
            # Create prompt for financial QA
            prompt = self._create_prompt(question)
            
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
            "task_name": "flame_fiqa_task2"
        }
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for FIQA_TASK2 task."""
        if results is None:
            return None
            
        outputs = results["outputs"]
        examples = results["examples"]
        
        # Simple evaluation metrics for QA
        total_examples = len(outputs)
        valid_answers = 0
        content_overlap_scores = []
        
        for output, example in zip(outputs, examples):
            reference_answer = example.get('answer', '')
            predicted_answer = output.strip()
            
            # Check if answer is valid (non-empty and substantial)
            if predicted_answer and len(predicted_answer.split()) >= 3:
                valid_answers += 1
                
                # Calculate simple content overlap (word-level)
                pred_words = set(predicted_answer.lower().split())
                ref_words = set(reference_answer.lower().split())
                
                if len(ref_words) > 0:
                    overlap = len(pred_words.intersection(ref_words)) / len(ref_words.union(pred_words))
                    content_overlap_scores.append(overlap)
        
        # Calculate metrics
        answer_rate = valid_answers / total_examples if total_examples > 0 else 0.0
        avg_content_overlap = np.mean(content_overlap_scores) if content_overlap_scores else 0.0
        
        return {
            "answer_rate": answer_rate,
            "content_overlap": avg_content_overlap,
            "valid_answers": valid_answers,
            "total_examples": total_examples
        }
        
    def _create_prompt(self, question: str) -> str:
        """Create prompt for FIQA_TASK2."""
        return f"""Discard all previous instructions. Behave like you are an expert financial advisor.

Answer the following financial question clearly and accurately. Provide practical, actionable advice based on sound financial principles.

Question: {question}

Answer:"""