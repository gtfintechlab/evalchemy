"""
FPB (Financial Phrase Bank) sentiment classification task.

Standalone implementation for Evalchemy.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from eval.task import BaseBenchmark
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import datasets
from eval.chat_benchmarks.flame_utils import (
    format_flame_prompt_consistently, 
    create_standard_flame_messages,
    load_dataset_with_fallback,
    parse_classification_output
)


class FPBBenchmark(BaseBenchmark):
    """Financial Phrase Bank sentiment classification benchmark."""
    
    def __init__(
        self,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize FPB benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.dataset_name = 'financial_phrasebank'
        self.config_name = 'sentences_allagree'
        self.split = 'train'  # FPB only has train split
        self.label_to_id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_dataset(self):
        """Load FPB dataset with fallback to mock data."""
        try:
            # Try to load the real dataset
            dataset = datasets.load_dataset(
                self.dataset_name,
                self.config_name,
                split=self.split,
                trust_remote_code=True
            )
            
            if self.debug:
                # Use only first 5 examples for debugging
                dataset = dataset.select(range(min(5, len(dataset))))
                self.logger.info(f"Debug mode: Using only {len(dataset)} examples")
            
            self.logger.info(f"Loaded {len(dataset)} examples from Financial Phrase Bank")
            return dataset
        except Exception as e:
            self.logger.warning(f"Failed to load real dataset: {e}")
            self.logger.info("Using mock FPB dataset for testing")
            
            # Use centralized mock data
            from ..flame_central_mock_data import get_mock_data
            mock_data = get_mock_data('fpb')
            
            if self.debug:
                mock_data = mock_data[:5]
            
            # Convert to dataset-like object
            class MockDataset:
                def __init__(self, data):
                    self.data = data
                
                def __iter__(self):
                    return iter(self.data)
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
                
                def select(self, indices):
                    selected_data = [self.data[i] for i in indices]
                    return MockDataset(selected_data)
            
            return MockDataset(mock_data)
    
    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create FPB sentiment classification prompt."""  
        prompt = (
            f"Discard all the previous instructions. Behave like you are an expert "
            f"sentence sentiment classifier.\n\n"
            f"Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL' class. "
            f"Label 'NEGATIVE' if it corresponds to negative sentiment, 'POSITIVE' if positive, "
            f"or 'NEUTRAL' if neutral.\n\n"
            f"Provide the label in the first line and provide a short explanation "
            f"in the second line.\n\n"
            f"This is the sentence: {example['sentence']}"
        )
        return prompt
    
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate model responses for all dataset instances."""
        # Load dataset
        dataset = self.load_dataset()
        
        instances = []
        for idx, example in enumerate(dataset):
            # Create prompt
            prompt = self.create_prompt(example)
            
            # Format prompt using standardized FLaME approach
            messages = create_standard_flame_messages(prompt)
            formatted_prompt = format_flame_prompt_consistently(messages, model, self)
            
            # Create instance
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": False if self.temperature == 0 else True,
            }
            
            instance = Instance(
                request_type="generate_until",
                doc=example,
                arguments=(formatted_prompt, gen_kwargs),
                idx=idx,
            )
            instance.task_name = "flame_fpb"
            instances.append(instance)
        
        self.logger.info(f"Generating responses for {len(instances)} instances")
        
        # Generate outputs
        outputs = self.compute(model, instances)
        
        # Only process on main rank
        if model.rank != 0:
            return None
        
        # Package results
        results = {
            "outputs": outputs,
            "instances": instances,
            "dataset": list(dataset),
        }
        
        return results
    
    def parse_output(self, output: str, example: Dict[str, Any]) -> int:
        """Parse model output to extract predicted label."""
        if not output:
            return -1
        
        # Get first line (FPB expects label on first line, explanation on second)
        lines = output.strip().split('\n')
        first_line = lines[0] if lines else ""
        
        # Use standardized parsing
        valid_labels = ["NEGATIVE", "POSITIVE", "NEUTRAL"]
        parsed_label = parse_classification_output(first_line, valid_labels, case_sensitive=False)
        
        if parsed_label:
            return self.label_to_id.get(parsed_label.lower(), -1)
        return -1
    
    def extract_reference(self, example: Dict[str, Any]) -> int:
        """Extract ground truth label from example."""
        label = example.get('label')
        
        # Handle numeric labels
        if isinstance(label, (int, float)):
            return int(label)
        
        # Handle string labels
        if isinstance(label, str):
            return self.label_to_id.get(label.lower(), -1)
        
        return -1
    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model responses and compute metrics."""
        if results is None:
            return None
        
        outputs = results["outputs"]
        dataset = results["dataset"]
        
        # Parse predictions and references
        predictions = []
        references = []
        
        for output, example in zip(outputs, dataset):
            pred = self.parse_output(output, example)
            ref = self.extract_reference(example)
            
            predictions.append(pred)
            references.append(ref)
        
        # Filter out invalid predictions
        valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                       if p != -1 and r != -1]
        
        if not valid_pairs:
            self.logger.warning("No valid predictions to evaluate")
            return {"accuracy": 0.0, "f1": 0.0}
        
        valid_preds, valid_refs = zip(*valid_pairs)
        
        # Compute metrics
        accuracy = accuracy_score(valid_refs, valid_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_refs, valid_preds, average='weighted', zero_division=0
        )
        
        # Log results
        self.logger.info(f"Evaluated {len(valid_pairs)} valid predictions out of {len(predictions)}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Weighted F1: {f1:.4f}")
        
        # Log confusion matrix for debugging
        if self.debug:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(valid_refs, valid_preds, labels=[0, 1, 2])
            self.logger.debug(f"Confusion Matrix:\n{cm}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "valid_predictions": len(valid_pairs),
            "total_predictions": len(predictions),
        }