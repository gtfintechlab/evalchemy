"""
CAUSAL_CLASSIFICATION (Causal relationship classification) task for Evalchemy.

This module implements the CAUSAL_CLASSIFICATION task from FLaME within the Evalchemy framework.
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
    load_dataset_with_fallback
)


class CausalClassificationBenchmark(BaseBenchmark):
    """CAUSAL_CLASSIFICATION benchmark for Evalchemy."""
    
    def __init__(
        self,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize CAUSAL_CLASSIFICATION benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.dataset_name = 'gtfintechlab/CausalClassification'
        self.split = 'test'
        
        # CausalClassification label mappings
        self.label_to_id = {"NON_CAUSAL": 0, "DIRECT_CAUSAL": 1, "INDIRECT_CAUSAL": 2}
        self.id_to_label = {0: "Non-causal", 1: "Direct causal", 2: "Indirect causal"}
        
    def load_dataset(self):
        """Load CAUSAL_CLASSIFICATION dataset with fallback to mock data."""
        return load_dataset_with_fallback(
            dataset_name=self.dataset_name,
            split=self.split,
            task_name='causal_classification',
            debug=self.debug,
            logger=self.logger,
            trust_remote_code=True
        )
    
    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create CausalClassification prompt."""
        text = example.get('text', example.get('sentence', ''))
        
        prompt = f"""Discard all the previous instructions. Behave like you are an expert causal classification model.
Below is a sentence. Classify it into one of the following categories:
                0 - Non-causal
                1 - Direct causal
                2 - Indirect causal
                Only return the label number without any additional text.

{text}"""
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
            instance.task_name = "flame_causal_classification"
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
        
        # Clean the output
        output_clean = output.strip()
        
        # Look for numeric labels first (most reliable)
        import re
        numeric_match = re.search(r'\b([012])\b', output_clean)
        if numeric_match:
            label_num = int(numeric_match.group(1))
            if 0 <= label_num <= 2:
                return label_num
        
        # Look for text labels
        output_upper = output_clean.upper()
        if 'NON-CAUSAL' in output_upper or 'NON CAUSAL' in output_upper or 'NONCAUSAL' in output_upper:
            return 0
        elif 'DIRECT' in output_upper and 'CAUSAL' in output_upper:
            return 1
        elif 'INDIRECT' in output_upper and 'CAUSAL' in output_upper:
            return 2
        elif 'DIRECT' in output_upper:
            return 1
        elif 'INDIRECT' in output_upper:
            return 2
        
        # Try to extract just the first digit
        first_digit = re.search(r'\d', output_clean)
        if first_digit:
            digit = int(first_digit.group())
            if 0 <= digit <= 2:
                return digit
        
        return -1
    
    def extract_reference(self, example: Dict[str, Any]) -> int:
        """Extract ground truth label from example."""
        label = example.get('label', example.get('class', example.get('classification')))
        
        # Handle numeric labels
        if isinstance(label, (int, float)):
            label_int = int(label)
            if 0 <= label_int <= 2:
                return label_int
        
        # Handle string labels
        if isinstance(label, str):
            label_upper = label.upper().replace(' ', '_').replace('-', '_')
            return self.label_to_id.get(label_upper, -1)
        
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
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "valid_count": len(valid_pairs),
            "total_count": len(predictions),
        }
