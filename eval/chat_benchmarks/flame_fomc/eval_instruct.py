"""
FOMC (Federal Open Market Committee) hawkish/dovish classification task.

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


class FOMCBenchmark(BaseBenchmark):
    """FOMC hawkish/dovish/neutral classification benchmark."""
    
    def __init__(
        self,
        max_new_tokens: int = 5,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize FOMC benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.dataset_name = 'gtfintechlab/fomc_communication'
        self.split = 'test'
        self.label_to_id = {"DOVISH": 0, "HAWKISH": 1, "NEUTRAL": 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_dataset(self):
        """Load FOMC dataset with fallback to mock data."""
        return load_dataset_with_fallback(
            dataset_name=self.dataset_name,
            split=self.split,
            task_name='fomc',
            debug=self.debug,
            logger=self.logger,
            trust_remote_code=True
        )
    
    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create FOMC classification prompt."""
        prompt = (
            f"Classify the following Federal Reserve statement as HAWKISH "
            f"(indicating a restrictive monetary policy stance), DOVISH "
            f"(indicating an accommodative monetary policy stance), or "
            f"NEUTRAL (balanced stance).\n\n"
            f"Statement: {example['sentence']}\n\n"
            f"Provide only one word as your answer: HAWKISH, DOVISH, or NEUTRAL."
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
            instance.task_name = "flame_fomc"
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
        valid_labels = list(self.label_to_id.keys())
        parsed_label = parse_classification_output(output, valid_labels, case_sensitive=False)
        
        if parsed_label:
            return self.label_to_id.get(parsed_label, -1)
        return -1
    
    def extract_reference(self, example: Dict[str, Any]) -> int:
        """Extract ground truth label from example."""
        label = example.get('label', example.get('fomc_label', ''))
        
        # Handle string labels
        if isinstance(label, str):
            return self.label_to_id.get(label.upper(), -1)
        
        # Handle numeric labels
        if isinstance(label, (int, float)):
            return int(label)
        
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
        
        # Log sample predictions
        self.logger.info(f"FOMC Evaluation: {len(valid_pairs)}/{len(predictions)} valid predictions")
        self.logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Log a few examples
        for i in range(min(3, len(outputs))):
            pred_label = self.id_to_label.get(predictions[i], "INVALID")
            ref_label = self.id_to_label.get(references[i], "INVALID")
            self.logger.debug(f"Example {i}: Output='{outputs[i][:50]}...', "
                             f"Predicted={pred_label}, Reference={ref_label}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "valid_count": len(valid_pairs),
            "total_count": len(predictions),
        }