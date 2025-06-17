"""
NUMCLAIM (Numerical claim verification) task for Evalchemy.

This module implements the NUMCLAIM task from FLaME within the Evalchemy framework.
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


class NumclaimBenchmark(BaseBenchmark):
    """NUMCLAIM benchmark for Evalchemy."""
    
    def __init__(
        self,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize NUMCLAIM benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.dataset_name = 'gtfintechlab/Numclaim'
        self.split = 'test'
        
        # NumClaim binary classification labels
        self.label_to_id = {"OUTOFCLAIM": 0, "INCLAIM": 1}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_dataset(self):
        """Load NUMCLAIM dataset with fallback to mock data."""
        return load_dataset_with_fallback(
            dataset_name=self.dataset_name,
            split=self.split,
            task_name="flame_numclaim",
            debug=self.debug,
            logger=self.logger
        )
    
    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create NUMCLAIM classification prompt."""
        sentence = example.get('context', example.get('sentence', example.get('text', '')))
        
        prompt = f"""Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier. Classify the following sentence into 'INCLAIM', or 'OUTOFCLAIM' class. Label 'INCLAIM' if consist of a claim and not just factual past or present information, or 'OUTOFCLAIM' if it has just factual past or present information. Provide the label in the first line and provide a short explanation in the second line.

The sentence: {sentence}"""
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
            instance.task_name = "flame_numclaim"
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
        # Use standardized parsing for binary classification
        valid_labels = ["INCLAIM", "OUTOFCLAIM"]
        parsed_label = parse_classification_output(output, valid_labels, case_sensitive=False)
        
        if parsed_label:
            # Handle variant spellings
            if parsed_label == "OUTOFCLAIM" or parsed_label == "OUT OF CLAIM":
                return self.label_to_id["OUTOFCLAIM"]
            elif parsed_label == "INCLAIM":
                return self.label_to_id["INCLAIM"]
        
        # Fallback: check for keywords that suggest claims vs facts
        if output:
            output_lower = output.lower()
            claim_keywords = ['will', 'could', 'may', 'might', 'expect', 'project', 'forecast', 'estimate', 'believe', 'opinion']
            fact_keywords = ['was', 'were', 'had', 'reported', 'recorded', 'announced', 'stated']
            
            claim_score = sum(1 for kw in claim_keywords if kw in output_lower)
            fact_score = sum(1 for kw in fact_keywords if kw in output_lower)
            
            if claim_score > fact_score:
                return self.label_to_id["INCLAIM"]
            elif fact_score > claim_score:
                return self.label_to_id["OUTOFCLAIM"]
        
        return -1
    
    def extract_reference(self, example: Dict[str, Any]) -> int:
        """Extract ground truth label from example."""
        # NumClaim dataset uses 'response' field for labels
        label = example.get('response', example.get('label'))
        
        # Handle numeric labels (0=OUTOFCLAIM, 1=INCLAIM)
        if isinstance(label, (int, float)):
            return int(label)
        
        # Handle string labels
        if isinstance(label, str):
            label_upper = label.upper().strip()
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
        
        # Compute metrics (binary classification)
        accuracy = accuracy_score(valid_refs, valid_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_refs, valid_preds, average='binary', pos_label=1, zero_division=0
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
