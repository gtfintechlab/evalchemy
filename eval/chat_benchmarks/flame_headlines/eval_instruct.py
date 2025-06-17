"""
HEADLINES (Financial headline classification) task for Evalchemy.

This module implements the HEADLINES task from FLaME within the Evalchemy framework.
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


class HeadlinesBenchmark(BaseBenchmark):
    """HEADLINES benchmark for Evalchemy."""
    
    def __init__(
        self,
        max_new_tokens: int = 20,  # Reduced for simple classification
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        use_mock: bool = False,  # Force mock data for testing
        **kwargs
    ):
        """Initialize HEADLINES benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.use_mock = use_mock
        # Headlines dataset - using gold price movement classification
        self.dataset_name = 'gtfintechlab/headlines'  # Placeholder until confirmed
        self.split = 'test'
        
        # Label mappings for price movement direction
        self.label_to_id = {"DOWN": 0, "NEUTRAL": 1, "UP": 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_dataset(self):
        """Load HEADLINES dataset with fallback to mock data."""
        # Use centralized loading with fallback
        return load_dataset_with_fallback(
            dataset_name=self.dataset_name,
            split=self.split,
            task_name='headlines',
            debug=self.debug,
            logger=self.logger,
            trust_remote_code=True
        )
    
    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create HEADLINES classification prompt."""
        # Get the headline text - FLaME headlines dataset uses 'News' field (capital N)
        headline = example.get('News', example.get('news', example.get('headline', example.get('text', ''))))
        
        prompt = f"""Classify the following financial headline based on what it implies about gold price movement.

An UP headline suggests gold prices are rising or will rise.
A DOWN headline suggests gold prices are falling or will fall.
A NEUTRAL headline suggests no clear direction or stable prices.

Headline: {headline}

Respond with only one word: UP, DOWN, or NEUTRAL."""
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
            instance.task_name = "flame_headlines"
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
            self.logger.warning("Empty output from model")
            return -1
        
        # Use standardized parsing
        valid_labels = list(self.label_to_id.keys())  # ["DOWN", "NEUTRAL", "UP"]
        parsed_label = parse_classification_output(output, valid_labels, case_sensitive=False)
        
        if parsed_label:
            return self.label_to_id.get(parsed_label, -1)
        
        # If no match found, log what we got
        if self.debug:
            self.logger.warning(f"Could not parse label from output: {output[:100]}")
        
        return -1
    
    def extract_reference(self, example: Dict[str, Any]) -> int:
        """Extract ground truth label from example."""
        # Headlines dataset has DirectionUp, DirectionConstant, DirectionDown fields
        # These are binary indicators (0 or 1) for the true direction
        
        if 'DirectionUp' in example:
            # FLaME headlines format
            if example.get('DirectionUp', 0) == 1:
                return self.label_to_id['UP']
            elif example.get('DirectionDown', 0) == 1:
                return self.label_to_id['DOWN']
            elif example.get('DirectionConstant', 0) == 1:
                return self.label_to_id['NEUTRAL']
            else:
                self.logger.warning("No direction label found in example")
                return -1
        
        # Fallback for other formats
        label = example.get('label', example.get('sentiment', ''))
        
        # Handle numeric labels
        if isinstance(label, (int, float)):
            return int(label)
        
        # Handle string labels
        if isinstance(label, str):
            label_upper = label.upper()
            if label_upper in self.label_to_id:
                return self.label_to_id[label_upper]
        
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