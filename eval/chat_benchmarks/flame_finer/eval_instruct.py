"""
FINER (Fine-grained financial entity recognition) task for Evalchemy.

This module implements the FINER task from FLaME within the Evalchemy framework.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from eval.task import BaseBenchmark
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import datasets


class FinerBenchmark(BaseBenchmark):
    """FINER benchmark for Evalchemy."""
    
    def __init__(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize FINER benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.dataset_name = 'gtfintechlab/finer-ord-bio'
        self.split = 'test'
        
        # BIO tag mappings for FINER (Person, Location, Organisation)
        self.label_to_id = {
            "B-PER": 0, "I-PER": 1,
            "B-LOC": 2, "I-LOC": 3, 
            "B-ORG": 4, "I-ORG": 5,
            "O": 6
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_dataset(self):
        """Load FINER dataset with fallback to mock data."""
        try:
            # Try to load the real dataset
            dataset = datasets.load_dataset(
                self.dataset_name,
                split=self.split,
                trust_remote_code=True
            )
            
            if self.debug:
                # Use only first 5 examples for debugging
                dataset = dataset.select(range(min(5, len(dataset))))
                self.logger.info(f"Debug mode: Using only {len(dataset)} examples")
            
            self.logger.info(f"Loaded {len(dataset)} examples from FINER dataset")
            return dataset
        except Exception as e:
            self.logger.warning(f"Failed to load real dataset: {e}")
            self.logger.info("Using mock FINER dataset for testing")
            
            # Use mock data
            from .mock_dataset import get_mock_finer_data
            mock_data = get_mock_finer_data()
            
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
            
            return MockDataset(mock_data)
    
    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create FINER sequence labeling prompt."""
        tokens = example.get('tokens', [])
        
        # Format tokens as a numbered list
        token_list = "\n".join([f"{i+1}. {token}" for i, token in enumerate(tokens)])
        
        prompt = f"""Discard all previous instructions. Behave like you are an expert financial named entity recognition system.

Your task is to identify 'Person', 'Location', and 'Organisation' entities from the following tokenized financial text.

For each token, provide one of these labels:
- B-PER: Beginning of a Person entity
- I-PER: Inside/continuation of a Person entity  
- B-LOC: Beginning of a Location entity
- I-LOC: Inside/continuation of a Location entity
- B-ORG: Beginning of an Organisation entity
- I-ORG: Inside/continuation of an Organisation entity
- O: Other (not an entity)

Tokens:
{token_list}

Provide your answer as a numbered list matching the input tokens, with one label per line:
1. <label>
2. <label>
..."""
        return prompt
    
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate model responses for all dataset instances."""
        # Load dataset
        dataset = self.load_dataset()
        
        instances = []
        for idx, example in enumerate(dataset):
            # Create prompt
            prompt = self.create_prompt(example)
            
            # Format prompt - use direct prompt for non-chat models
            # Check if model has chat template support
            has_chat_template = hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'chat_template') and model.tokenizer.chat_template is not None
            
            if has_chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self._prepare_messages(messages, model)
            else:
                # For models without chat templates, use the prompt directly
                formatted_prompt = prompt
            
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
            instance.task_name = "flame_finer"
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
    
    def parse_output(self, output: str, example: Dict[str, Any]) -> List[int]:
        """Parse model output to extract predicted BIO tags for all tokens."""
        if not output:
            return []
        
        tokens = example.get('tokens', [])
        predicted_labels = []
        
        # Split output into lines and extract labels
        lines = output.strip().split('\n')
        
        for i, line in enumerate(lines):
            if i >= len(tokens):
                break
                
            # Extract label from line (format: "1. B-PER" or "B-PER")
            line_clean = line.strip().upper()
            
            # Remove numbering if present
            if '. ' in line_clean:
                line_clean = line_clean.split('. ', 1)[1]
            
            # Find matching label
            label_found = False
            for label in self.label_to_id.keys():
                if label in line_clean:
                    predicted_labels.append(self.label_to_id[label])
                    label_found = True
                    break
            
            if not label_found:
                # Default to 'O' (Other) for unparseable labels
                predicted_labels.append(self.label_to_id['O'])
        
        # Pad or truncate to match token count
        while len(predicted_labels) < len(tokens):
            predicted_labels.append(self.label_to_id['O'])
        
        return predicted_labels[:len(tokens)]
    
    def extract_reference(self, example: Dict[str, Any]) -> List[int]:
        """Extract ground truth BIO tags from example."""
        tags = example.get('tags', example.get('ner_tags', []))
        
        # Convert to label IDs
        reference_labels = []
        for tag in tags:
            if isinstance(tag, int):
                # Tags are numeric IDs - use them directly if in valid range
                if 0 <= tag < len(self.id_to_label):
                    reference_labels.append(tag)
                else:
                    reference_labels.append(self.label_to_id['O'])
            elif isinstance(tag, str):
                reference_labels.append(self.label_to_id.get(tag.upper(), self.label_to_id['O']))
            else:
                reference_labels.append(self.label_to_id['O'])
        
        return reference_labels
    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model responses and compute metrics."""
        if results is None:
            return None
        
        outputs = results["outputs"]
        dataset = results["dataset"]
        
        # Parse predictions and references for sequence labeling
        all_predictions = []
        all_references = []
        
        for output, example in zip(outputs, dataset):
            pred_sequence = self.parse_output(output, example)
            ref_sequence = self.extract_reference(example)
            
            # Only use examples where we have both predictions and references
            if pred_sequence and ref_sequence and len(pred_sequence) == len(ref_sequence):
                all_predictions.extend(pred_sequence)
                all_references.extend(ref_sequence)
        
        if not all_predictions:
            self.logger.warning("No valid sequence predictions to evaluate")
            return {"accuracy": 0.0, "f1": 0.0}
        
        # Compute token-level metrics
        accuracy = accuracy_score(all_references, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_references, all_predictions, average='macro', zero_division=0
        )
        
        # Log results
        self.logger.info(f"Evaluated {len(all_predictions)} tokens from {len(outputs)} sequences")
        self.logger.info(f"Token-level Accuracy: {accuracy:.4f}")
        self.logger.info(f"Macro-averaged F1: {f1:.4f}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "token_count": len(all_predictions),
            "sequence_count": len(outputs),
        }
