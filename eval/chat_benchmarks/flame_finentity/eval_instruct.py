"""
FINENTITY (Financial entity recognition with sentiment) task for Evalchemy.

This module implements the FINENTITY task from FLaME within the Evalchemy framework.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from eval.task import BaseBenchmark
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import datasets


class FinentityBenchmark(BaseBenchmark):
    """FINENTITY benchmark for Evalchemy."""
    
    def __init__(
        self,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize FINENTITY benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.dataset_name = 'gtfintechlab/finentity'
        self.dataset_config = '5768'  # Use the 5768 config
        self.split = 'test'
        
        # Sentiment label mappings for FinEntity
        self.label_to_id = {"POSITIVE": 0, "NEGATIVE": 1, "NEUTRAL": 2}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
    def load_dataset(self):
        """Load FINENTITY dataset with fallback to mock data."""
        try:
            # Try to load the real dataset
            dataset = datasets.load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                split=self.split,
                trust_remote_code=True
            )
            
            if self.debug:
                # Use only first 5 examples for debugging
                dataset = dataset.select(range(min(5, len(dataset))))
                self.logger.info(f"Debug mode: Using only {len(dataset)} examples")
            
            self.logger.info(f"Loaded {len(dataset)} examples from FINENTITY dataset")
            return dataset
        except Exception as e:
            self.logger.warning(f"Failed to load real dataset: {e}")
            self.logger.info("Using mock FINENTITY dataset for testing")
            
            # Use mock data
            from .mock_dataset import get_mock_finentity_data
            mock_data = get_mock_finentity_data()
            
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
        """Create FinEntity prompt for entity recognition and sentiment."""
        content = example.get('content', '')
        
        prompt = f"""Discard all previous instructions. Behave like you are an expert financial named entity recognition and sentiment analysis system.

Your task is to identify company/organization entities in the financial text and classify their sentiment as Positive, Negative, or Neutral.

For each company or organization entity found, provide:
1. The entity name (value)
2. Start position (character index)
3. End position (character index) 
4. Sentiment label (Positive, Negative, or Neutral)

Text: {content}

Provide your answer as a JSON list in this format:
[
  {{
    "value": "entity_name",
    "start": start_index,
    "end": end_index,
    "label": "Positive/Negative/Neutral"
  }}
]

If no entities are found, return an empty list []."""
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
            instance.task_name = "flame_finentity"
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
    
    def parse_output(self, output: str, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse model output to extract predicted entities with sentiment."""
        import json
        import re
        
        if not output:
            return []
        
        # Try to extract JSON from the output
        try:
            # Look for JSON pattern in the output
            json_match = re.search(r'\[(.*?)\]', output, re.DOTALL)
            if json_match:
                json_str = '[' + json_match.group(1) + ']'
                entities = json.loads(json_str)
                
                # Validate and clean the entities
                valid_entities = []
                for entity in entities:
                    if isinstance(entity, dict) and all(k in entity for k in ['value', 'start', 'end', 'label']):
                        # Normalize sentiment label
                        label = entity['label'].upper()
                        if label in self.label_to_id:
                            valid_entities.append({
                                'value': str(entity['value']),
                                'start': int(entity['start']),
                                'end': int(entity['end']),
                                'label': label
                            })
                return valid_entities
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Fallback: try to extract entities from simple text format
        entities = []
        lines = output.strip().split('\n')
        for line in lines:
            # Look for patterns like "Apple: Positive (0-5)"
            match = re.match(r'([^:]+):\s*(Positive|Negative|Neutral)', line, re.IGNORECASE)
            if match:
                entity_name = match.group(1).strip()
                sentiment = match.group(2).upper()
                # For fallback, we can't get exact positions, so use 0
                entities.append({
                    'value': entity_name,
                    'start': 0,
                    'end': len(entity_name),
                    'label': sentiment
                })
        
        return entities
    
    def extract_reference(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract ground truth entities from example."""
        annotations = example.get('annotations', [])
        
        reference_entities = []
        for ann in annotations:
            if isinstance(ann, dict) and 'value' in ann and 'label' in ann:
                reference_entities.append({
                    'value': ann['value'],
                    'start': ann.get('start', 0),
                    'end': ann.get('end', 0),
                    'label': ann['label'].upper()
                })
        
        return reference_entities
    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model responses and compute metrics."""
        if results is None:
            return None
        
        outputs = results["outputs"]
        dataset = results["dataset"]
        
        # Parse predictions and references for entity matching
        total_examples = len(outputs)
        total_predicted_entities = 0
        total_reference_entities = 0
        correct_entities = 0
        
        for output, example in zip(outputs, dataset):
            pred_entities = self.parse_output(output, example)
            ref_entities = self.extract_reference(example)
            
            total_predicted_entities += len(pred_entities)
            total_reference_entities += len(ref_entities)
            
            # Count exact matches (entity value and sentiment must match)
            for pred in pred_entities:
                for ref in ref_entities:
                    if (pred['value'].lower() == ref['value'].lower() and 
                        pred['label'] == ref['label']):
                        correct_entities += 1
                        break
        
        # Compute entity-level metrics
        precision = correct_entities / total_predicted_entities if total_predicted_entities > 0 else 0.0
        recall = correct_entities / total_reference_entities if total_reference_entities > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # For accuracy, use percentage of examples with at least one correct entity
        examples_with_correct = 0
        for output, example in zip(outputs, dataset):
            pred_entities = self.parse_output(output, example)
            ref_entities = self.extract_reference(example)
            
            found_match = False
            for pred in pred_entities:
                for ref in ref_entities:
                    if (pred['value'].lower() == ref['value'].lower() and 
                        pred['label'] == ref['label']):
                        found_match = True
                        break
                if found_match:
                    break
            
            if found_match or (len(pred_entities) == 0 and len(ref_entities) == 0):
                examples_with_correct += 1
        
        accuracy = examples_with_correct / total_examples if total_examples > 0 else 0.0
        
        # Log results
        self.logger.info(f"Evaluated {total_examples} examples")
        self.logger.info(f"Total predicted entities: {total_predicted_entities}")
        self.logger.info(f"Total reference entities: {total_reference_entities}")
        self.logger.info(f"Correct entities: {correct_entities}")
        self.logger.info(f"Entity-level Precision: {precision:.4f}")
        self.logger.info(f"Entity-level Recall: {recall:.4f}")
        self.logger.info(f"Entity-level F1: {f1:.4f}")
        self.logger.info(f"Example-level Accuracy: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct_entities": correct_entities,
            "predicted_entities": total_predicted_entities,
            "reference_entities": total_reference_entities,
        }
