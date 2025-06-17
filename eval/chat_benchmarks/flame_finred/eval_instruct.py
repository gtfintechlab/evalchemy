"""
FINRED task implementation for Evalchemy.

This task evaluates Financial relationship classification with 14 relation types.
Dataset: gtfintechlab/FinRed
Task type: classification
Metrics: Accuracy, Precision, Recall, F1 Score
"""

import datasets
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class FlameFinredBenchmark(BaseBenchmark):
    """FLaME FINRED benchmark for Evalchemy."""
    
    # FinRed relationship types from FLaME
    FINRED_RELATIONSHIPS = [
        "Affiliated-with",
        "Collaborated-with",
        "Compete-with",
        "Customer-of",
        "Illegal-manipulate",
        "Investigated-by",
        "Negative-impression",
        "No-Relation",
        "Operator-of",
        "Owner-of",
        "Positive-impression",
        "Regulator-of",
        "Supplier-of",
        "Unfair-practice",
        "Was-merged-into",
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "gtfintechlab/FinRed"
        self.max_new_tokens = kwargs.get('max_new_tokens', 50)
        self.temperature = kwargs.get('temperature', 0.0)
        
        # Create label mappings
        self.label_to_idx = {label: idx for idx, label in enumerate(self.FINRED_RELATIONSHIPS)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for FINRED task."""
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
            from .mock_dataset import create_mock_finred_dataset
            dataset = create_mock_finred_dataset()
            
        instances = []
        all_prompts_data = []
        
        # Process each example - FinRed has entity pairs
        for idx, example in enumerate(dataset):
            # Extract sentence and entity pairs
            sentence = example.get("sentence", "")
            entities = example.get("entities", [])
            relations = example.get("relations", [])
            
            # Process each entity pair
            for entity_idx, entity_pair in enumerate(entities):
                if len(entity_pair) >= 2:
                    entity1 = entity_pair[0]  # tail entity
                    entity2 = entity_pair[1]  # head entity
                    
                    # Get the corresponding relation
                    relation = relations[entity_idx] if entity_idx < len(relations) else "No-Relation"
                    
                    # Create prompt based on FLaME prompt style
                    prompt = self._create_prompt(sentence, entity1, entity2)
                    
                    # Format messages
                    messages = [{"role": "user", "content": prompt}]
                    formatted = self._prepare_messages(messages, model)
                    
                    # Create instance
                    gen_kwargs = {
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": self.temperature,
                    }
                    
                    # Store prompt data
                    prompt_data = {
                        "sentence": sentence,
                        "entity1": entity1,
                        "entity2": entity2,
                        "true_relation": relation,
                        "idx": idx,
                        "entity_idx": entity_idx
                    }
                    all_prompts_data.append(prompt_data)
                    
                    instance = Instance(
                        "generate_until",
                        prompt_data,
                        (formatted, gen_kwargs),
                        len(instances)
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
            "prompts_data": all_prompts_data,
            "task_name": "flame_finred"
        }
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for FINRED task."""
        if results is None:
            return None
            
        outputs = results["outputs"]
        prompts_data = results["prompts_data"]
        
        # Extract predictions and ground truth
        predictions = []
        ground_truths = []
        
        for output, prompt_data in zip(outputs, prompts_data):
            # Parse model output
            pred_relation = self._parse_output(output)
            pred_label = self.label_to_idx.get(pred_relation, -1)
            predictions.append(pred_label)
            
            # Get ground truth
            true_relation = prompt_data["true_relation"]
            true_label = self.label_to_idx.get(true_relation, -1)
            ground_truths.append(true_label)
            
        # Convert to numpy arrays
        y_true = np.array(ground_truths)
        y_pred = np.array(predictions)
        
        # Filter out invalid predictions
        valid_mask = (y_pred != -1) & (y_true != -1)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
            
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    def _create_prompt(self, sentence: str, entity1: str, entity2: str) -> str:
        """Create prompt for FINRED task based on FLaME implementation."""
        prompt = f"""Classify what relationship {entity2} (the head) has to {entity1} (the tail) within the following sentence:
"{sentence}"

The relationship should match one of the following categories, where the relationship is what the head entity is to the tail entity:
{", ".join(self.FINRED_RELATIONSHIPS)}.

You must output one, and only one, relationship out of the previous list that connects the head entity {entity2} to the tail entity {entity1}. Find what relationship best fits {entity2} 'RELATIONSHIP' {entity1} for this sentence."""
        return prompt
        
    def _parse_output(self, output: str) -> str:
        """Parse model output to extract predicted relation."""
        if not output:
            return "No-Relation"
            
        # Clean the output
        output = output.strip()
        
        # Try to find any of the valid relationships in the output
        # Check for exact matches first
        for relation in self.FINRED_RELATIONSHIPS:
            if relation in output:
                return relation
                
        # Check for case-insensitive matches
        output_lower = output.lower()
        for relation in self.FINRED_RELATIONSHIPS:
            if relation.lower() in output_lower:
                return relation
                
        # Check if just the first line contains a valid relation
        first_line = output.split('\n')[0].strip()
        for relation in self.FINRED_RELATIONSHIPS:
            if relation.lower() == first_line.lower():
                return relation
                
        # Default to No-Relation if nothing found
        return "No-Relation"