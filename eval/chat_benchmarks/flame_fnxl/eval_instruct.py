"""
FNXL task implementation for Evalchemy.

This task evaluates Cross-lingual financial numeral extraction with XBRL tagging.
Dataset: gtfintechlab/fnxl
Task type: classification
Metrics: Accuracy, Precision, Recall, F1 Score
"""

import datasets
import json
import re
from typing import Dict, Any, List, Optional, Set

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark


class FlameFnxlBenchmark(BaseBenchmark):
    """FLaME FNXL benchmark for Evalchemy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = "gtfintechlab/fnxl"
        self.max_new_tokens = kwargs.get('max_new_tokens', 256)  # Need more tokens for JSON output
        self.temperature = kwargs.get('temperature', 0.0)
        
    def generate_responses(self, model: LM) -> Optional[Dict[str, Any]]:
        """Generate model responses for FNXL task."""
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
            from .mock_dataset import create_mock_fnxl_dataset
            dataset = create_mock_fnxl_dataset()
            
        instances = []
        
        # Process each example
        for idx, example in enumerate(dataset):
            # Extract fields based on FNXL dataset structure
            sentence = example.get("sentence", "")
            company = example.get("company", "")
            doc_type = example.get("docType", "")
            
            # Create prompt based on FLaME prompt style
            prompt = self._create_prompt(sentence, company, doc_type)
            
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
            "task_name": "flame_fnxl"
        }
        
        return results
        
    def evaluate_responses(self, results: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate model outputs for FNXL task."""
        if results is None:
            return None
            
        outputs = results["outputs"]
        examples = results["examples"]
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_actual = 0
        total_predicted = 0
        
        for output, example in zip(outputs, examples):
            # Parse model output to get predicted tags
            pred_dict = self._parse_json_output(output)
            
            # Get ground truth tags
            true_dict = self._parse_ground_truth(example.get("numerals-tags", "{}"))
            
            # Normalize both dictionaries
            pred_normalized = self._normalize_taglist_json(pred_dict)
            true_normalized = self._normalize_taglist_json(true_dict)
            
            # Compare predictions with ground truth
            tp, fp, fn, actual_count, pred_count = self._compare_taglist_dicts(
                true_normalized, pred_normalized
            )
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_actual += actual_count
            total_predicted += pred_count
            
        # Calculate metrics
        if total_tp + total_fp == 0:
            precision = 0.0
        else:
            precision = total_tp / (total_tp + total_fp)
            
        if total_tp + total_fn == 0:
            recall = 0.0
        else:
            recall = total_tp / (total_tp + total_fn)
            
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        if total_actual + total_predicted - total_tp == 0:
            accuracy = 0.0
        else:
            accuracy = total_tp / (total_actual + total_predicted - total_tp)
            
        return {
            "accuracy_jaccard": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
    def _create_prompt(self, sentence: str, company: str = None, doc_type: str = None) -> str:
        """Create prompt for FNXL task based on FLaME implementation."""
        prompt = f"""
You are an SEC reporting expert. Given a sentence from a financial filing, do two things:
1) Identify every numeral in the sentence.
2) For each numeral, assign the most appropriate US-GAAP XBRL tag based on context.
If no tag is appropriate, label it as "other".

Return only valid JSON in this format:
```json
{{
  "12.0": "us-gaap:Revenue",
  "9.5": "us-gaap:SomeExpense",
  "100.0": "other"
}}```
The sentence is: {sentence}"""
        return prompt
        
    def _parse_json_output(self, output: str) -> Dict:
        """Parse JSON from model output."""
        if not output:
            return {}
            
        # Clean the output and extract JSON
        output = output.strip()
        
        # Look for JSON between ```json and ```
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON-like content
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {}
                
        try:
            # Replace single quotes with double quotes for JSON parsing
            json_str = json_str.replace("'", '"')
            return json.loads(json_str)
        except:
            return {}
            
    def _parse_ground_truth(self, tags_str: str) -> Dict:
        """Parse ground truth tags from string."""
        if isinstance(tags_str, dict):
            return tags_str
            
        try:
            # Replace single quotes with double quotes
            tags_str = str(tags_str).replace("'", '"')
            return json.loads(tags_str)
        except:
            return {}
            
    def _normalize_taglist_json(self, json_input: Dict) -> Dict[str, Set[float]]:
        """
        Convert the JSON dict into: { tag (lowercased): set_of_floats }
        ignoring any non-numeric items.
        """
        normalized = {}
        
        for tag, val_list in json_input.items():
            floats_set = set()
            
            if isinstance(val_list, list):
                for v in val_list:
                    try:
                        if isinstance(v, (int, float)):
                            floats_set.add(float(v))
                        elif isinstance(v, str):
                            val_str = v.replace(",", "").strip()
                            floats_set.add(float(val_str))
                    except ValueError:
                        pass
            else:
                # Single value
                try:
                    if isinstance(val_list, (int, float)):
                        floats_set.add(float(val_list))
                    elif isinstance(val_list, str):
                        val_str = val_list.replace(",", "").strip()
                        floats_set.add(float(val_str))
                except ValueError:
                    pass
                    
            normalized[tag.lower().strip()] = floats_set
            
        return normalized
        
    def _compare_taglist_dicts(self, actual: Dict, predicted: Dict) -> tuple:
        """
        Partial-credit set comparison for "tag -> set_of_floats".
        Returns (tp, fp, fn, total_actual, total_predicted).
        """
        tp = 0
        fp = 0
        fn = 0
        
        all_tags = set(actual.keys()).union(set(predicted.keys()))
        
        for tag in all_tags:
            actual_vals = actual.get(tag, set())
            pred_vals = predicted.get(tag, set())
            
            overlap = actual_vals.intersection(pred_vals)
            tp += len(overlap)
            fp += len(pred_vals - actual_vals)
            fn += len(actual_vals - pred_vals)
            
        total_actual = sum(len(s) for s in actual.values())
        total_predicted = sum(len(s) for s in predicted.values())
        
        return tp, fp, fn, total_actual, total_predicted