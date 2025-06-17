"""
EDTSUM (Earnings document summarization) task for Evalchemy.

This module implements the EDTSUM question answering task from FLaME.
"""

from typing import Dict, Any, List, Optional
import logging
import re

from eval.task import BaseBenchmark
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import datasets


class EdtsumBenchmark(BaseBenchmark):
    """EDTSUM question answering benchmark."""
    
    def __init__(
        self,
        max_new_tokens: int = 150,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize EDTSUM benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.dataset_name = 'gtfintechlab/edtsum'
        self.split = 'test'
        
    def load_dataset(self):
        """Load EDTSUM dataset with fallback to mock data."""
        try:
            dataset = datasets.load_dataset(
                self.dataset_name,
                split=self.split,
                trust_remote_code=True
            )
            
            if self.debug:
                dataset = dataset.select(range(min(5, len(dataset))))
                self.logger.info(f"Debug mode: Using only {len(dataset)} examples")
            
            self.logger.info(f"Loaded {len(dataset)} examples from EDTSUM dataset")
            return dataset
        except Exception as e:
            self.logger.warning(f"Failed to load real dataset: {e}")
            self.logger.info("Using mock EDTSUM dataset for testing")
            
            from .mock_dataset import get_mock_edtsum_data
            mock_data = get_mock_edtsum_data()
            
            if self.debug:
                mock_data = mock_data[:5]
            
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
        """Create EDTSUM summarization prompt."""
        text = example.get('text', '')
        query = example.get('query', 'Summarize this document concisely.')
        
        # Use the query if it provides summarization instructions, otherwise use default
        if 'summarize' in query.lower() or 'summary' in query.lower():
            prompt = f"""Discard all previous instructions. Behave like you are an expert financial analyst.

{query}

Document:
{text}

Summary:"""
        else:
            prompt = f"""Discard all previous instructions. Behave like you are an expert financial analyst.

Your task is to create a concise summary of the following financial document. Focus on key information, important findings, and crucial details. Keep the summary under 50 words.

Document:
{text}

Summary:"""
        
        return prompt
    
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate model responses for all dataset instances."""
        dataset = self.load_dataset()
        
        instances = []
        for idx, example in enumerate(dataset):
            prompt = self.create_prompt(example)
            
            # Handle chat vs non-chat models
            has_chat_template = hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'chat_template') and model.tokenizer.chat_template is not None
            
            if has_chat_template:
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self._prepare_messages(messages, model)
            else:
                formatted_prompt = prompt
            
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
            instance.task_name = "flame_edtsum"
            instances.append(instance)
        
        self.logger.info(f"Generating responses for {len(instances)} instances")
        outputs = self.compute(model, instances)
        
        if model.rank != 0:
            return None
        
        results = {
            "outputs": outputs,
            "instances": instances,
            "dataset": list(dataset),
        }
        
        return results
    
    def extract_summary(self, text: str) -> str:
        """Extract summary from model output."""
        # Clean up the output text
        text = text.strip()
        
        # Remove any instruction echoing
        if 'summary:' in text.lower():
            parts = text.lower().split('summary:')
            if len(parts) > 1:
                text = parts[-1].strip()
        
        # Remove common prefixes
        prefixes_to_remove = ['summary:', 'here is the summary:', 'the summary is:']
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # Clean up and return
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith('note:'):
                clean_lines.append(line)
        
        return '\n'.join(clean_lines) if clean_lines else text
    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model responses for document summarization quality."""
        if results is None:
            return None
        
        outputs = results["outputs"]
        dataset = results["dataset"]
        
        total_examples = len(outputs)
        valid_summaries = 0
        total_length_ratio = 0
        
        for output, example in zip(outputs, dataset):
            predicted_summary = self.extract_summary(output)
            reference_summary = example.get('answer', '')
            source_text = example.get('text', '')
            
            # Check if summary is valid (non-empty and shorter than source)
            if predicted_summary and len(predicted_summary.strip()) > 0:
                valid_summaries += 1
                
                # Calculate compression ratio
                if len(source_text) > 0:
                    compression_ratio = len(predicted_summary) / len(source_text)
                    total_length_ratio += compression_ratio
        
        # Calculate metrics
        valid_rate = valid_summaries / total_examples if total_examples > 0 else 0.0
        avg_compression_ratio = total_length_ratio / valid_summaries if valid_summaries > 0 else 0.0
        
        # Simple content overlap check (word-level)
        content_overlap_scores = []
        for output, example in zip(outputs, dataset):
            predicted_summary = self.extract_summary(output)
            reference_summary = example.get('answer', '')
            
            pred_words = set(predicted_summary.lower().split())
            ref_words = set(reference_summary.lower().split())
            
            if len(ref_words) > 0:
                overlap = len(pred_words.intersection(ref_words)) / len(ref_words)
                content_overlap_scores.append(overlap)
        
        avg_overlap = sum(content_overlap_scores) / len(content_overlap_scores) if content_overlap_scores else 0.0
        
        self.logger.info(f"EDTSum Evaluation:")
        self.logger.info(f"  Valid summaries: {valid_summaries}/{total_examples} ({valid_rate:.4f})")
        self.logger.info(f"  Avg compression ratio: {avg_compression_ratio:.4f}")
        self.logger.info(f"  Avg content overlap: {avg_overlap:.4f}")
        
        return {
            "valid_rate": valid_rate,
            "compression_ratio": avg_compression_ratio,
            "content_overlap": avg_overlap,
            "valid_summaries": valid_summaries,
            "total_examples": total_examples,
        }
