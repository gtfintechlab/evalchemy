"""
FinQA (Financial Question Answering) task.

Standalone implementation for Evalchemy.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import re

from eval.task import BaseBenchmark
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import datasets
from eval.chat_benchmarks.flame_utils import (
    format_flame_prompt_consistently,
    create_standard_flame_messages,
    load_dataset_with_fallback
)


class FinQABenchmark(BaseBenchmark):
    """Financial Question Answering benchmark."""
    
    def __init__(
        self,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize FinQA benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.dataset_name = 'ibm/finqa'  # Public version of FinQA
        self.split = 'test'
        
    def load_dataset(self):
        """Load FinQA dataset with fallback to mock data."""
        return load_dataset_with_fallback(
            dataset_name=self.dataset_name,
            split=self.split,
            task_name='finqa',
            debug=self.debug,
            logger=self.logger,
            trust_remote_code=True
        )
    
    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create FinQA prompt combining text, table, and question."""
        # Extract components
        pre_text = example.get('pre_text', [])
        post_text = example.get('post_text', [])
        table = example.get('table', example.get('table_ori', []))
        question = example['question']
        
        # Combine pre_text and post_text
        text_parts = []
        if pre_text:
            if isinstance(pre_text, list):
                text_parts.extend(pre_text)
            else:
                text_parts.append(str(pre_text))
        
        if post_text:
            if isinstance(post_text, list):
                text_parts.extend(post_text)
            else:
                text_parts.append(str(post_text))
        
        # Format table if present
        table_str = ""
        if table and len(table) > 0:
            # Check if it's a list of rows
            if isinstance(table[0], list):
                # Format as a simple table
                table_lines = []
                for row in table:
                    table_lines.append(" | ".join(str(cell) for cell in row))
                table_str = "\n".join(table_lines)
            else:
                # Single row or different format
                table_str = str(table)
        
        # Combine all context
        context_parts = []
        if text_parts:
            context_parts.append(" ".join(text_parts))
        if table_str:
            context_parts.append(f"\nTable:\n{table_str}")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = (
            "Discard all the previous instructions. Behave like you are a financial expert "
            "in question answering. Your task is to answer a financial question based on "
            "the provided context.\n\n"
            f"The context: {context}\n\n"
            f"Question: {question}\n\n"
            "Please provide your reasoning and then state your final answer clearly. "
            "Repeat your final answer at the end of your response."
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
            instance.task_name = "flame_finqa"
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
    
    def parse_output(self, output: str, example: Dict[str, Any]) -> str:
        """Parse model output to extract the final answer."""
        if not output:
            return ""
        
        # Try to find the final answer in various formats
        output_lower = output.lower()
        
        # Look for explicit final answer patterns
        patterns = [
            r"final answer[:\s]+([^\n]+)",
            r"answer[:\s]+([^\n]+)$",
            r"therefore[,\s]+([^\n]+)$",
            r"= ([^\n]+)$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output_lower)
            if match:
                answer = match.group(1).strip()
                # Clean up common suffixes
                answer = re.sub(r'\.$', '', answer)
                return answer
        
        # If no pattern matched, try to extract the last number or short phrase
        # Look for numbers at the end
        numbers = re.findall(r'[-+]?\d*\.?\d+%?', output)
        if numbers:
            # Return the last number found
            return numbers[-1]
        
        # As a last resort, return the last line that's not empty
        lines = output.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) < 50:  # Reasonable length for an answer
                return line
        
        return output.strip()[:50]  # Return first 50 chars if nothing else works
    
    def extract_reference(self, example: Dict[str, Any]) -> str:
        """Extract ground truth answer from example."""
        answer = example.get('answer', example.get('final_answer', ''))
        
        # Handle list answers (some datasets store as list)
        if isinstance(answer, list):
            answer = answer[0] if answer else ''
        
        return str(answer).strip()
    
    def normalize_number(self, text: str) -> Optional[float]:
        """Normalize a text string to a number for comparison."""
        # Remove common formatting
        text = text.strip()
        text = text.replace(',', '')
        text = text.replace('$', '')
        
        # Handle percentages
        is_percentage = text.endswith('%')
        if is_percentage:
            text = text[:-1]
        
        # Try to convert to float
        try:
            value = float(text)
            return value
        except ValueError:
            return None
    
    def numbers_match(self, pred: str, ref: str, tolerance: float = 0.001) -> bool:
        """Check if two number strings match within tolerance."""
        # Try to normalize both as numbers
        pred_num = self.normalize_number(pred)
        ref_num = self.normalize_number(ref)
        
        if pred_num is not None and ref_num is not None:
            # Both are numbers - compare with tolerance
            if ref_num == 0:
                return abs(pred_num) < tolerance
            else:
                return abs(pred_num - ref_num) / abs(ref_num) < tolerance
        
        # If not both numbers, fall back to string comparison
        return pred.lower() == ref.lower()
    
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
        
        # Compute accuracy
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            if self.numbers_match(pred, ref):
                correct += 1
            else:
                self.logger.debug(f"Mismatch: predicted='{pred}', reference='{ref}'")
        
        accuracy = correct / total if total > 0 else 0.0
        
        self.logger.info(f"FinQA Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }