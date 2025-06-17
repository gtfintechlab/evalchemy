"""
CONVFINQA (Conversational financial QA) task for Evalchemy.

This module implements the CONVFINQA question answering task from FLaME.
"""

from typing import Dict, Any, List, Optional
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


class ConvfinqaBenchmark(BaseBenchmark):
    """CONVFINQA question answering benchmark."""
    
    def __init__(
        self,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize CONVFINQA benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.dataset_name = 'gtfintechlab/convfinqa'
        self.split = 'dev'  # ConvFinQA uses dev split for evaluation
        
    def load_dataset(self):
        """Load CONVFINQA dataset with fallback to mock data."""
        return load_dataset_with_fallback(
            dataset_name=self.dataset_name,
            split=self.split,
            task_name='convfinqa',
            debug=self.debug,
            logger=self.logger,
            trust_remote_code=True
        )
    
    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create CONVFINQA conversational QA prompt."""
        # Build context from pre_text, post_text, and table
        pre_text = ' '.join(example.get('pre_text', []))
        post_text = ' '.join(example.get('post_text', []))
        
        # Format table if present
        table_text = ''
        if 'table_ori' in example and example['table_ori']:
            table_rows = example['table_ori']
            table_text = ' '.join([' '.join(str(cell) for cell in row) for row in table_rows])
        
        # Combine document context
        document = f"{pre_text} {post_text} {table_text}".strip()
        
        # Add previous Q&A context for conversation
        question_0 = example.get('question_0', '')
        answer_0 = example.get('answer_0', '')
        question_1 = example.get('question_1', '')
        
        # Build conversational context
        conversation_context = f"Question 0: {question_0} Answer: {answer_0}. Now answer the following question: {question_1}"
        
        # Create full prompt
        prompt = f"""Discard all previous instructions. You are a financial expert specializing in answering questions.
The context provided includes a previous question and its answer, followed by a new question that you need to answer.
Focus on answering only the final question based on the entire provided context:
{document}.
{conversation_context}
Answer the final question based on the context above. Repeat your final answer at the end of your response."""
        
        return prompt
    
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """Generate model responses for all dataset instances."""
        dataset = self.load_dataset()
        
        instances = []
        for idx, example in enumerate(dataset):
            prompt = self.create_prompt(example)
            
            # Format prompt using standardized FLaME approach
            messages = create_standard_flame_messages(prompt)
            formatted_prompt = format_flame_prompt_consistently(messages, model, self)
            
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
            instance.task_name = "flame_convfinqa"
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
    
    def extract_answer(self, text: str) -> str:
        """Extract numerical answer from model output."""
        if not text:
            return ''
        
        # Look for final answer patterns first
        final_patterns = [
            r'(?:final answer|Final answer|Final Answer)[:\s]+([^\n\.]+)',
            r'(?:answer|Answer|ANSWER)[:\s]+([^\n\.]+)',
            r'(?:the answer is|The answer is)[:\s]*([^\n\.]+)',
        ]
        
        for pattern in final_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                # Extract numerical value from answer text
                num_match = re.search(r'(\d+(?:\.\d+)?%?)', answer_text)
                if num_match:
                    return num_match.group(1)
                return answer_text
        
        # Extract any numerical values from the entire text
        numerical_patterns = [
            r'(\d+(?:\.\d+)?%)',  # Percentages
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)',  # Dollar amounts
            r'(\d+(?:,\d{3})*(?:\.\d+)?)',  # Numbers with commas
            r'(\d+(?:\.\d+)?)',  # Simple decimals
        ]
        
        for pattern in numerical_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return the last numerical value found (often the final answer)
                return matches[-1]
        
        # Fallback: return cleaned text
        return text.strip()[:50]  # Limit length
    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate ConvFinQA responses."""
        if results is None:
            return None
        
        outputs = results["outputs"]
        dataset = results["dataset"]
        
        correct = 0
        total = 0
        valid_predictions = []
        valid_references = []
        
        for output, example in zip(outputs, dataset):
            predicted = self.extract_answer(output)
            reference = str(example.get('answer_1', '')).strip()  # answer_1 is the target
            
            if not reference:  # Skip if no reference answer
                continue
                
            total += 1
            
            # Normalize both predictions and references
            pred_clean = predicted.lower().strip()
            ref_clean = reference.lower().strip()
            
            # Handle percentage normalization
            if pred_clean.endswith('%'):
                pred_clean = pred_clean[:-1]
            if ref_clean.endswith('%'):
                ref_clean = ref_clean[:-1]
            
            # Try exact match first
            is_correct = pred_clean == ref_clean
            
            # Try numerical comparison with tolerance
            if not is_correct:
                try:
                    pred_val = float(pred_clean.replace(',', ''))
                    ref_val = float(ref_clean.replace(',', ''))
                    # Allow small numerical tolerance
                    is_correct = abs(pred_val - ref_val) < 0.01
                except (ValueError, TypeError):
                    pass
            
            if is_correct:
                correct += 1
            
            # Store for additional metrics
            valid_predictions.append(1 if is_correct else 0)
            valid_references.append(1)  # Binary correct/incorrect
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Log detailed results
        self.logger.info(f"ConvFinQA Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
