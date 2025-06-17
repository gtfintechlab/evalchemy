"""
Utility functions for FLaME task implementations.

This module provides common utilities and helper functions used across
all FLaME task implementations to ensure consistency and reduce code duplication.
"""

from typing import Dict, List, Union
from lm_eval.api.model import LM


def prepare_flame_prompt(messages: List[Dict[str, str]], model: LM) -> str:
    """
    Prepare a prompt for FLaME tasks with consistent chat template handling.
    
    This function provides a standardized way to handle chat templates across
    all FLaME tasks, ensuring consistent behavior regardless of model type.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Language model instance
        
    Returns:
        Formatted prompt string ready for model inference
        
    Usage:
        messages = [{"role": "user", "content": "Your prompt here"}]
        formatted_prompt = prepare_flame_prompt(messages, model)
    """
    # Check if model supports chat templates
    has_chat_template = (
        hasattr(model, 'tokenizer') and 
        hasattr(model.tokenizer, 'chat_template') and 
        model.tokenizer.chat_template is not None
    )
    
    if has_chat_template:
        # Use Evalchemy's _prepare_messages method for chat template models
        # Note: We can't call _prepare_messages directly from here since it's a method
        # of BaseBenchmark. The caller should use this check and call _prepare_messages.
        return model.apply_chat_template(messages)
    else:
        # For non-chat models, extract content from user message
        # This assumes the last message is the main prompt
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        return user_messages[-1] if user_messages else ""


def has_chat_template_support(model: LM) -> bool:
    """
    Check if a model supports chat templates.
    
    Args:
        model: Language model instance
        
    Returns:
        True if model supports chat templates, False otherwise
    """
    return (
        hasattr(model, 'tokenizer') and 
        hasattr(model.tokenizer, 'chat_template') and 
        model.tokenizer.chat_template is not None
    )


def format_flame_prompt_consistently(messages: List[Dict[str, str]], model: LM, benchmark_instance) -> str:
    """
    Format a prompt with consistent chat template handling for FLaME tasks.
    
    This is the recommended method for all FLaME tasks to ensure consistent
    prompt formatting across different model types.
    
    Args:
        messages: List of message dictionaries
        model: Language model instance
        benchmark_instance: Instance of BaseBenchmark (needed for _prepare_messages)
        
    Returns:
        Formatted prompt string
    """
    if has_chat_template_support(model):
        # Use BaseBenchmark's _prepare_messages for proper chat template handling
        return benchmark_instance._prepare_messages(messages, model)
    else:
        # For non-chat models, extract the user content
        user_messages = [msg['content'] for msg in messages if msg['role'] == 'user']
        return user_messages[-1] if user_messages else ""


def create_standard_flame_messages(prompt_content: str, system_instruction: str = None) -> List[Dict[str, str]]:
    """
    Create standardized message structure for FLaME tasks.
    
    Args:
        prompt_content: The main prompt content
        system_instruction: Optional system instruction
        
    Returns:
        List of message dictionaries in standard format
    """
    messages = []
    
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
        
    messages.append({"role": "user", "content": prompt_content})
    
    return messages


def standardize_debug_params(debug: bool = False, limit: int = None, dataset_size: int = None) -> Dict[str, Union[bool, int]]:
    """
    Standardize debug and limiting parameters across FLaME tasks.
    
    Args:
        debug: Whether debug mode is enabled
        limit: Maximum number of samples to process (None for no limit)
        dataset_size: Size of the full dataset
        
    Returns:
        Dictionary with standardized debug parameters
    """
    # If debug is True but no limit specified, use a reasonable default
    if debug and limit is None:
        limit = min(5, dataset_size) if dataset_size else 5
    
    # If limit is specified but not debug, enable debug mode
    if limit is not None and not debug:
        debug = True
        
    return {
        "debug": debug,
        "limit": limit,
        "effective_size": min(limit, dataset_size) if limit and dataset_size else dataset_size
    }


def load_dataset_with_fallback(dataset_name: str, split: str = "test", task_name: str = None, 
                             debug: bool = False, logger = None, trust_remote_code: bool = True):
    """
    Load dataset with automatic fallback to centralized mock data.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to load
        task_name: Task name for mock data fallback
        debug: Whether debug mode is enabled
        logger: Logger instance for status messages
        trust_remote_code: Whether to trust remote code in dataset
        
    Returns:
        Dataset object (real or mock)
    """
    import datasets
    from .flame_central_mock_data import get_mock_data
    
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    try:
        # Try to load the real dataset
        dataset = datasets.load_dataset(
            dataset_name,
            split=split,
            trust_remote_code=trust_remote_code
        )
        
        if debug:
            # Use only first 5 examples for debugging
            dataset = dataset.select(range(min(5, len(dataset))))
            logger.info(f"Debug mode: Using only {len(dataset)} examples")
        
        logger.info(f"Loaded {len(dataset)} examples from {dataset_name}")
        return dataset
        
    except Exception as e:
        logger.warning(f"Failed to load real dataset: {e}")
        
        if task_name:
            logger.info(f"Using mock {task_name} dataset for testing")
            
            # Use centralized mock data
            mock_data = get_mock_data(task_name)
            
            if debug:
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
                
                def select(self, indices):
                    selected_data = [self.data[i] for i in indices]
                    return MockDataset(selected_data)
            
            return MockDataset(mock_data)
        else:
            raise e


def parse_classification_output(output: str, valid_labels: List[str], 
                              case_sensitive: bool = False) -> str:
    """
    Parse model output for classification tasks with standardized logic.
    
    Args:
        output: Raw model output
        valid_labels: List of valid label strings
        case_sensitive: Whether label matching should be case sensitive
        
    Returns:
        Parsed label string or None if no valid label found
    """
    if not output:
        return None
    
    # Clean and optionally normalize case
    output = output.strip()
    if not case_sensitive:
        output = output.upper()
        valid_labels = [label.upper() for label in valid_labels]
    
    # Take first word only for single-word classification
    first_word = output.split()[0] if output else ""
    
    # Check if first word matches any valid label
    if first_word in valid_labels:
        return first_word
    
    # Fallback: check if any valid label appears in the output
    for label in valid_labels:
        if label in output:
            return label
    
    return None


def extract_label_with_llm(output: str, task_type: str, valid_labels: List[str] = None, 
                          model = None, benchmark = None) -> str:
    """
    Use LLM to extract clean label from model output (FLaME-style extraction).
    
    Args:
        output: Raw model output that may contain the label
        task_type: Type of task (e.g., 'classification', 'qa', 'fomc', 'fpb')
        valid_labels: List of valid labels for classification tasks
        model: LM model instance for extraction (optional)
        benchmark: Benchmark instance for accessing utilities (optional)
        
    Returns:
        Extracted label or answer string
    """
    if not output or not model:
        return parse_classification_output(output, valid_labels) if valid_labels else ""
    
    # Create extraction prompt based on task type
    if task_type == 'fomc':
        extraction_prompt = f"""Extract the classification label from the following LLM response. The label should be one of the following: 'HAWKISH', 'DOVISH', or 'NEUTRAL'.

Here is the LLM response to analyze:
"{output}"
Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation."""
    
    elif task_type == 'fpb':
        extraction_prompt = f"""Based on the following list of labels: 'NEGATIVE', 'POSITIVE', or 'NEUTRAL', extract the most relevant label from the following response:
"{output}"
Provide only the label that best matches the response. Only output alphanumeric characters and spaces. Do not include any special characters or punctuation."""
    
    elif task_type == 'qa':
        extraction_prompt = f"""You will receive a response from a language model that may include a numerical answer within its text.
Your task is to extract and return only the main/final answer. This could be represented as an integer, decimal, percentage, or text.
Respond with whatever is labeled as the final answer, if that exists, even if that contains text. Otherwise, stick to numerical answers.
Do not include any additional text or formatting.

Model Response: {output}

Please respond with the final answer. If a final answer was not provided, respond NA."""
    
    elif valid_labels:
        # Generic classification extraction
        labels_str = "', '".join(valid_labels)
        extraction_prompt = f"""Extract the classification label from the following response. The label should be one of: '{labels_str}'.

Response: "{output}"

Provide only the label that best matches the response, exactly as it appears in the list above."""
    
    else:
        # Fallback to non-LLM parsing
        return output.strip()
    
    # Use the model to extract the label
    if benchmark and hasattr(benchmark, '_prepare_messages'):
        messages = [{"role": "user", "content": extraction_prompt}]
        formatted_prompt = benchmark._prepare_messages(messages, model)
    else:
        formatted_prompt = extraction_prompt
    
    # Create minimal generation kwargs
    gen_kwargs = {
        "max_new_tokens": 20,  # Short response expected
        "temperature": 0.0,
        "do_sample": False,
    }
    
    # Generate extraction
    try:
        if hasattr(model, 'generate_until'):
            extracted = model.generate_until([[formatted_prompt, gen_kwargs]])[0]
        else:
            # Fallback to non-LLM extraction
            extracted = parse_classification_output(output, valid_labels) if valid_labels else output
    except Exception as e:
        # Fallback to non-LLM extraction
        extracted = parse_classification_output(output, valid_labels) if valid_labels else output
    
    # Clean and validate the extracted label
    if valid_labels:
        cleaned = parse_classification_output(extracted, valid_labels)
        return cleaned if cleaned else extracted.strip()
    else:
        return extracted.strip()


def compare_answers_with_tolerance(pred: str, ref: str, model = None, benchmark = None) -> bool:
    """
    Compare answers with numeric tolerance using LLM (FLaME-style evaluation).
    
    Args:
        pred: Predicted answer
        ref: Reference answer
        model: LM model instance for comparison (optional)
        benchmark: Benchmark instance for accessing utilities (optional)
        
    Returns:
        True if answers match within tolerance, False otherwise
    """
    # First try simple comparison
    if pred.strip().lower() == ref.strip().lower():
        return True
    
    # Try numeric comparison with tolerance
    pred_clean = pred.strip().replace(',', '').replace('$', '')
    ref_clean = ref.strip().replace(',', '').replace('$', '')
    
    # Handle percentages
    if pred_clean.endswith('%'):
        pred_clean = pred_clean[:-1]
    if ref_clean.endswith('%'):
        ref_clean = ref_clean[:-1]
    
    try:
        pred_val = float(pred_clean)
        ref_val = float(ref_clean)
        
        # Determine tolerance based on precision of reference
        if '.' in ref:
            decimal_places = len(ref.split('.')[-1])
            tolerance = 10 ** (-decimal_places - 1)
        else:
            tolerance = 0.5
        
        return abs(pred_val - ref_val) < tolerance
    except (ValueError, TypeError):
        pass
    
    # If model available, use LLM-based comparison
    if model:
        comparison_prompt = f"""You will receive two answers. Your job is to evaluate if they are exactly the same, with some caveats.
If they are wholly different answers (eg: 8 and 9), they are considered different.
If the first answer is a more precise version of the second answer (eg: units listed, more decimal points reported, etc), they are the same.
If the first answer can be rounded to the second answer, with the exact level of precision that the second answer uses, they are considered the same. If they cannot, they are different.
If the answers are numbers and the first number cannot be rounded to the second number, respond with 'different'.
For example, if the first answer is '1.02' and the second answer is '1', they are considered the same,
but if the second answer is '1.02' and the first answer is '1.03' or '1', they are considered different.
If the first answer is '5%' and the second answer is '5', they are considered the same.
If the answers are the same, respond with 'correct'. If they are different, respond with 'wrong'.
First answer: {pred}. Second answer: {ref}"""
        
        try:
            if benchmark and hasattr(benchmark, '_prepare_messages'):
                messages = [{"role": "user", "content": comparison_prompt}]
                formatted_prompt = benchmark._prepare_messages(messages, model)
            else:
                formatted_prompt = comparison_prompt
            
            gen_kwargs = {
                "max_new_tokens": 10,
                "temperature": 0.0,
                "do_sample": False,
            }
            
            if hasattr(model, 'generate_until'):
                result = model.generate_until([[formatted_prompt, gen_kwargs]])[0]
                return 'correct' in result.lower()
        except Exception:
            pass
    
    # Final fallback
    return False