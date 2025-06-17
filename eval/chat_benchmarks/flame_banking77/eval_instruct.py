"""
BANKING77 (Banking intent classification) task for Evalchemy.

This module implements the BANKING77 task from FLaME within the Evalchemy framework.
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


# Complete list of 77 Banking categories from FLaME
BANKING77_CATEGORIES = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "refund_not_showing_up", "request_refund", "reverted_card_payment",
    "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity",
    "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"
]


class Banking77Benchmark(BaseBenchmark):
    """BANKING77 benchmark for Evalchemy."""
    
    def __init__(
        self,
        max_new_tokens: int = 150,
        temperature: float = 0.0,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize BANKING77 benchmark."""
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug = debug
        self.dataset_name = 'gtfintechlab/banking77'
        self.split = 'test'
        
        # Banking77 label mappings (77 categories)
        self.categories = BANKING77_CATEGORIES
        self.label_to_id = {category: idx for idx, category in enumerate(self.categories)}
        self.id_to_label = {idx: category for idx, category in enumerate(self.categories)}
        self.categories_str = ", ".join(self.categories)
        
    def load_dataset(self):
        """Load BANKING77 dataset with fallback to mock data."""
        # Use centralized loading with fallback
        return load_dataset_with_fallback(
            dataset_name=self.dataset_name,
            split=self.split,
            task_name='banking77',
            debug=self.debug,
            logger=self.logger
        )
    
    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create BANKING77 classification prompt."""
        text = example.get('text', example.get('sentence', ''))
        
        prompt = f"""Discard all the previous instructions. Behave like you are an expert at fine-grained single-domain intent detection. From the following list: {self.categories_str}, identify which category the following sentence belongs to.

{text}

Provide only the category name from the list above, exactly as it appears in the list."""
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
            instance.task_name = "flame_banking77"
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
            return -1
        
        # First try standardized parser
        parsed_label = parse_classification_output(output, self.categories, case_sensitive=False)
        if parsed_label:
            return self.label_to_id.get(parsed_label.lower(), -1)
        
        # If standardized parser fails, use custom Banking77 logic
        # Clean the output
        output_clean = output.strip().lower()
        
        # Try exact match first (most reliable)
        if output_clean in self.label_to_id:
            return self.label_to_id[output_clean]
        
        # Try finding category name in the output
        for category in self.categories:
            if category in output_clean:
                return self.label_to_id[category]
        
        # Try finding category name with spaces instead of underscores
        for category in self.categories:
            category_spaced = category.replace('_', ' ')
            if category_spaced in output_clean:
                return self.label_to_id[category]
        
        # Try partial matching on key words
        output_words = output_clean.split()
        for category in self.categories:
            category_words = category.split('_')
            # Check if all category words appear in output
            if all(word in output_words for word in category_words):
                return self.label_to_id[category]
        
        return -1
    
    def extract_reference(self, example: Dict[str, Any]) -> int:
        """Extract ground truth label from example."""
        label = example.get('label')
        
        # Handle numeric labels (0-76)
        if isinstance(label, (int, float)):
            label_int = int(label)
            if 0 <= label_int < len(self.categories):
                return label_int
        
        # Handle string labels (category names)
        if isinstance(label, str):
            return self.label_to_id.get(label.lower(), -1)
        
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
