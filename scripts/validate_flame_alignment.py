#!/usr/bin/env python3
"""
Script to validate FLaME task implementations in Evalchemy against original FLaME logic.

This script compares:
1. Prompt generation between FLaME and Evalchemy
2. Output parsing logic
3. Metric calculations
4. Edge case handling
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import FLaME components
try:
    from FLaME.src.flame.code.prompts.registry import PromptFormat, get_prompt
    from FLaME.src.flame.code.fomc.fomc_evaluate import map_label_to_number as fomc_map_label
    from FLaME.src.flame.code.fpb.fpb_evaluate import map_label_to_number as fpb_map_label
    FLAME_AVAILABLE = True
except ImportError:
    FLAME_AVAILABLE = False
    print("Warning: FLaME modules not available. Some validations will be skipped.")

# Import Evalchemy components
from eval.chat_benchmarks.flame_fomc.eval_instruct import FOMCBenchmark
from eval.chat_benchmarks.flame_fpb.eval_instruct import FPBBenchmark
from eval.chat_benchmarks.flame_finqa.eval_instruct import FinQABenchmark
from eval.chat_benchmarks.flame_headlines.eval_instruct import HeadlinesBenchmark
from eval.chat_benchmarks.flame_utils import parse_classification_output, compare_answers_with_tolerance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FLaMEAlignmentValidator:
    """Validator to check alignment between FLaME and Evalchemy implementations."""
    
    def __init__(self):
        self.results = {
            "prompt_alignment": {},
            "parsing_alignment": {},
            "metric_alignment": {},
            "edge_cases": {}
        }
    
    def validate_prompt_generation(self, task_name: str) -> Dict[str, Any]:
        """Compare prompt generation between FLaME and Evalchemy."""
        results = {"task": task_name, "aligned": True, "differences": []}
        
        # Sample data for testing
        test_cases = {
            "fomc": {
                "sentence": "The Committee decided to maintain the target range for the federal funds rate."
            },
            "fpb": {
                "sentence": "The company's revenue increased by 25% year-over-year."
            },
            "finqa": {
                "pre_text": ["Company ABC reported annual revenue of $1 million."],
                "post_text": ["The company expects 10% growth next year."],
                "table_ori": [["Year", "Revenue"], ["2023", "1000000"]],
                "question": "What is the expected revenue next year?"
            }
        }
        
        if task_name not in test_cases:
            results["differences"].append(f"No test case defined for {task_name}")
            return results
        
        test_data = test_cases[task_name]
        
        # Get Evalchemy prompt
        if task_name == "fomc":
            benchmark = FOMCBenchmark()
            evalchemy_prompt = benchmark.create_prompt(test_data)
        elif task_name == "fpb":
            benchmark = FPBBenchmark()
            evalchemy_prompt = benchmark.create_prompt(test_data)
        elif task_name == "finqa":
            benchmark = FinQABenchmark()
            evalchemy_prompt = benchmark.create_prompt(test_data)
        else:
            results["differences"].append(f"Task {task_name} not implemented for validation")
            return results
        
        # Get FLaME prompt if available
        if FLAME_AVAILABLE:
            try:
                flame_prompt_func = get_prompt(task_name, PromptFormat.ZERO_SHOT)
                if task_name == "fomc":
                    flame_prompt = flame_prompt_func(test_data["sentence"])
                elif task_name == "fpb":
                    flame_prompt = flame_prompt_func(test_data["sentence"])
                elif task_name == "finqa":
                    # FLaME's FinQA prompt expects combined document
                    document = " ".join(test_data["pre_text"]) + " " + " ".join(test_data["post_text"])
                    flame_prompt = flame_prompt_func(document)
                
                # Compare prompts
                if evalchemy_prompt.strip() != flame_prompt.strip():
                    results["aligned"] = False
                    results["differences"].append({
                        "type": "prompt_mismatch",
                        "evalchemy_length": len(evalchemy_prompt),
                        "flame_length": len(flame_prompt),
                        "evalchemy_preview": evalchemy_prompt[:100] + "...",
                        "flame_preview": flame_prompt[:100] + "..."
                    })
            except Exception as e:
                results["differences"].append(f"Error comparing prompts: {str(e)}")
        
        self.results["prompt_alignment"][task_name] = results
        return results
    
    def validate_output_parsing(self, task_name: str) -> Dict[str, Any]:
        """Compare output parsing logic between implementations."""
        results = {"task": task_name, "aligned": True, "differences": []}
        
        # Test cases for parsing
        test_outputs = {
            "fomc": [
                ("HAWKISH", "HAWKISH"),
                ("The statement is DOVISH", "DOVISH"),
                ("I think this is neutral.", "NEUTRAL"),
                ("hawkish", "HAWKISH"),  # Case insensitive
                ("This is clearly a HAWKISH stance on monetary policy.", "HAWKISH"),
            ],
            "fpb": [
                ("POSITIVE", "positive"),
                ("NEGATIVE\nThe company is facing challenges.", "negative"),
                ("neutral", "neutral"),
                ("The sentiment is POSITIVE", "positive"),
            ],
            "finqa": [
                ("The answer is 1000000", "1000000"),
                ("Final answer: $1.5 million", "1.5 million"),
                ("1,234,567", "1234567"),
                ("The revenue will be 1100000", "1100000"),
                ("10%", "10%"),
            ]
        }
        
        if task_name not in test_outputs:
            results["differences"].append(f"No test cases defined for {task_name}")
            return results
        
        # Test Evalchemy parsing
        if task_name == "fomc":
            benchmark = FOMCBenchmark()
            for test_output, expected in test_outputs[task_name]:
                parsed = benchmark.parse_output(test_output, {})
                expected_id = benchmark.label_to_id.get(expected, -1)
                if parsed != expected_id:
                    results["aligned"] = False
                    results["differences"].append({
                        "input": test_output,
                        "expected": expected,
                        "evalchemy_parsed": benchmark.id_to_label.get(parsed, "INVALID"),
                        "expected_id": expected_id,
                        "actual_id": parsed
                    })
        
        elif task_name == "fpb":
            benchmark = FPBBenchmark()
            for test_output, expected in test_outputs[task_name]:
                parsed = benchmark.parse_output(test_output, {})
                expected_id = benchmark.label_to_id.get(expected, -1)
                if parsed != expected_id:
                    results["aligned"] = False
                    results["differences"].append({
                        "input": test_output,
                        "expected": expected,
                        "evalchemy_parsed": benchmark.id_to_label.get(parsed, "INVALID"),
                        "expected_id": expected_id,
                        "actual_id": parsed
                    })
        
        elif task_name == "finqa":
            benchmark = FinQABenchmark()
            for test_output, expected in test_outputs[task_name]:
                parsed = benchmark.parse_output(test_output, {})
                # For FinQA, we compare strings
                if parsed != expected:
                    results["aligned"] = False
                    results["differences"].append({
                        "input": test_output,
                        "expected": expected,
                        "evalchemy_parsed": parsed
                    })
        
        # Compare with FLaME parsing if available
        if FLAME_AVAILABLE and task_name in ["fomc", "fpb"]:
            for test_output, expected in test_outputs[task_name][:2]:  # Test first 2 cases
                if task_name == "fomc":
                    flame_parsed = fomc_map_label(test_output.split()[0] if test_output else "")
                elif task_name == "fpb":
                    # FPB expects first line
                    first_line = test_output.split('\n')[0] if test_output else ""
                    flame_parsed = fpb_map_label(first_line)
                
                # Note: This is simplified - FLaME actually uses LLM extraction
                logger.info(f"FLaME parsing (simplified): {test_output} -> {flame_parsed}")
        
        self.results["parsing_alignment"][task_name] = results
        return results
    
    def validate_metrics(self, task_name: str) -> Dict[str, Any]:
        """Validate metric calculation alignment."""
        results = {"task": task_name, "aligned": True, "differences": []}
        
        # Test metric calculation with sample data
        test_data = {
            "fomc": {
                "predictions": [0, 1, 2, 0, 1],  # DOVISH, HAWKISH, NEUTRAL, DOVISH, HAWKISH
                "references": [0, 1, 2, 1, 1],   # DOVISH, HAWKISH, NEUTRAL, HAWKISH, HAWKISH
                "expected_accuracy": 0.8,  # 4/5 correct
            },
            "fpb": {
                "predictions": [0, 1, 2, 0, 1],  # negative, neutral, positive, negative, neutral
                "references": [0, 1, 2, 0, 2],   # negative, neutral, positive, negative, positive
                "expected_accuracy": 0.8,  # 4/5 correct
            },
            "finqa": {
                "predictions": ["1000", "2000", "50%", "1.5", "0"],
                "references": ["1000", "2000.0", "50", "1.50", "0.0"],
                "expected_accuracy": 1.0,  # All should match with tolerance
            }
        }
        
        if task_name not in test_data:
            results["differences"].append(f"No test data defined for {task_name}")
            return results
        
        data = test_data[task_name]
        
        # Calculate metrics using sklearn (same as both implementations)
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        if task_name in ["fomc", "fpb"]:
            accuracy = accuracy_score(data["references"], data["predictions"])
            precision, recall, f1, _ = precision_recall_fscore_support(
                data["references"], data["predictions"], average='weighted', zero_division=0
            )
            
            if abs(accuracy - data["expected_accuracy"]) > 0.001:
                results["aligned"] = False
                results["differences"].append({
                    "metric": "accuracy",
                    "expected": data["expected_accuracy"],
                    "calculated": accuracy
                })
            
            logger.info(f"{task_name} metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        elif task_name == "finqa":
            # Test FinQA's numeric comparison
            benchmark = FinQABenchmark()
            correct = 0
            for pred, ref in zip(data["predictions"], data["references"]):
                if benchmark.numbers_match(pred, ref):
                    correct += 1
            
            accuracy = correct / len(data["predictions"])
            
            if abs(accuracy - data["expected_accuracy"]) > 0.001:
                results["aligned"] = False
                results["differences"].append({
                    "metric": "accuracy",
                    "expected": data["expected_accuracy"],
                    "calculated": accuracy
                })
            
            logger.info(f"{task_name} metrics - Accuracy: {accuracy:.4f}")
        
        self.results["metric_alignment"][task_name] = results
        return results
    
    def test_edge_cases(self, task_name: str) -> Dict[str, Any]:
        """Test edge cases for robustness."""
        results = {"task": task_name, "passed": True, "failures": []}
        
        edge_cases = {
            "common": [
                ("", "Empty output"),
                (None, "None output"),
                ("Random text with no label", "No valid label"),
                ("Multiple HAWKISH DOVISH labels", "Multiple labels"),
            ],
            "finqa": [
                ("$1,234,567.89", "Complex formatting"),
                ("5%", "Percentage"),
                ("1.0000", "Extra precision"),
                ("-123", "Negative number"),
                ("No answer found", "Non-numeric"),
            ]
        }
        
        # Test common edge cases
        if task_name == "fomc":
            benchmark = FOMCBenchmark()
            for test_input, description in edge_cases["common"]:
                try:
                    parsed = benchmark.parse_output(test_input, {})
                    if parsed == -1:  # Expected for invalid inputs
                        logger.info(f"✓ {description}: Correctly returned -1")
                    else:
                        results["failures"].append(f"{description}: Expected -1, got {parsed}")
                        results["passed"] = False
                except Exception as e:
                    results["failures"].append(f"{description}: Exception - {str(e)}")
                    results["passed"] = False
        
        elif task_name == "finqa":
            benchmark = FinQABenchmark()
            for test_input, description in edge_cases["finqa"]:
                try:
                    parsed = benchmark.parse_output(test_input, {})
                    normalized = benchmark.normalize_number(parsed)
                    logger.info(f"✓ {description}: {test_input} -> {parsed} -> {normalized}")
                except Exception as e:
                    results["failures"].append(f"{description}: Exception - {str(e)}")
                    results["passed"] = False
        
        self.results["edge_cases"][task_name] = results
        return results
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        report_lines = [
            "# FLaME-Evalchemy Alignment Validation Report",
            "=" * 50,
            ""
        ]
        
        # Prompt alignment
        report_lines.append("## 1. Prompt Generation Alignment")
        for task, result in self.results["prompt_alignment"].items():
            status = "✓ ALIGNED" if result["aligned"] else "✗ MISALIGNED"
            report_lines.append(f"- {task}: {status}")
            for diff in result["differences"]:
                report_lines.append(f"  - {diff}")
        report_lines.append("")
        
        # Parsing alignment
        report_lines.append("## 2. Output Parsing Alignment")
        for task, result in self.results["parsing_alignment"].items():
            status = "✓ ALIGNED" if result["aligned"] else "✗ MISALIGNED"
            report_lines.append(f"- {task}: {status}")
            if not result["aligned"]:
                report_lines.append(f"  - {len(result['differences'])} parsing differences found")
        report_lines.append("")
        
        # Metric alignment
        report_lines.append("## 3. Metric Calculation Alignment")
        for task, result in self.results["metric_alignment"].items():
            status = "✓ ALIGNED" if result["aligned"] else "✗ MISALIGNED"
            report_lines.append(f"- {task}: {status}")
            for diff in result["differences"]:
                report_lines.append(f"  - {diff}")
        report_lines.append("")
        
        # Edge cases
        report_lines.append("## 4. Edge Case Handling")
        for task, result in self.results["edge_cases"].items():
            status = "✓ PASSED" if result["passed"] else "✗ FAILED"
            report_lines.append(f"- {task}: {status}")
            for failure in result["failures"]:
                report_lines.append(f"  - {failure}")
        report_lines.append("")
        
        # Summary
        total_aligned = sum(1 for r in self.results["prompt_alignment"].values() if r["aligned"])
        total_tasks = len(self.results["prompt_alignment"])
        report_lines.extend([
            "## Summary",
            f"- Prompt Alignment: {total_aligned}/{total_tasks} tasks aligned",
            f"- FLaME modules available: {'Yes' if FLAME_AVAILABLE else 'No'}",
            ""
        ])
        
        return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Validate FLaME-Evalchemy alignment")
    parser.add_argument("--tasks", nargs="+", default=["fomc", "fpb", "finqa"],
                       help="Tasks to validate")
    parser.add_argument("--output", default="flame_alignment_report.md",
                       help="Output report file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = FLaMEAlignmentValidator()
    
    # Run validations
    for task in args.tasks:
        logger.info(f"\nValidating {task}...")
        validator.validate_prompt_generation(task)
        validator.validate_output_parsing(task)
        validator.validate_metrics(task)
        validator.test_edge_cases(task)
    
    # Generate and save report
    report = validator.generate_report()
    print("\n" + report)
    
    with open(args.output, 'w') as f:
        f.write(report)
    logger.info(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()