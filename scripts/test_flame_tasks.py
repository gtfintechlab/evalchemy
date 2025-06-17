#!/usr/bin/env python3
"""
Comprehensive test runner for FLaME tasks in Evalchemy.

This script:
1. Tests all FLaME tasks with mock data
2. Validates parsing logic
3. Checks metric calculations
4. Tests with different model types
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import all FLaME task benchmarks
from eval.chat_benchmarks.flame_fomc.eval_instruct import FOMCBenchmark
from eval.chat_benchmarks.flame_fpb.eval_instruct import FPBBenchmark
from eval.chat_benchmarks.flame_finqa.eval_instruct import FinQABenchmark
from eval.chat_benchmarks.flame_headlines.eval_instruct import HeadlinesBenchmark
from eval.chat_benchmarks.flame_numclaim.eval_instruct import NumclaimBenchmark
from eval.chat_benchmarks.flame_convfinqa.eval_instruct import ConvfinqaBenchmark
from eval.chat_benchmarks.flame_banking77.eval_instruct import Banking77Benchmark
from eval.chat_benchmarks.flame_central_mock_data import get_mock_data, get_supported_tasks, validate_mock_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Container for test results."""
    task_name: str
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class MockLM:
    """Mock language model for testing."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or self._get_default_responses()
        self.rank = 0
        self.tokenizer = None  # No chat template
        
    def _get_default_responses(self) -> Dict[str, str]:
        """Default responses for different task types."""
        return {
            "fomc": "HAWKISH",
            "fpb": "POSITIVE\nThe financial outlook is strong.",
            "headlines": "UP",
            "numclaim": "INCLAIM",
            "banking77": "balance",
            "finqa": "The final answer is 1100000",
            "convfinqa": "Based on the context, the answer is 280",
        }
    
    def generate_until(self, requests):
        """Mock generation method."""
        responses = []
        for request in requests:
            if isinstance(request, list):
                prompt = request[0] if request else ""
            else:
                prompt = str(request)
            
            # Determine task type from prompt
            task_type = None
            if "HAWKISH" in prompt and "DOVISH" in prompt:
                task_type = "fomc"
            elif "NEGATIVE" in prompt and "POSITIVE" in prompt:
                task_type = "fpb"
            elif "gold price" in prompt.lower():
                task_type = "headlines"
            elif "INCLAIM" in prompt:
                task_type = "numclaim"
            elif "banking" in prompt.lower():
                task_type = "banking77"
            elif "final answer" in prompt.lower() and "financial" in prompt.lower():
                task_type = "finqa" if "conversation" not in prompt.lower() else "convfinqa"
            
            response = self.responses.get(task_type, "Unknown response")
            responses.append(response)
        
        return responses


class FLaMETaskTester:
    """Comprehensive tester for FLaME tasks."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.task_classes = {
            "fomc": FOMCBenchmark,
            "fpb": FPBBenchmark,
            "finqa": FinQABenchmark,
            "headlines": HeadlinesBenchmark,
            "numclaim": NumclaimBenchmark,
            "convfinqa": ConvfinqaBenchmark,
            "banking77": Banking77Benchmark,
        }
    
    def test_mock_data_availability(self) -> None:
        """Test that mock data is available for all tasks."""
        logger.info("Testing mock data availability...")
        
        # Validate mock data structure
        validation_results = validate_mock_data()
        
        for task, is_valid in validation_results.items():
            result = TestResult(
                task_name=task,
                test_name="mock_data_validation",
                passed=is_valid,
                message=f"Mock data {'valid' if is_valid else 'invalid'} for {task}"
            )
            self.results.append(result)
            
        # Test data retrieval
        for task in get_supported_tasks():
            try:
                data = get_mock_data(task, limit=2)
                result = TestResult(
                    task_name=task,
                    test_name="mock_data_retrieval",
                    passed=len(data) > 0,
                    message=f"Retrieved {len(data)} mock samples for {task}",
                    details={"sample_count": len(data)}
                )
            except Exception as e:
                result = TestResult(
                    task_name=task,
                    test_name="mock_data_retrieval",
                    passed=False,
                    message=f"Failed to retrieve mock data: {str(e)}"
                )
            self.results.append(result)
    
    def test_task_initialization(self) -> None:
        """Test that all tasks can be initialized."""
        logger.info("Testing task initialization...")
        
        for task_name, task_class in self.task_classes.items():
            try:
                benchmark = task_class(debug=True)
                result = TestResult(
                    task_name=task_name,
                    test_name="initialization",
                    passed=True,
                    message=f"{task_name} initialized successfully"
                )
            except Exception as e:
                result = TestResult(
                    task_name=task_name,
                    test_name="initialization",
                    passed=False,
                    message=f"Failed to initialize: {str(e)}"
                )
            self.results.append(result)
    
    def test_dataset_loading(self) -> None:
        """Test dataset loading with fallback."""
        logger.info("Testing dataset loading...")
        
        for task_name, task_class in self.task_classes.items():
            try:
                benchmark = task_class(debug=True)
                dataset = benchmark.load_dataset()
                
                # Check if dataset has items
                has_items = len(dataset) > 0
                
                result = TestResult(
                    task_name=task_name,
                    test_name="dataset_loading",
                    passed=has_items,
                    message=f"Loaded {len(dataset)} examples",
                    details={"dataset_size": len(dataset)}
                )
            except Exception as e:
                result = TestResult(
                    task_name=task_name,
                    test_name="dataset_loading",
                    passed=False,
                    message=f"Failed to load dataset: {str(e)}"
                )
            self.results.append(result)
    
    def test_prompt_generation(self) -> None:
        """Test prompt generation for each task."""
        logger.info("Testing prompt generation...")
        
        for task_name in self.task_classes:
            if task_name not in get_supported_tasks():
                continue
                
            try:
                # Get mock data
                mock_data = get_mock_data(task_name, limit=1)[0]
                
                # Initialize benchmark
                benchmark = self.task_classes[task_name](debug=True)
                
                # Generate prompt
                prompt = benchmark.create_prompt(mock_data)
                
                # Validate prompt
                is_valid = (
                    isinstance(prompt, str) and
                    len(prompt) > 50 and  # Reasonable minimum length
                    any(key in prompt for key in ["question", "classify", "answer", "label"])
                )
                
                result = TestResult(
                    task_name=task_name,
                    test_name="prompt_generation",
                    passed=is_valid,
                    message=f"Generated prompt of length {len(prompt)}",
                    details={"prompt_length": len(prompt), "prompt_preview": prompt[:100] + "..."}
                )
            except Exception as e:
                result = TestResult(
                    task_name=task_name,
                    test_name="prompt_generation",
                    passed=False,
                    message=f"Failed to generate prompt: {str(e)}"
                )
            self.results.append(result)
    
    def test_response_generation(self) -> None:
        """Test response generation with mock model."""
        logger.info("Testing response generation...")
        
        mock_model = MockLM()
        
        for task_name, task_class in self.task_classes.items():
            try:
                benchmark = task_class(debug=True)
                results = benchmark.generate_responses(mock_model)
                
                # Check results structure
                is_valid = (
                    results is not None and
                    "outputs" in results and
                    "dataset" in results and
                    len(results["outputs"]) > 0
                )
                
                result = TestResult(
                    task_name=task_name,
                    test_name="response_generation",
                    passed=is_valid,
                    message=f"Generated {len(results['outputs']) if results else 0} responses",
                    details={"response_count": len(results["outputs"]) if results else 0}
                )
            except Exception as e:
                result = TestResult(
                    task_name=task_name,
                    test_name="response_generation",
                    passed=False,
                    message=f"Failed to generate responses: {str(e)}"
                )
            self.results.append(result)
    
    def test_output_parsing(self) -> None:
        """Test output parsing for different response formats."""
        logger.info("Testing output parsing...")
        
        test_cases = {
            "fomc": [
                ("HAWKISH", 1),
                ("The stance is DOVISH", 0),
                ("neutral", 2),
                ("Invalid response", -1),
            ],
            "fpb": [
                ("POSITIVE\nGood outlook", 2),
                ("NEGATIVE", 0),
                ("The sentiment is neutral", 1),
                ("No clear label", -1),
            ],
            "finqa": [
                ("The answer is 1000", "1000"),
                ("Final answer: $1,500", "1500"),
                ("Revenue: 2.5 million", "2.5 million"),
            ],
        }
        
        for task_name, cases in test_cases.items():
            if task_name not in self.task_classes:
                continue
                
            benchmark = self.task_classes[task_name](debug=True)
            
            for test_input, expected in cases:
                try:
                    if hasattr(benchmark, 'parse_output'):
                        parsed = benchmark.parse_output(test_input, {})
                        
                        # Check if parsing is correct
                        if task_name in ["fomc", "fpb"]:
                            passed = parsed == expected
                        else:  # finqa
                            passed = parsed == expected or parsed in expected
                        
                        result = TestResult(
                            task_name=task_name,
                            test_name="output_parsing",
                            passed=passed,
                            message=f"Parse '{test_input[:30]}...' -> {parsed} (expected {expected})",
                            details={"input": test_input, "parsed": parsed, "expected": expected}
                        )
                        self.results.append(result)
                except Exception as e:
                    result = TestResult(
                        task_name=task_name,
                        test_name="output_parsing",
                        passed=False,
                        message=f"Parsing failed for '{test_input[:30]}...': {str(e)}"
                    )
                    self.results.append(result)
    
    def test_evaluation_metrics(self) -> None:
        """Test metric calculation."""
        logger.info("Testing evaluation metrics...")
        
        mock_model = MockLM()
        
        for task_name, task_class in self.task_classes.items():
            try:
                benchmark = task_class(debug=True)
                
                # Generate responses
                gen_results = benchmark.generate_responses(mock_model)
                
                if gen_results is None:
                    continue
                
                # Evaluate responses
                metrics = benchmark.evaluate_responses(gen_results)
                
                # Check metrics structure
                is_valid = (
                    metrics is not None and
                    isinstance(metrics, dict) and
                    any(key in metrics for key in ["accuracy", "f1", "correct", "total"])
                )
                
                result = TestResult(
                    task_name=task_name,
                    test_name="evaluation_metrics",
                    passed=is_valid,
                    message=f"Computed metrics: {list(metrics.keys()) if metrics else []}",
                    details=metrics
                )
            except Exception as e:
                result = TestResult(
                    task_name=task_name,
                    test_name="evaluation_metrics",
                    passed=False,
                    message=f"Failed to compute metrics: {str(e)}"
                )
            self.results.append(result)
    
    def generate_report(self) -> str:
        """Generate test report."""
        report_lines = [
            "# FLaME Tasks Test Report",
            "=" * 50,
            ""
        ]
        
        # Group results by test type
        test_types = {}
        for result in self.results:
            if result.test_name not in test_types:
                test_types[result.test_name] = []
            test_types[result.test_name].append(result)
        
        # Report by test type
        for test_name, results in test_types.items():
            report_lines.append(f"## {test_name.replace('_', ' ').title()}")
            
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            report_lines.append(f"Passed: {passed}/{total}")
            report_lines.append("")
            
            for result in results:
                status = "✓" if result.passed else "✗"
                report_lines.append(f"- {status} {result.task_name}: {result.message}")
                if result.details and not result.passed:
                    report_lines.append(f"  Details: {json.dumps(result.details, indent=2)}")
            report_lines.append("")
        
        # Summary
        total_passed = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report_lines.extend([
            "## Summary",
            f"- Total Tests: {total_tests}",
            f"- Passed: {total_passed}",
            f"- Failed: {total_tests - total_passed}",
            f"- Pass Rate: {pass_rate:.1f}%",
            ""
        ])
        
        return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Test FLaME tasks in Evalchemy")
    parser.add_argument("--tasks", nargs="+", help="Specific tasks to test")
    parser.add_argument("--output", default="flame_test_report.md", help="Output report file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    tester = FLaMETaskTester()
    
    # If specific tasks requested, filter
    if args.tasks:
        tester.task_classes = {k: v for k, v in tester.task_classes.items() if k in args.tasks}
    
    # Run all tests
    logger.info("Starting FLaME task tests...")
    tester.test_mock_data_availability()
    tester.test_task_initialization()
    tester.test_dataset_loading()
    tester.test_prompt_generation()
    tester.test_response_generation()
    tester.test_output_parsing()
    tester.test_evaluation_metrics()
    
    # Generate report
    report = tester.generate_report()
    print("\n" + report)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()