#!/usr/bin/env python3
"""
Script to apply TODO improvements across all FLaME tasks.

This script systematically applies the standardized improvements from our TODO plan:
1. Standardized chat template handling
2. Centralized mock data fallback
3. Consistent debug mode and data limiting
4. Standardized output parsing for classification tasks

Usage:
    python apply_flame_improvements.py --task flame_headlines --dry-run
    python apply_flame_improvements.py --all --apply
"""

import argparse
import glob
import os
import re
from pathlib import Path


FLAME_TASKS = [
    'flame_fomc', 'flame_fpb', 'flame_finqa', 'flame_headlines', 
    'flame_numclaim', 'flame_convfinqa', 'flame_banking77', 
    'flame_causal_classification'
]


def get_flame_task_paths():
    """Get all FLaME task directories."""
    base_dir = Path(__file__).parent
    task_paths = []
    
    for task_name in FLAME_TASKS:
        task_dir = base_dir / task_name
        eval_file = task_dir / "eval_instruct.py"
        if eval_file.exists():
            task_paths.append((task_name, task_dir, eval_file))
        else:
            print(f"Warning: {task_name} not found at {task_dir}")
    
    return task_paths


def analyze_task_file(task_file_path):
    """Analyze a task file to determine what improvements are needed."""
    with open(task_file_path, 'r') as f:
        content = f.read()
    
    improvements_needed = {
        'add_flame_utils_import': 'from eval.chat_benchmarks.flame_utils' not in content,
        'update_chat_template_handling': 'has_chat_template =' in content or '_prepare_messages(messages' in content,
        'centralize_mock_data': 'from .mock_dataset import' in content and 'from ..flame_central_mock_data import get_mock_data' not in content,
        'standardize_dataset_loading': 'datasets.load_dataset(' in content and 'load_dataset_with_fallback' not in content,
        'improve_output_parsing': 'def parse_output(' in content and 'parse_classification_output' not in content,
    }
    
    return improvements_needed


def apply_import_improvements(content):
    """Add flame_utils imports if not present."""
    if 'from eval.chat_benchmarks.flame_utils' in content:
        return content
    
    # Find existing imports
    import_pattern = r'(from eval\.task import BaseBenchmark\nfrom lm_eval\.api\.instance import Instance\nfrom lm_eval\.api\.model import LM\nimport datasets)'
    
    replacement = '''from eval.task import BaseBenchmark
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import datasets
from eval.chat_benchmarks.flame_utils import (
    format_flame_prompt_consistently, 
    create_standard_flame_messages,
    load_dataset_with_fallback,
    parse_classification_output
)'''
    
    updated_content = re.sub(import_pattern, replacement, content)
    return updated_content


def apply_chat_template_improvements(content):
    """Update chat template handling to use standardized functions."""
    # Replace old chat template check pattern
    old_pattern = r'''# Format prompt.*?
.*?has_chat_template = hasattr\(model.*?\n.*?
.*?if has_chat_template:.*?\n.*?
.*?messages = \[.*?\].*?\n.*?
.*?formatted_prompt = self\._prepare_messages\(messages, model\).*?\n.*?
.*?else:.*?\n.*?
.*?formatted_prompt = prompt'''
    
    new_pattern = '''# Format prompt using standardized FLaME approach
            messages = create_standard_flame_messages(prompt)
            formatted_prompt = format_flame_prompt_consistently(messages, model, self)'''
    
    # Use MULTILINE and DOTALL flags for proper matching
    updated_content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE | re.DOTALL)
    return updated_content


def apply_mock_data_improvements(content, task_name):
    """Update mock data to use centralized system."""
    if 'flame_central_mock_data' in content:
        return content
    
    # Extract task name without flame_ prefix
    simple_task_name = task_name.replace('flame_', '')
    
    # Replace mock data imports and usage
    old_mock_pattern = r'from \.mock_dataset import get_mock_.*?_data.*?\n.*?mock_data = get_mock_.*?_data\(\)'
    new_mock_pattern = f'''from ..flame_central_mock_data import get_mock_data
            mock_data = get_mock_data('{simple_task_name}')'''
    
    updated_content = re.sub(old_mock_pattern, new_mock_pattern, content, flags=re.MULTILINE)
    return updated_content


def apply_dataset_loading_improvements(content):
    """Replace manual dataset loading with standardized function."""
    # This is more complex and would require parsing the specific dataset loading logic
    # For now, we'll provide a template
    print("  - Dataset loading standardization requires manual review")
    return content


def apply_output_parsing_improvements(content):
    """Update output parsing to use standardized function for classification tasks."""
    # This is task-specific and requires understanding the label structure
    print("  - Output parsing standardization requires manual review")
    return content


def apply_improvements_to_task(task_name, task_dir, eval_file, dry_run=True):
    """Apply all relevant improvements to a single task."""
    print(f"\nAnalyzing {task_name}...")
    
    improvements = analyze_task_file(eval_file)
    if not any(improvements.values()):
        print(f"  ✓ No improvements needed for {task_name}")
        return
    
    print(f"  Improvements needed:")
    for improvement, needed in improvements.items():
        if needed:
            print(f"    - {improvement}")
    
    if dry_run:
        print(f"  [DRY RUN] Would apply improvements to {eval_file}")
        return
    
    # Read current content
    with open(eval_file, 'r') as f:
        content = f.read()
    
    # Apply improvements
    if improvements['add_flame_utils_import']:
        content = apply_import_improvements(content)
        print("  ✓ Added flame_utils imports")
    
    if improvements['update_chat_template_handling']:
        content = apply_chat_template_improvements(content)
        print("  ✓ Updated chat template handling")
    
    if improvements['centralize_mock_data']:
        content = apply_mock_data_improvements(content, task_name)
        print("  ✓ Centralized mock data")
    
    # Write back
    with open(eval_file, 'w') as f:
        f.write(content)
    
    print(f"  ✓ Applied improvements to {task_name}")


def validate_improvements():
    """Validate that improvements have been applied correctly."""
    print("\nValidating improvements...")
    
    task_paths = get_flame_task_paths()
    validation_results = {}
    
    for task_name, task_dir, eval_file in task_paths:
        improvements = analyze_task_file(eval_file)
        issues = [k for k, v in improvements.items() if v]
        validation_results[task_name] = issues
        
        if not issues:
            print(f"  ✓ {task_name}: All improvements applied")
        else:
            print(f"  ⚠ {task_name}: Still needs {len(issues)} improvements")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description='Apply FLaME TODO improvements')
    parser.add_argument('--task', help='Specific task to improve (e.g., flame_headlines)')
    parser.add_argument('--all', action='store_true', help='Apply to all FLaME tasks')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Show what would be done without applying')
    parser.add_argument('--apply', action='store_true', help='Actually apply the improvements')
    parser.add_argument('--validate', action='store_true', help='Validate current state of improvements')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_improvements()
        return
    
    # Determine dry run mode
    dry_run = not args.apply
    
    if args.task:
        # Apply to single task
        task_paths = [(name, dir_path, file_path) for name, dir_path, file_path in get_flame_task_paths() 
                     if name == args.task]
        if not task_paths:
            print(f"Error: Task {args.task} not found")
            return
    elif args.all:
        # Apply to all tasks
        task_paths = get_flame_task_paths()
    else:
        print("Error: Must specify --task, --all, or --validate")
        return
    
    print(f"FLaME TODO Improvements Script")
    print(f"{'=' * 40}")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLY CHANGES'}")
    print(f"Tasks: {len(task_paths)}")
    
    for task_name, task_dir, eval_file in task_paths:
        apply_improvements_to_task(task_name, task_dir, eval_file, dry_run)
    
    if not dry_run:
        print(f"\n✓ Improvements applied to {len(task_paths)} tasks")
        print(f"Run with --validate to check results")
    else:
        print(f"\nTo apply these changes, run with --apply instead of --dry-run")


if __name__ == "__main__":
    main()