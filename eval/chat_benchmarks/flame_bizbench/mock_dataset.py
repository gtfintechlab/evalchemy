"""Mock dataset for BIZBENCH task."""

def create_mock_bizbench_dataset():
    """Create a mock dataset for testing BIZBENCH task."""
    mock_data = []
    
    # Add realistic SEC filing examples
    examples = [
        {
            "question": "What was the net income for the fiscal year?",
            "context": "The Company reported total revenues of $45.2 billion for the fiscal year ended December 31, 2023. After accounting for operating expenses of $38.1 billion and interest expenses of $1.4 billion, the net income for the fiscal year was $5.7 billion.",
            "answer": "5700000000"  # 5.7 billion in raw number format
        },
        {
            "question": "What is the total number of employees?",
            "context": "As of December 31, 2023, we employed approximately 156,000 full-time employees, including 89,000 in our retail operations, 45,000 in our technology division, and 22,000 in corporate and administrative roles.",
            "answer": "156000"
        },
        {
            "question": "What was the revenue growth rate?",
            "context": "Our revenue increased from $41.3 billion in fiscal year 2022 to $45.2 billion in fiscal year 2023, representing a growth rate of 9.4 percent year-over-year.",
            "answer": "9.4"
        },
        {
            "question": "What is the gross profit margin?",
            "context": "For the fiscal year 2023, we achieved gross revenues of $45.2 billion and cost of goods sold of $27.1 billion, resulting in a gross profit of $18.1 billion and a gross profit margin of 40.0 percent.",
            "answer": "40.0"
        },
        {
            "question": "What was the total capital expenditure?",
            "context": "During fiscal year 2023, we invested heavily in infrastructure and technology. Our capital expenditures totaled $3.8 billion, including $2.1 billion for new data centers and $1.7 billion for equipment upgrades.",
            "answer": "3800000000"  # 3.8 billion
        }
    ]
    
    for i, example in enumerate(examples):
        example["id"] = i
        mock_data.append(example)
    
    return mock_data