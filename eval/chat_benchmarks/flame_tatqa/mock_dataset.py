"""Mock dataset for TATQA task."""

def create_mock_tatqa_dataset():
    """Create a mock dataset for testing TATQA task."""
    mock_data = []
    
    # Add 5 mock QA examples
    questions = [
        "What is the revenue?",
        "What is the profit margin?",
        "How many employees?",
        "What is the growth rate?",
        "What is the market share?"
    ]
    
    answers = ["100", "15%", "5000", "10%", "25%"]
    
    for i, (q, a) in enumerate(zip(questions, answers)):
        example = {
            "id": i,
            "question": q,
            "context": f"Company X reported strong results in Q{i+1}.",
            "answer": a,
        }
        mock_data.append(example)
    
    return mock_data
