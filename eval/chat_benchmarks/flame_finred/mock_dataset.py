"""Mock dataset for FINRED task."""

def create_mock_finred_dataset():
    """Create a mock dataset for testing FINRED task."""
    mock_data = []
    
    # Add realistic financial relationship examples
    examples = [
        {
            "sentence": "Apple Inc. announced a strategic partnership with Goldman Sachs to launch the Apple Card credit card service.",
            "entities": [["Goldman Sachs", "Apple Inc."], ["Apple Card", "Apple Inc."]],
            "relations": ["Collaborated-with", "Owner-of"]
        },
        {
            "sentence": "The SEC launched an investigation into Tesla's CEO Elon Musk's tweets about taking the company private.",
            "entities": [["Tesla", "SEC"], ["Elon Musk", "Tesla"]],
            "relations": ["Investigated-by", "Operator-of"]
        },
        {
            "sentence": "Amazon Web Services competes directly with Microsoft Azure and Google Cloud Platform in the cloud computing market.",
            "entities": [["Microsoft Azure", "Amazon Web Services"], ["Google Cloud Platform", "Amazon Web Services"]],
            "relations": ["Compete-with", "Compete-with"]
        },
        {
            "sentence": "JPMorgan Chase acquired Bear Stearns during the 2008 financial crisis for $10 per share.",
            "entities": [["Bear Stearns", "JPMorgan Chase"]],
            "relations": ["Was-merged-into"]
        },
        {
            "sentence": "Wells Fargo faced criticism and regulatory penalties for creating millions of unauthorized customer accounts.",
            "entities": [["Wells Fargo", "regulatory penalties"], ["customer accounts", "Wells Fargo"]],
            "relations": ["Negative-impression", "Unfair-practice"]
        }
    ]
    
    for i, example in enumerate(examples):
        example["id"] = i
        mock_data.append(example)
    
    return mock_data