"""
Mock dataset for FINER task testing.
"""

def get_mock_finer_data():
    """Return mock FINER data for testing."""
    return [
        {
            "tokens": ["Apple", "Inc.", "reported", "strong", "earnings", "from", "Cupertino", "operations"],
            "tags": [4, 5, 6, 6, 6, 6, 2, 6]  # B-ORG, I-ORG, O, O, O, O, B-LOC, O
        },
        {
            "tokens": ["John", "Smith", "from", "Goldman", "Sachs", "visited", "New", "York"],
            "tags": [0, 1, 6, 4, 5, 6, 2, 3]  # B-PER, I-PER, O, B-ORG, I-ORG, O, B-LOC, I-LOC
        },
        {
            "tokens": ["Microsoft", "Corporation", "CEO", "announced", "quarterly", "results"],
            "tags": [4, 5, 6, 6, 6, 6]  # B-ORG, I-ORG, O, O, O, O
        },
        {
            "tokens": ["The", "Federal", "Reserve", "in", "Washington", "DC", "made", "announcement"],
            "tags": [6, 4, 5, 6, 2, 3, 6, 6]  # O, B-ORG, I-ORG, O, B-LOC, I-LOC, O, O
        },
        {
            "tokens": ["Jane", "Doe", "analyzed", "Tesla", "stock", "performance"],
            "tags": [0, 1, 6, 4, 6, 6]  # B-PER, I-PER, O, B-ORG, O, O
        },
    ]
