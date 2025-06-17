"""
Mock dataset for REFIND testing.
"""

def create_mock_refind_dataset():
    """Return mock REFinD data for testing."""
    
    class MockDataset:
        def __init__(self, data):
            self.data = data
        
        def __iter__(self):
            return iter(self.data)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    mock_data = [
        {
            "id": "mock_1",
            "token": ["Apple", "Inc.", "acquired", "the", "startup", "company", "Workflow", "for", "$", "200", "million"],
            "e1": "Apple Inc.",
            "e2": "Workflow", 
            "e1_start": 0,
            "e1_end": 2,
            "e2_start": 6,
            "e2_end": 7,
            "e1_type": "ORG",
            "e2_type": "ORG",
            "relation": "acquired",
            "rel_group": "ORG-ORG"
        },
        {
            "id": "mock_2",
            "token": ["Microsoft", "Corporation", "reported", "strong", "quarterly", "earnings", "today"],
            "e1": "Microsoft Corporation",
            "e2": "earnings",
            "e1_start": 0,
            "e1_end": 2,
            "e2_start": 5,
            "e2_end": 6,
            "e1_type": "ORG", 
            "e2_type": "OTHER",
            "relation": "no_relation",
            "rel_group": "ORG-OTHER"
        },
        {
            "id": "mock_3",
            "token": ["Tesla", "CEO", "Elon", "Musk", "announced", "new", "production", "targets"],
            "e1": "Tesla",
            "e2": "Elon Musk",
            "e1_start": 0,
            "e1_end": 1,
            "e2_start": 2,
            "e2_end": 4,
            "e1_type": "ORG",
            "e2_type": "PER",
            "relation": "CEO_of",
            "rel_group": "ORG-PER"
        },
        {
            "id": "mock_4",
            "token": ["Goldman", "Sachs", "upgraded", "Amazon", "stock", "to", "buy"],
            "e1": "Goldman Sachs",
            "e2": "Amazon",
            "e1_start": 0,
            "e1_end": 2,
            "e2_start": 3,
            "e2_end": 4,
            "e1_type": "ORG",
            "e2_type": "ORG", 
            "relation": "analyst_coverage",
            "rel_group": "ORG-ORG"
        },
        {
            "id": "mock_5",
            "token": ["The", "Federal", "Reserve", "is", "located", "in", "Washington", "DC"],
            "e1": "Federal Reserve",
            "e2": "Washington DC",
            "e1_start": 1,
            "e1_end": 3,
            "e2_start": 6,
            "e2_end": 8,
            "e1_type": "ORG",
            "e2_type": "LOC",
            "relation": "located_in",
            "rel_group": "ORG-LOC"
        }
    ]
    
    return MockDataset(mock_data)