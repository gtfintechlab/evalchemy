"""
Mock dataset for FIQA_TASK1 testing.
"""

def create_mock_fiqa_task1_dataset():
    """Return mock FiQA Task1 data for testing."""
    
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
            "sentence": "Apple Inc. reported strong quarterly earnings, beating analyst expectations",
            "snippets": "Apple Inc. reported strong quarterly earnings, beating analyst expectations", 
            "target": "Apple Inc.",
            "sentiment_score": 0.65,
            "aspects": "Earnings/Financial Results"
        },
        {
            "sentence": "Tesla stock declined amid production concerns and supply chain issues",
            "snippets": "Tesla stock declined amid production concerns and supply chain issues",
            "target": "Tesla", 
            "sentiment_score": -0.45,
            "aspects": "Stock/Price Action"
        },
        {
            "sentence": "Microsoft maintained steady performance with consistent cloud revenue growth",
            "snippets": "Microsoft maintained steady performance with consistent cloud revenue growth",
            "target": "Microsoft",
            "sentiment_score": 0.25,
            "aspects": "Revenue/Sales"
        },
        {
            "sentence": "Amazon faced regulatory challenges that could impact future operations",
            "snippets": "Amazon faced regulatory challenges that could impact future operations", 
            "target": "Amazon",
            "sentiment_score": -0.30,
            "aspects": "Regulatory/Legal"
        },
        {
            "sentence": "Google's ad revenue remained stable despite market uncertainties",
            "snippets": "Google's ad revenue remained stable despite market uncertainties",
            "target": "Google",
            "sentiment_score": 0.10,
            "aspects": "Revenue/Sales"
        }
    ]
    
    return MockDataset(mock_data)