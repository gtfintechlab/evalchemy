"""Mock dataset for FNXL task."""

def create_mock_fnxl_dataset():
    """Create a mock dataset for testing FNXL task."""
    
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
            "sentence": "Total revenue for the year was $145.2 million, up from $122.8 million the previous year.",
            "company": "TechCorp",
            "docType": "10-K",
            "numerals-tags": '{"145.2": "us-gaap:Revenue", "122.8": "us-gaap:Revenue"}'
        },
        {
            "sentence": "Operating expenses increased to $89.5 million compared to $76.3 million in 2023.",
            "company": "RetailCo",
            "docType": "10-Q",
            "numerals-tags": '{"89.5": "us-gaap:OperatingExpenses", "76.3": "us-gaap:OperatingExpenses", "2023": "other"}'
        },
        {
            "sentence": "Net income was $12.7 million or $0.45 per share for the quarter.",
            "company": "FinanceInc",
            "docType": "8-K",
            "numerals-tags": '{"12.7": "us-gaap:NetIncomeLoss", "0.45": "us-gaap:EarningsPerShareBasic"}'
        },
        {
            "sentence": "Cash and cash equivalents totaled $234.1 million at December 31, 2024.",
            "company": "BankHolding",
            "docType": "10-K",
            "numerals-tags": '{"234.1": "us-gaap:CashAndCashEquivalentsAtCarryingValue", "31": "other", "2024": "other"}'
        },
        {
            "sentence": "Research and development costs were $56.8 million, representing 15.2% of total revenue.",
            "company": "BioTech",
            "docType": "10-Q",
            "numerals-tags": '{"56.8": "us-gaap:ResearchAndDevelopmentExpense", "15.2": "other"}'
        }
    ]
    
    return MockDataset(mock_data)
