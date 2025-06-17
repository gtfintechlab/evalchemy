"""
Mock dataset for SUBJECTIVEQA testing.
"""

def create_mock_subjectiveqa_dataset():
    """Return mock SubjectiveQA data for testing."""
    
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
            "COMPANYNAME": "TECH",
            "QUARTER": "Q1", 
            "YEAR": 2024,
            "ASKER": "Analyst A",
            "RESPONDER": "CEO",
            "QUESTION": "Can you provide guidance for next quarter's revenue expectations?",
            "ANSWER": "We expect continued strong growth in Q2, driven by our cloud services division. Our pipeline looks robust and we're confident in our market position.",
            "CLEAR": 2,
            "ASSERTIVE": 2, 
            "CAUTIOUS": 0,
            "OPTIMISTIC": 2,
            "SPECIFIC": 1,
            "RELEVANT": 2
        },
        {
            "COMPANYNAME": "BANK",
            "QUARTER": "Q2",
            "YEAR": 2024, 
            "ASKER": "Analyst B",
            "RESPONDER": "CFO",
            "QUESTION": "How will rising interest rates affect your loan portfolio?",
            "ANSWER": "Well, it's difficult to say exactly. There are many factors at play and we're monitoring the situation closely. We'll have to see how things develop.",
            "CLEAR": 0,
            "ASSERTIVE": 0,
            "CAUTIOUS": 2,
            "OPTIMISTIC": 1,
            "SPECIFIC": 0,
            "RELEVANT": 1
        },
        {
            "COMPANYNAME": "RETAIL",
            "QUARTER": "Q3",
            "YEAR": 2024,
            "ASKER": "Analyst C", 
            "RESPONDER": "CEO",
            "QUESTION": "What are your cost reduction initiatives?",
            "ANSWER": "We've implemented several efficiency programs including supply chain optimization and workforce restructuring. These should save us approximately $50M annually.",
            "CLEAR": 2,
            "ASSERTIVE": 1,
            "CAUTIOUS": 1,
            "OPTIMISTIC": 1,
            "SPECIFIC": 2,
            "RELEVANT": 2
        },
        {
            "COMPANYNAME": "ENERGY",
            "QUARTER": "Q4",
            "YEAR": 2024,
            "ASKER": "Analyst D",
            "RESPONDER": "CEO", 
            "QUESTION": "How do you see commodity prices impacting margins?",
            "ANSWER": "Commodity prices remain volatile. We have some hedging in place but it's an ongoing challenge. We're working to mitigate risks where possible.",
            "CLEAR": 1,
            "ASSERTIVE": 1,
            "CAUTIOUS": 2,
            "OPTIMISTIC": 0,
            "SPECIFIC": 1,
            "RELEVANT": 2
        },
        {
            "COMPANYNAME": "PHARMA", 
            "QUARTER": "Q1",
            "YEAR": 2024,
            "ASKER": "Analyst E",
            "RESPONDER": "CTO",
            "QUESTION": "What's the status of your drug pipeline?",
            "ANSWER": "Our pipeline is extremely robust with 15 compounds in Phase 2 trials and 8 in Phase 3. We expect 3 major FDA approvals this year and significant revenue growth.",
            "CLEAR": 2,
            "ASSERTIVE": 2,
            "CAUTIOUS": 0, 
            "OPTIMISTIC": 2,
            "SPECIFIC": 2,
            "RELEVANT": 2
        }
    ]
    
    return MockDataset(mock_data)