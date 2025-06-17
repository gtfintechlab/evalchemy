"""
Mock dataset for FIQA_TASK2 testing.
"""

def create_mock_fiqa_task2_dataset():
    """Return mock FiQA Task2 data for testing."""
    
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
            "question": "How does a 2 year treasury note work?",
            "answer": "Treasury notes are government bonds that pay interest every six months until maturity. A 2-year note means you lend money to the government for 2 years. When interest rates rise, bond prices fall, and vice versa. Individual investors should consider bond funds for diversification unless holding to maturity."
        },
        {
            "question": "What is the difference between stocks and bonds?",
            "answer": "Stocks represent ownership in a company and offer potential for capital appreciation and dividends. Bonds are loans to companies or governments that pay fixed interest. Stocks are generally riskier but offer higher potential returns, while bonds provide more stable, predictable income."
        },
        {
            "question": "How do I calculate my retirement savings needs?",
            "answer": "A common rule is to save 10-15% of your income and aim for 10-12 times your final working year's salary by retirement. Consider factors like expected retirement age, lifestyle, healthcare costs, and Social Security benefits. Use retirement calculators and consider consulting a financial advisor."
        },
        {
            "question": "What is dollar-cost averaging?",
            "answer": "Dollar-cost averaging involves investing a fixed amount regularly regardless of market conditions. This strategy reduces the impact of market volatility by buying more shares when prices are low and fewer when prices are high, potentially lowering your average cost per share over time."
        },
        {
            "question": "Should I pay off debt or invest?",
            "answer": "Generally, pay off high-interest debt (credit cards, personal loans) first, as the guaranteed savings from eliminating high interest rates often exceed potential investment returns. For low-interest debt like mortgages, investing may be better if expected returns exceed the debt interest rate."
        }
    ]
    
    return MockDataset(mock_data)