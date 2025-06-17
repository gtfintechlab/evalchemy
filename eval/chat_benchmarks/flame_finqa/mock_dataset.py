"""
Mock FinQA dataset for testing when the real dataset is not available.
"""

def get_mock_finqa_data():
    """Return mock FinQA data for testing."""
    return [
        {
            "pre_text": ["Company XYZ reported strong financial results for Q4 2023."],
            "post_text": ["The company's performance exceeded market expectations."],
            "table": [
                ["Metric", "Q3 2023", "Q4 2023"],
                ["Revenue (M)", "100", "120"],
                ["Profit (M)", "10", "15"]
            ],
            "question": "What was the revenue in Q4 2023?",
            "answer": "120"
        },
        {
            "pre_text": ["ABC Corp announced its annual results."],
            "post_text": ["Growth was driven by international expansion."],
            "table": [
                ["Year", "2022", "2023"],
                ["Sales", "500M", "650M"],
                ["Growth", "10%", "30%"]
            ],
            "question": "What was the growth rate in 2023?",
            "answer": "30%"
        },
        {
            "pre_text": ["The quarterly report shows mixed results."],
            "post_text": ["Management remains optimistic about future prospects."],
            "table": [
                ["Quarter", "Q1", "Q2", "Q3", "Q4"],
                ["EPS", "1.2", "1.5", "1.3", "1.8"]
            ],
            "question": "What was the EPS in Q2?",
            "answer": "1.5"
        },
        {
            "pre_text": ["Financial highlights for the fiscal year."],
            "post_text": ["The company maintained strong margins."],
            "table": [
                ["Item", "Amount"],
                ["Total Assets", "2500"],
                ["Total Liabilities", "1500"],
                ["Equity", "1000"]
            ],
            "question": "What is the total equity?",
            "answer": "1000"
        },
        {
            "pre_text": ["Operating metrics showed improvement."],
            "post_text": ["Efficiency gains contributed to profitability."],
            "table": [
                ["Metric", "Previous", "Current"],
                ["Operating Margin", "15%", "18%"],
                ["Net Margin", "8%", "10%"]
            ],
            "question": "What is the current operating margin?",
            "answer": "18%"
        }
    ]