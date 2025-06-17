"""
Mock dataset for CONVFINQA task testing.
"""

def get_mock_convfinqa_data():
    """Return mock CONVFINQA data for testing."""
    return [
        {
            "pre_text": ["Company ABC reported strong financial results for Q3 2023."],
            "post_text": ["The results exceeded analyst expectations."],
            "table_ori": [["Revenue", "2023 Q3", "$100M"], ["Profit", "2023 Q3", "$20M"]],
            "question_0": "What was the revenue in Q3 2023?",
            "answer_0": "$100M",
            "question_1": "What was the profit margin?",
            "answer_1": "20%"
        },
        {
            "pre_text": ["XYZ Corp announced its quarterly earnings."],
            "post_text": ["Sales growth was attributed to new product launches."],
            "table_ori": [["Sales", "Q4 2023", "$250M"], ["Growth", "Q4 2023", "15%"]],
            "question_0": "What were the total sales?",
            "answer_0": "$250M",
            "question_1": "How much did sales grow compared to the previous quarter?",
            "answer_1": "15%"
        },
        {
            "pre_text": ["Financial performance improved significantly."],
            "post_text": ["The company plans to expand operations."],
            "table_ori": [["Net Income", "2023", "$50M"], ["Cash Flow", "2023", "$75M"]],
            "question_0": "What was the net income for 2023?",
            "answer_0": "$50M",
            "question_1": "What is the ratio of cash flow to net income?",
            "answer_1": "1.5"
        },
        {
            "pre_text": ["Market conditions remained challenging."],
            "post_text": ["Despite headwinds, the company maintained profitability."],
            "table_ori": [["Operating Margin", "2023", "12%"], ["Net Margin", "2023", "8%"]],
            "question_0": "What was the operating margin?",
            "answer_0": "12%",
            "question_1": "How much lower is the net margin compared to operating margin?",
            "answer_1": "4%"
        },
        {
            "pre_text": ["Technology investments drove efficiency gains."],
            "post_text": ["Cost reduction initiatives were successful."],
            "table_ori": [["Cost Savings", "2023", "$30M"], ["Investment", "2023", "$20M"]],
            "question_0": "How much was invested in technology?",
            "answer_0": "$20M",
            "question_1": "What was the net benefit from the technology investment?",
            "answer_1": "$10M"
        },
    ]
