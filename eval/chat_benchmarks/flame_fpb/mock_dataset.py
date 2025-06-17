"""
Mock Financial Phrase Bank dataset for testing when the real dataset is not available.
"""

def get_mock_fpb_data():
    """Return mock FPB data for testing."""
    return [
        {
            "sentence": "The company's revenue increased by 25% year-over-year, exceeding analyst expectations.",
            "label": 2  # POSITIVE
        },
        {
            "sentence": "Profits declined sharply due to increased competition and rising costs.",
            "label": 0  # NEGATIVE
        },
        {
            "sentence": "The board of directors announced a regular quarterly dividend.",
            "label": 1  # NEUTRAL
        },
        {
            "sentence": "Strong demand for the company's new product line boosted quarterly earnings.",
            "label": 2  # POSITIVE
        },
        {
            "sentence": "The company filed for bankruptcy protection after failing to meet debt obligations.",
            "label": 0  # NEGATIVE
        },
        {
            "sentence": "Management provided guidance for the upcoming fiscal year.",
            "label": 1  # NEUTRAL
        },
        {
            "sentence": "Market share grew significantly as competitors struggled with supply chain issues.",
            "label": 2  # POSITIVE
        },
        {
            "sentence": "The CEO resigned amid allegations of financial misconduct.",
            "label": 0  # NEGATIVE
        },
        {
            "sentence": "The company announced plans to open new facilities in three countries.",
            "label": 1  # NEUTRAL
        },
        {
            "sentence": "Record-breaking sales drove the stock price to an all-time high.",
            "label": 2  # POSITIVE
        }
    ]