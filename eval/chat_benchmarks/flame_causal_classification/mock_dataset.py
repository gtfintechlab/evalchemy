"""
Mock dataset for CAUSAL_CLASSIFICATION task testing.
"""

def get_mock_causal_classification_data():
    """Return mock CAUSAL_CLASSIFICATION data for testing."""
    return [
        {
            "text": "The stock price rose 70.0 ores or 0.9% to close at SEK77.65, ending a two-day streak of losses.",
            "label": 0  # Non-causal (just stating facts)
        },
        {
            "text": "Due to strong quarterly earnings, the stock price increased by 15%.",
            "label": 1  # Direct causal (clear cause-effect)
        },
        {
            "text": "An Android app will be coming soon.",
            "label": 2  # Indirect causal (future planning/intentions)
        },
        {
            "text": "The company reported net income of $2.3 million for Q3 2023.",
            "label": 0  # Non-causal (factual reporting)
        },
        {
            "text": "Higher raw material costs led to reduced profit margins this quarter.",
            "label": 1  # Direct causal (explicit cause-effect relationship)
        },
    ]
