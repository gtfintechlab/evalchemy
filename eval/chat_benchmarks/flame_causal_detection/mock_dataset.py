"""
Mock dataset for CAUSAL_DETECTION task testing.
"""

def get_mock_causal_detection_data():
    """Return mock CAUSAL_DETECTION data for testing."""
    return [
        {
            "tokens": ["The", "company", "filed", "bankruptcy", "due", "to", "declining", "sales"],
            "tags": ["B-EFFECT", "I-EFFECT", "I-EFFECT", "I-EFFECT", "O", "O", "B-CAUSE", "I-CAUSE"]
        },
        {
            "tokens": ["Rising", "interest", "rates", "caused", "mortgage", "defaults", "to", "increase"],
            "tags": ["B-CAUSE", "I-CAUSE", "I-CAUSE", "O", "B-EFFECT", "I-EFFECT", "I-EFFECT", "I-EFFECT"]
        },
        {
            "tokens": ["Stock", "prices", "fell", "because", "of", "poor", "earnings", "reports"],
            "tags": ["B-EFFECT", "I-EFFECT", "I-EFFECT", "O", "O", "B-CAUSE", "I-CAUSE", "I-CAUSE"]
        },
        {
            "tokens": ["The", "merger", "resulted", "in", "significant", "cost", "savings"],
            "tags": ["B-CAUSE", "I-CAUSE", "O", "O", "B-EFFECT", "I-EFFECT", "I-EFFECT"]
        },
        {
            "tokens": ["Layoffs", "were", "announced", "following", "the", "acquisition"],
            "tags": ["B-EFFECT", "I-EFFECT", "I-EFFECT", "O", "B-CAUSE", "I-CAUSE"]
        },
    ]
