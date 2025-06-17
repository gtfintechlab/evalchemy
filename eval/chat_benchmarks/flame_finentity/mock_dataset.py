"""
Mock dataset for FINENTITY task testing.
"""

def get_mock_finentity_data():
    """Return mock FINENTITY data for testing."""
    return [
        {
            "content": "Apple Inc. reported strong quarterly earnings, while Tesla stock declined due to production concerns.",
            "annotations": [
                {"value": "Apple", "start": 0, "end": 5, "label": "Positive"},
                {"value": "Tesla", "start": 51, "end": 56, "label": "Negative"}
            ]
        },
        {
            "content": "Microsoft Corporation announced new cloud services, boosting investor confidence.",
            "annotations": [
                {"value": "Microsoft", "start": 0, "end": 9, "label": "Positive"}
            ]
        },
        {
            "content": "The Federal Reserve maintained interest rates, providing market stability.",
            "annotations": [
                {"value": "Federal Reserve", "start": 4, "end": 19, "label": "Neutral"}
            ]
        },
        {
            "content": "Goldman Sachs upgraded Amazon while downgrading Meta due to regulatory concerns.",
            "annotations": [
                {"value": "Goldman Sachs", "start": 0, "end": 13, "label": "Neutral"},
                {"value": "Amazon", "start": 23, "end": 29, "label": "Positive"},
                {"value": "Meta", "start": 49, "end": 53, "label": "Negative"}
            ]
        },
        {
            "content": "JP Morgan Chase reported mixed results in the financial sector analysis.",
            "annotations": [
                {"value": "JP Morgan Chase", "start": 0, "end": 15, "label": "Neutral"}
            ]
        },
    ]
