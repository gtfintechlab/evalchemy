"""
Mock FOMC dataset for testing when the real dataset is not available.
"""

def get_mock_fomc_data():
    """Return mock FOMC data for testing."""
    return [
        {
            "sentence": "The Committee decided to raise the federal funds rate by 25 basis points.",
            "label": "HAWKISH"
        },
        {
            "sentence": "Members agreed that the current accommodative stance of monetary policy remained appropriate.",
            "label": "DOVISH"
        },
        {
            "sentence": "The Committee will continue to monitor economic conditions closely.",
            "label": "NEUTRAL"
        },
        {
            "sentence": "Several participants favored a more restrictive policy stance to combat inflation.",
            "label": "HAWKISH"
        },
        {
            "sentence": "The Committee judged that further easing may be necessary to support the recovery.",
            "label": "DOVISH"
        },
        {
            "sentence": "Participants noted that economic data had been mixed in recent weeks.",
            "label": "NEUTRAL"
        },
        {
            "sentence": "The Committee emphasized the need for tighter financial conditions.",
            "label": "HAWKISH"
        },
        {
            "sentence": "Members agreed to maintain an accommodative policy for an extended period.",
            "label": "DOVISH"
        },
        {
            "sentence": "The economic outlook remained uncertain according to most participants.",
            "label": "NEUTRAL"
        },
        {
            "sentence": "The Committee voted unanimously to increase interest rates.",
            "label": "HAWKISH"
        }
    ]