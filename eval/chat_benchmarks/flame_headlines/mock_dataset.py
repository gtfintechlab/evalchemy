"""
Mock dataset for HEADLINES task testing.
"""

def get_mock_headlines_data():
    """Return mock HEADLINES data for testing."""
    return [
        {
            "headline": "Apple Reports Record-Breaking Q4 Earnings, Stock Soars 8%",
            "label": "BULLISH"
        },
        {
            "headline": "Federal Reserve Announces Unexpected Rate Cut to Boost Economy", 
            "label": "BULLISH"
        },
        {
            "headline": "Major Bank Files for Bankruptcy Protection Amid Credit Crisis",
            "label": "BEARISH"
        },
        {
            "headline": "Oil Prices Plunge 15% on Oversupply Concerns",
            "label": "BEARISH"
        },
        {
            "headline": "S&P 500 Closes Flat as Investors Await Fed Meeting",
            "label": "NEUTRAL"
        },
        {
            "headline": "Tesla Recalls 500,000 Vehicles Over Safety Concerns",
            "label": "BEARISH"
        },
        {
            "headline": "Amazon Announces Stock Split and $10 Billion Buyback Program",
            "label": "BULLISH"
        },
        {
            "headline": "Quarterly GDP Growth Meets Expectations at 2.1%",
            "label": "NEUTRAL"
        },
        {
            "headline": "Tech Giants Lead Market Rally with 5% Gains",
            "label": "BULLISH"
        },
        {
            "headline": "Trading Volume Remains Steady Ahead of Holiday Weekend",
            "label": "NEUTRAL"
        },
    ]