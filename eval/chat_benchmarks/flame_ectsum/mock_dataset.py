"""
Mock dataset for ECTSUM task testing.
"""

def get_mock_ectsum_data():
    """Return mock ECTSUM data for testing."""
    return [
        {
            "context": "Good morning everyone. Thank you for joining our Q3 earnings call. I'm pleased to report that we exceeded expectations across all key metrics. Revenue grew 15% year-over-year to $2.5 billion, driven by strong performance in our cloud services division. Operating margin improved to 18.5%, up from 16.2% last quarter. We also announced a $500 million share buyback program and increased our dividend by 8%. Looking ahead to Q4, we expect continued growth with revenue guidance of $2.7-2.8 billion.",
            "response": "• Q3 revenue $2.5B, up 15% YoY\n• Operating margin improved to 18.5%\n• $500M share buyback announced\n• Dividend increased 8%\n• Q4 revenue guidance $2.7-2.8B"
        },
        {
            "context": "Welcome to our quarterly earnings discussion. This quarter presented challenges due to supply chain disruptions and increased raw material costs. Despite these headwinds, we maintained profitability with revenue of $1.8 billion, down 3% from last year. We implemented cost reduction measures saving $50 million annually. Our new product line launched successfully, contributing $100 million in revenue. We're investing $200 million in automation to improve efficiency.",
            "response": "• Revenue $1.8B, down 3% YoY due to supply chain issues\n• Cost reduction measures saved $50M annually\n• New product line generated $100M revenue\n• $200M automation investment planned"
        },
        {
            "context": "Thank you for participating in today's call. I'm excited to share our outstanding Q2 results. Revenue reached a record $3.2 billion, representing 22% growth compared to the same period last year. Our international expansion contributed significantly, with overseas revenue up 35%. We completed two strategic acquisitions totaling $800 million. R&D spending increased to $150 million as we invest in next-generation technologies.",
            "response": "• Record Q2 revenue $3.2B, up 22% YoY\n• International revenue surged 35%\n• Two acquisitions completed for $800M total\n• R&D spending increased to $150M"
        },
        {
            "context": "Good afternoon. Our Q1 performance reflects the challenging market conditions we've been navigating. Revenue declined 8% to $1.4 billion due to reduced consumer spending. However, we maintained strong cash flow of $400 million and reduced debt by $200 million. We're restructuring operations to improve efficiency, which will result in 500 job cuts but save $75 million annually. New partnerships with three major retailers will expand our distribution.",
            "response": "• Q1 revenue $1.4B, down 8% amid market challenges\n• Strong cash flow of $400M maintained\n• Debt reduced by $200M\n• Restructuring to save $75M annually\n• New retail partnerships for expanded distribution"
        },
        {
            "context": "Welcome everyone to our fiscal year-end earnings call. We're pleased to report a strong finish to the year with Q4 revenue of $2.9 billion, bringing full-year revenue to $10.8 billion, up 12% from the previous year. Net income reached $1.2 billion with earnings per share of $4.85. We returned $600 million to shareholders through dividends and buybacks. Our balance sheet remains robust with $2.5 billion in cash and minimal debt.",
            "response": "• Q4 revenue $2.9B; full-year $10.8B, up 12%\n• Net income $1.2B; EPS $4.85\n• $600M returned to shareholders\n• Strong balance sheet with $2.5B cash"
        },
    ]
