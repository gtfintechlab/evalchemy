"""
Mock dataset for EDTSUM task testing.
"""

def get_mock_edtsum_data():
    """Return mock EDTSUM data for testing."""
    return [
        {
            "id": "edtsum_mock_1",
            "query": "Summarize this financial document focusing on key metrics and performance indicators.",
            "text": "TechCorp Inc. announced its Q3 2024 financial results today, reporting record revenue of $4.2 billion, representing a 28% increase compared to the same quarter last year. The company's net income reached $850 million, with earnings per share of $3.15. Operating margin improved significantly to 22.1%, up from 19.8% in Q3 2023.",
            "answer": "TechCorp Q3 2024: Record $4.2B revenue (+28% YoY), $850M net income, EPS $3.15. Operating margin improved to 22.1%."
        },
        {
            "id": "edtsum_mock_2", 
            "query": "Create a concise summary highlighting the main financial developments.",
            "text": "Global Manufacturing Corp filed its annual 10-K report, detailing a challenging fiscal year 2024. Total revenue declined 12% to $8.7 billion due to supply chain disruptions. Despite revenue challenges, the company maintained positive cash flow of $950 million and reduced total debt by $400 million.",
            "answer": "Global Manufacturing 2024: Revenue down 12% to $8.7B due to supply challenges. Maintained $950M cash flow, reduced debt $400M."
        },
        {
            "id": "edtsum_mock_3",
            "query": "Summarize the key points of this earnings report.",
            "text": "BioPharm Solutions released its Q2 2024 earnings results, showcasing strong performance. Revenue increased 18% to $2.1 billion, driven by robust sales of its flagship cancer treatment drug. The company received FDA approval for two new medications.",
            "answer": "BioPharm Q2 2024: Revenue up 18% to $2.1B led by cancer drug sales. FDA approved 2 new drugs."
        },
        {
            "id": "edtsum_mock_4",
            "query": "Provide an executive summary emphasizing performance metrics.",
            "text": "RetailGiant Corp reported mixed Q1 2024 results. Total revenue remained flat at $12.3 billion, with online sales growing 25% while brick-and-mortar declined 8%. The company closed 150 underperforming locations.",
            "answer": "RetailGiant Q1 2024: Flat $12.3B revenue. Online up 25%, stores down 8%. Closed 150 locations."
        },
        {
            "id": "edtsum_mock_5",
            "query": "Summarize this document focusing on financial performance.",
            "text": "Energy Dynamics Inc. presented its fiscal 2024 year-end results. Annual revenue reached $15.6 billion, up 8% from fiscal 2023. Net income was $1.9 billion with a return on equity of 14.2%. The company invested $2.3 billion in capital expenditures.",
            "answer": "Energy Dynamics FY2024: $15.6B revenue (+8%), $1.9B net income, 14.2% ROE. $2.3B capex."
        },
    ]
