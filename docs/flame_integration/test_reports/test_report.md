# FLaME Tasks Test Report
==================================================

## Mock Data Validation
Passed: 8/8

- ✓ fomc: Mock data valid for fomc
- ✓ fpb: Mock data valid for fpb
- ✓ headlines: Mock data valid for headlines
- ✓ causal_classification: Mock data valid for causal_classification
- ✓ numclaim: Mock data valid for numclaim
- ✓ banking77: Mock data valid for banking77
- ✓ finqa: Mock data valid for finqa
- ✓ convfinqa: Mock data valid for convfinqa

## Mock Data Retrieval
Passed: 8/8

- ✓ fomc: Retrieved 2 mock samples for fomc
- ✓ fpb: Retrieved 2 mock samples for fpb
- ✓ headlines: Retrieved 2 mock samples for headlines
- ✓ causal_classification: Retrieved 2 mock samples for causal_classification
- ✓ numclaim: Retrieved 2 mock samples for numclaim
- ✓ banking77: Retrieved 2 mock samples for banking77
- ✓ finqa: Retrieved 2 mock samples for finqa
- ✓ convfinqa: Retrieved 2 mock samples for convfinqa

## Initialization
Passed: 7/7

- ✓ fomc: fomc initialized successfully
- ✓ fpb: fpb initialized successfully
- ✓ finqa: finqa initialized successfully
- ✓ headlines: headlines initialized successfully
- ✓ numclaim: numclaim initialized successfully
- ✓ convfinqa: convfinqa initialized successfully
- ✓ banking77: banking77 initialized successfully

## Dataset Loading
Passed: 7/7

- ✓ fomc: Loaded 5 examples
- ✓ fpb: Loaded 5 examples
- ✓ finqa: Loaded 5 examples
- ✓ headlines: Loaded 5 examples
- ✓ numclaim: Loaded 5 examples
- ✓ convfinqa: Loaded 5 examples
- ✓ banking77: Loaded 5 examples

## Prompt Generation
Passed: 5/7

- ✓ fomc: Generated prompt of length 391
- ✓ fpb: Generated prompt of length 496
- ✓ finqa: Generated prompt of length 678
- ✗ headlines: Generated prompt of length 396
  Details: {
  "prompt_length": 396,
  "prompt_preview": "Classify the following financial headline based on what it implies about gold price movement.\n\nAn UP..."
}
- ✓ numclaim: Generated prompt of length 537
- ✓ convfinqa: Generated prompt of length 466
- ✗ banking77: Generated prompt of length 2049
  Details: {
  "prompt_length": 2049,
  "prompt_preview": "Discard all the previous instructions. Behave like you are an expert at fine-grained single-domain i..."
}

## Response Generation
Passed: 0/7

- ✗ fomc: Failed to generate responses: 'MockLM' object has no attribute 'world_size'
- ✗ fpb: Failed to generate responses: 'MockLM' object has no attribute 'world_size'
- ✗ finqa: Failed to generate responses: 'MockLM' object has no attribute 'world_size'
- ✗ headlines: Failed to generate responses: 'MockLM' object has no attribute 'world_size'
- ✗ numclaim: Failed to generate responses: 'MockLM' object has no attribute 'world_size'
- ✗ convfinqa: Failed to generate responses: 'MockLM' object has no attribute 'world_size'
- ✗ banking77: Failed to generate responses: 'MockLM' object has no attribute 'world_size'

## Output Parsing
Passed: 9/11

- ✓ fomc: Parse 'HAWKISH...' -> 1 (expected 1)
- ✓ fomc: Parse 'The stance is DOVISH...' -> 0 (expected 0)
- ✓ fomc: Parse 'neutral...' -> 2 (expected 2)
- ✓ fomc: Parse 'Invalid response...' -> -1 (expected -1)
- ✓ fpb: Parse 'POSITIVE
Good outlook...' -> 2 (expected 2)
- ✓ fpb: Parse 'NEGATIVE...' -> 0 (expected 0)
- ✓ fpb: Parse 'The sentiment is neutral...' -> 1 (expected 1)
- ✓ fpb: Parse 'No clear label...' -> -1 (expected -1)
- ✗ finqa: Parse 'The answer is 1000...' -> is 1000 (expected 1000)
  Details: {
  "input": "The answer is 1000",
  "parsed": "is 1000",
  "expected": "1000"
}
- ✗ finqa: Parse 'Final answer: $1,500...' -> $1,500 (expected 1500)
  Details: {
  "input": "Final answer: $1,500",
  "parsed": "$1,500",
  "expected": "1500"
}
- ✓ finqa: Parse 'Revenue: 2.5 million...' -> 2.5 (expected 2.5 million)

## Evaluation Metrics
Passed: 0/7

- ✗ fomc: Failed to compute metrics: 'MockLM' object has no attribute 'world_size'
- ✗ fpb: Failed to compute metrics: 'MockLM' object has no attribute 'world_size'
- ✗ finqa: Failed to compute metrics: 'MockLM' object has no attribute 'world_size'
- ✗ headlines: Failed to compute metrics: 'MockLM' object has no attribute 'world_size'
- ✗ numclaim: Failed to compute metrics: 'MockLM' object has no attribute 'world_size'
- ✗ convfinqa: Failed to compute metrics: 'MockLM' object has no attribute 'world_size'
- ✗ banking77: Failed to compute metrics: 'MockLM' object has no attribute 'world_size'

## Summary
- Total Tests: 62
- Passed: 44
- Failed: 18
- Pass Rate: 71.0%
