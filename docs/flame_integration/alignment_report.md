# FLaME-Evalchemy Alignment Validation Report
==================================================

## 1. Prompt Generation Alignment
- fomc: ✓ ALIGNED
- fpb: ✓ ALIGNED
- finqa: ✓ ALIGNED

## 2. Output Parsing Alignment
- fomc: ✓ ALIGNED
- fpb: ✓ ALIGNED
- finqa: ✗ MISALIGNED
  - 3 parsing differences found

## 3. Metric Calculation Alignment
- fomc: ✓ ALIGNED
- fpb: ✓ ALIGNED
- finqa: ✓ ALIGNED

## 4. Edge Case Handling
- fomc: ✗ FAILED
  - Multiple labels: Expected -1, got 0
- fpb: ✓ PASSED
- finqa: ✓ PASSED

## Summary
- Prompt Alignment: 3/3 tasks aligned
- FLaME modules available: No
