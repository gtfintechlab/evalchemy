# AGENTS.MD

This document provides **extensive and complete guidance** for an AI coding agent tasked with integrating the FLaME evaluation tasks into the Evalchemy framework. It outlines objectives, codebase comprehension, step-by-step implementation instructions, and validation procedures.

---

## 1. Objective

* **Primary Goal:** Adapt all LM evaluation tasks defined in the FLaME repository (e.g., FinQA, FPB, FOMC) for use within the Evalchemy harness, producing new `BaseBenchmark` subclasses under `eval/chat_benchmarks/`.
* **End Deliverable:** A set of Python modules (`eval_instruct.py` per task), registered automatically by Evalchemy’s `TaskManager`, each implementing:

  * `generate_responses(model) -> Dict`: inference logic producing model outputs for all examples.
  * `evaluate_responses(results) -> Dict[str, float]`: metric computation (accuracy, F1, etc.) based on model outputs.

## 2. Prerequisites

* Familiarity with:

  * **Python** (3.8+).
  * **Hugging Face Datasets API** (`datasets` library).
  * **Evalchemy architecture** (especially `BaseBenchmark`, `TaskManager`, and `Instance` types).
  * **FLaME repository structure** under `src/flame/`.
* Required libraries:

  * `datasets`
  * `numpy`
  * `scikit-learn` (for classification metrics)
  * `pandas` (optional for DataFrame operations)
  * `torch` (optional, if local PyTorch model wrappers are used)

## 3. Repository Layouts

### 3.1 Evalchemy

```
evalchemy/
├── eval/
│   ├── chat_benchmarks/
│   │   ├── existing_task1/
│   │   │   └── eval_instruct.py  # defines class ExistingBenchmark(BaseBenchmark)
│   │   └── existing_task2/
│   ├── core/
│   ├── lm_eval_harness/        # underlying harness code
│   └── eval.py                 # CLI entrypoint invoking TaskManager
└── setup.py
```

Key modules:

* `BaseBenchmark` (abstract base class with: `_prepare_messages()`, `compute()`, stub methods).
* `TaskManager` (discovers `eval/chat_benchmarks/*/eval_instruct.py`, loads first `BaseBenchmark` subclass).
* `Instance` (represents a single inference prompt with generation parameters).

### 3.2 FLaME

```
gtfintechlab/flame/
├── src/flame/
│   ├── finqa.py      # defines finqa_inference(), finqa_evaluate()
│   ├── fpb.py        # defines fpb_inference(), fpb_evaluate()
│   ├── fomc.py       # defines fomc_inference(), fomc_evaluate()
│   ├── prompts/
│   └── fewshot.py
└── README.md
```

Key patterns:

* **Inference function:** loads dataset, builds prompt(s), uses `process_batch_with_retry()` to call LLM API, collects outputs in DataFrame.
* **Evaluation function:** processes outputs (parsing, numeric comparison or classification metrics), optionally uses LLM for judging.

---

## 4. Integration Strategy Overview

1. **Scaffold new tasks** under `eval/chat_benchmarks/{finqa, fpb, fomc}/eval_instruct.py`.
2. **Subclass `BaseBenchmark`** for each FLaME task:

   * Implement `__init__()` to set dataset names, splits, prompt formats, generation params.
   * Implement `generate_responses(self, model)`:

     * Load HF dataset.
     * Convert each example into an `Instance` with properly formatted prompt.
     * Invoke `self.compute(model, instances)` to generate outputs.
     * Return a results dict carrying model outputs and ground truths.
   * Implement `evaluate_responses(self, results)`:

     * Extract list of outputs.
     * Parse predictions (numeric extraction or first-line label).
     * Compute metrics (accuracy, precision, recall, F1).
     * Return metrics dict.
3. **Ensure automatic discovery** by naming files and classes correctly.
4. **Test each task** end-to-end on a small model.

---

## 5. Detailed Walkthrough per Task

### 5.1 FinQA Task Integration

#### 5.1.1 File & Class Setup

* **Path:** `eval/chat_benchmarks/finqa/eval_instruct.py`
* **Class:** `class FinQABenchmark(BaseBenchmark):`

  * Set class variable `REQUIRES_OPENAI_ANNOTATOR = False` (we’ll use code-based numeric comparison).

#### 5.1.2 **init**

```python
from evalchemy.core import BaseBenchmark
class FinQABenchmark(BaseBenchmark):
    def __init__(
        self,
        split: str = "test",
        max_new_tokens: int = 128,
        temperature: float = 0.0
    ):
        super().__init__(
            name="finqa",
            description="Financial QA from FLaME integrated into Evalchemy",
        )
        self.split = split
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
```

#### 5.1.3 generate\_responses

```python
import datasets
from evalchemy.core import Instance

def generate_responses(self, model):
    # Load dataset
    ds = datasets.load_dataset(
        "gtfintechlab/finqa", split=self.split, trust_remote_code=True
    )
    instances = []
    for idx, item in enumerate(ds):
        # Build combined context
        context = (
            " ".join(item["pre_text"]) + " " +
            " ".join(item["post_text"]) + " " +
            " ".join([" ".join(r) for r in item["table_ori"]])
        )
        prompt = (
            f"Discard all previous instructions. Behave like a financial expert in QA.\n"
            f"Context: {context}. Question: {item['question']}."
            f" Repeat your final answer at the end."
        )
        messages = [{"role": "user", "content": prompt}]
        formatted = self._prepare_messages(messages, model)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        instances.append(
            Instance("generate_until", item, (formatted, gen_kwargs), idx)
        )
    # Generate
    outputs = self.compute(model, instances)
    # Only on rank 0
    if model.rank != 0:
        return None
    # Collect results
    results = {
        "outputs": [o for o in outputs],
        "ground_truths": [item['answer'] for item in ds]
    }
    return results
```

#### 5.1.4 evaluate\_responses

```python
def evaluate_responses(self, results):
    if results is None:
        return None
    preds = results['outputs']
    trues = results['ground_truths']
    correct = 0
    for pred, true in zip(preds, trues):
        p = pred.strip()
        t = true.strip()
        # Normalize percentages
        if p.endswith('%'): p = p[:-1]
        if t.endswith('%'): t = t[:-1]
        # Numeric compare
        try:
            pv, tv = float(p), float(t)
            dp = len(t.split('.')[-1]) if '.' in t else 0
            if round(pv, dp) == round(tv, dp):
                correct += 1
        except:
            if p.lower() == t.lower():
                correct += 1
    accuracy = correct / len(preds)
    return {"accuracy": accuracy}
```

### 5.2 FPB Task Integration

* **Folder:** `eval/chat_benchmarks/fpb/`
* **Class:** `class FPBBenchmark(BaseBenchmark)`
* **Dataset:** `gtfintechlab/financial_phrasebank_sentences_allagree`, name="5768"
* **Prompt:** First line: label, second line: explanation.
* **Metrics:** accuracy, precision, recall, f1 (weighted).

> **See code sample in Section 5.1** (analogous structure, replacing dataset & parsing logic).

### 5.3 FOMC Task Integration

* **Folder:** `eval/chat_benchmarks/fomc/`
* **Class:** `class FOMCBenchmark(BaseBenchmark)`
* **Dataset:** `gtfintechlab/fomc_communication`
* **Prompt:** One-word answer: HAWKISH, DOVISH, NEUTRAL.
* **Metrics:** accuracy, precision, recall, f1 (weighted).

> **See code sample in Section 5.1** (analogous structure, replacing dataset & parsing logic).

---

## 6. Testing & Validation

1. **Smoke Test:** Run on a tiny subset (modify code to slice ds\[:5]) to verify no errors.
2. **Full Run:** Execute:

   ```bash
   python -m eval.eval \
     --model_name <MODEL> \
     --tasks finqa,fpb,fomc \
     --batch_size 8 \
     --max_new_tokens 128
   ```
3. **Verify Metrics:** Check that metrics are computed and reported under each task.
4. **Error Handling:** Ensure graceful handling of empty/model-failed outputs.

---

## 7. Notes & Best Practices

* **Prompt Tuning:** Keep prompts concise; embed critical instructions.
* **Batching:** Leverage `self.compute` for efficient batching; do not re-batch manually.
* **Reproducibility:** Use deterministic generation (`temperature=0`) for classification tasks.
* **Extensibility:** To add few-shot variants, parameterize prompt assembly in `__init__`, and accept `few_shot_examples` list.