# Integrating FLaME Tasks into Evalchemy

## Evalchemy’s Custom Task Architecture

**Task Interface (BaseBenchmark):** Evalchemy defines an abstract base class `BaseBenchmark` for evaluation tasks. Each custom task is implemented as a subclass of `BaseBenchmark` with two key methods:

* `generate_responses(model: LM) -> Dict[str, Any]`: Should generate model outputs for all instances of the task and return a results dictionary (typically containing model outputs and any identifiers or metadata).
* `evaluate_responses(results: Dict[str, Any]) -> Dict[str, float]`: Should compute evaluation metrics from the model’s outputs (the `results` from `generate_responses`) and return a dictionary of metric names to values.

Each `BaseBenchmark` subclass can also define an `__init__` to load datasets or configure parameters (e.g. dataset name, prompt variations, etc.). The base class provides utilities like `_prepare_messages` to format prompts with system instructions and handle chat templates. It also provides a `compute()` helper that dispatches a list of prompt **Instances** to the model and gathers the generated outputs (with support for distributed/multi-GPU execution). An **Instance** in this context represents a single evaluation prompt, including the input prompt message(s) and generation parameters (like max tokens).

**Task Discovery and Management:** Evalchemy’s `TaskManager` dynamically loads task classes from the `eval/chat_benchmarks/` directory. Each subfolder in `chat_benchmarks` represents a benchmark, containing an `eval_instruct.py` file that defines a `BaseBenchmark` subclass. The folder name becomes the task name. For example, a folder `finqa/` with `eval_instruct.py` defining class `FinQABenchmark` would be loaded as task `"finqa"`. The `TaskManager` scans the directory, imports each `eval_instruct.py`, and registers the first subclass of `BaseBenchmark` it finds. It instantiates each task, optionally passing in any `benchmark_kwargs` that match the task’s `__init__` parameters (this is how task-specific options can be provided). The `TaskManager` stores instances in a registry and is used to run evaluations.

**Evaluation Flow:** To run evaluations, Evalchemy calls each task’s `run_benchmark` method, which by default simply calls `generate_responses` then `evaluate_responses`. The results for each task are collected and returned or saved. This design cleanly separates **generation** (inference) from **evaluation** (scoring), allowing custom logic at each stage. Notably, tasks can incorporate special handling in these methods (for example, using an auxiliary model to judge outputs). Evalchemy even allows marking tasks with `REQUIRES_OPENAI_ANNOTATOR = True` to indicate the evaluation step will call an OpenAI model for judging.

**Datasets and Metrics:** Evalchemy tasks often use Hugging Face datasets for inputs. A task’s `generate_responses` can load a dataset (via `datasets.load_dataset`) and iterate through the test samples, creating an `Instance` per sample. The `Instance` typically includes the prompt (possibly structured as a chat message with a system/user role) and generation parameters like `max_new_tokens`. The outputs returned by the model (usually raw text completions) are then post-processed in `evaluate_responses` to compute metrics such as accuracy, F1, etc., depending on the task. Each task defines its own metrics – the return of `evaluate_responses` is a dict of metric names to values (which the framework will report).

## Overview of FLaME Task Implementations

FLaME implements tasks as **functions** for inference and evaluation rather than classes. In the `src/flame/` code, each task (e.g. FinQA, FPB, FOMC) has:

* An **inference function** (e.g. `finqa_inference`, `fpb_inference`, `fomc_inference`) that loads the dataset, formats prompts for the model, batches the inputs, and calls the language model API to get outputs.
* An **evaluation function** (e.g. `finqa_evaluate`, `fpb_evaluate`, `fomc_evaluate`) that takes the model outputs (often saved as a CSV or DataFrame) and computes metrics, possibly using additional model calls for parsing or judging outputs.

Despite the procedural style, the logic in these functions maps closely to what we need to implement in `generate_responses` and `evaluate_responses` for Evalchemy. Key aspects for the example tasks:

**FinQA (Financial Question Answering):** This task involves answering a financial question given a context of textual data and tables. In FLaME:

* **Input format:** Each sample provides some text (`pre_text` and `post_text` segments) and a table (`table_ori`), plus a `question`. The FLaME inference function constructs a single string by concatenating pre\_text, post\_text, the table (flattened), and the question. This combined **document** is the context+question for the model.
* **Prompting:** FinQA can be done zero-shot or few-shot. In zero-shot, FLaME’s prompt template instructs: *“Behave like a financial expert in question answering. Your task is to answer a financial question based on the provided context… The context: {document}. Repeat your final answer at the end of your response.”*. This ensures the model provides an answer (often with reasoning) and then states the final answer explicitly at the end.
* **Inference (generation):** The FLaME `finqa_inference` loads the HF dataset `"gtfintechlab/finqa"` (test split). It then prepares prompts for each sample using the above template (selected via `get_prompt("finqa", PromptFormat.ZERO_SHOT)` in the code). Each prompt is wrapped as a user message for a chat completion call. The function calls `process_batch_with_retry` to get model responses for a batch of prompts. Outputs are collected into a DataFrame with columns like `context` (the input text), `response` (the model’s answer), and `actual_label` (the ground-truth answer).
* **Evaluation:** The ground truth and model answers are typically numbers (or short texts). FinQA’s evaluation in FLaME is more involved: it uses the model itself (or another LLM) to judge correctness. First, it runs an **extraction step** to pull out the numeric answer from the model’s possibly long response. This uses a prompt (registered as a generic QA extraction prompt) that asks: *“Extract and return only the main/final answer… If a final answer was not provided, respond NA.”*. Then, it feeds the extracted model answer and the true answer into another prompt (`evaluate_answer`) that instructs the model to compare them with tolerance: e.g. allowing differences only in precision or formatting. The model’s judgment (“correct” or “wrong”) for each example is recorded and accuracy is computed. Essentially, FinQA uses an LLM-based evaluator to handle cases like “5” vs “5%” or rounding differences. The primary metric is **accuracy** (fraction of answers deemed correct).

**FPB (Financial Phrase Bank Sentiment):** This is a sentiment classification task for financial sentences.

* **Input format:** Each sample is a single financial news sentence with a sentiment label (Positive, Negative, or Neutral).
* **Prompting:** The prompt instructs the model to classify the sentence into *'NEGATIVE', 'POSITIVE', or 'NEUTRAL'* and to provide the label in the first line and a brief explanation in the second line. (FLaME provides multiple prompt variants for experimentation, but the core idea is the same: first line = predicted label, second line = explanation).
* **Inference:** The FLaME `fpb_inference` function loads the dataset `"financial_phrasebank_sentences_allagree"` (a specific config `"5768"` for the split). It retrieves either a few-shot or zero-shot prompt template via `get_prompt("fpb", ...)`. In code, FLaME often uses a variant called `"flame"` format for FPB, which internally sets up a system message and user prompt with slightly different wording (including an explicit request for an explanation). However, ultimately the prompt is given to the model as a single combined string (system and user instructions concatenated) in the user role. The inference function sends batches of sentences to the model and collects responses. Each response should look like, for example:

  ```
  POSITIVE
  The statement indicates an optimistic outlook for the company’s profits.
  ```

  The code then appends the results to a DataFrame with columns: `sentences`, `llm_responses` (the model’s raw output label text), `actual_labels` (true sentiment), etc..
* **Evaluation:** FLaME’s `fpb_evaluate` reads the model outputs and focuses on extracting the predicted label from each response. Since the model might produce extra text (explanation), they use an LLM extraction prompt: *“Based on the labels NEGATIVE, POSITIVE, NEUTRAL, extract the most relevant label from the response.”*. In practice, this just pulls the first line or the key word. The extracted label is then mapped to a numeric code (NEGATIVE→0, NEUTRAL→1, POSITIVE→2). With predicted and true labels as integers, standard classification metrics are computed: **Accuracy**, **Precision**, **Recall**, **F1-score** (F1 and other metrics are weighted across the 3 classes in their implementation). Any outputs where no valid label was found are mapped to -1 and effectively counted as errors (the FOMC evaluator filters these out for metric calculation, as we’ll see). In short, FPB’s evaluation reduces to standard sentiment classification metrics once the model’s text output is turned into a label.

**FOMC (Fed Communications Hawkish/Dovish Classification):** This task classifies sentences from Federal Reserve meeting minutes as Hawkish, Dovish, or Neutral.

* **Input format:** Each sample is a sentence (likely a quote or statement from FOMC text) with one of three labels (Hawkish = indicating tightening stance, Dovish = easing stance, Neutral).
* **Prompting:** The zero-shot prompt explicitly asks: *“Classify the following statement as HAWKISH, DOVISH, or NEUTRAL. Statement: {sentence}. Provide only one word as your answer: HAWKISH, DOVISH, or NEUTRAL.”*. This clearly instructs the model to output a single-word label.
* **Inference:** The `fomc_inference` function loads the `"gtfintechlab/fomc_communication"` dataset (test split). It sets up the prompt for each sentence using the above template. Batches of sentences are sent to the model via `process_batch_with_retry` similar to FPB. The results are collected with columns: `sentences`, `llm_responses` (model outputs), `actual_labels` (true labels), etc.. The model’s response in this case is expected to be just one of the words “HAWKISH”, “DOVISH”, or “NEUTRAL” (though it might occasionally include a newline or extra text, depending on the model’s behavior).
* **Evaluation:** The `fomc_evaluate` function again uses an extraction step to ensure we get a clean label from the model output (similar to FPB). The prompt for extraction is analogous: *“Based on labels ‘DOVISH’, ‘HAWKISH’, ‘NEUTRAL’, extract the label from the response… output only the label.”* (defined in the registry). Extracted labels are mapped to numeric codes (DOVISH→0, HAWKISH→1, NEUTRAL→2) to compare with ground truth. The evaluator in FLaME explicitly filters out any cases where no valid label was extracted (label = -1) before computing metrics. Metrics computed are **Accuracy, Precision, Recall, F1** (weighted) just like FPB. In essence, FOMC is handled very similarly to FPB, just with different label names and a strict “one-word answer” prompt.

**Summary:** Across these tasks, FLaME’s approach is to: (a) load the appropriate dataset, (b) prepare a prompt that instructs the model how to respond (often including format instructions like “answer with one word” or “give label then explanation”), (c) run the model to get outputs, and (d) post-process those outputs to compute metrics. Post-processing often involves either simple string parsing (for classifications) or invoking an LLM to help evaluate correctness (for generative answers like FinQA). The **core challenge** in integrating these into Evalchemy is to repackage this logic into the `BaseBenchmark` interface.

## Strategy for Adapting FLaME Tasks to Evalchemy

We will integrate FinQA, FPB, and FOMC into Evalchemy by creating new benchmark classes for each, implementing their logic in `generate_responses` and `evaluate_responses`. Below is a step-by-step plan:

### 1. **Create Task Folders and Classes**

For each task, create a new subdirectory under `eval/chat_benchmarks/` (assuming we keep the naming consistent with other Evalchemy tasks). For example:

* `eval/chat_benchmarks/finqa/eval_instruct.py`
* `eval/chat_benchmarks/fpb/eval_instruct.py`
* `eval/chat_benchmarks/fomc/eval_instruct.py`

In each `eval_instruct.py`, define a class subclassing `BaseBenchmark`. For clarity, we can name them `FinQABenchmark`, `FPBBenchmark`, and `FOMCBenchmark` respectively (the Evalchemy loader will strip the “Benchmark” suffix for the task name, but naming isn’t critical as long as it’s unique and a subclass of `BaseBenchmark`).

Each class will have an `__init__` to set up any parameters (e.g. dataset names, prompt format flags) and must implement `generate_responses` and `evaluate_responses`. We should also import any needed libraries (e.g. `datasets`, `numpy/pandas` for metrics, etc.) at the top of the file.

### 2. **Implement `FinQABenchmark`**

**Initialization:** In `FinQABenchmark.__init__`, set defaults for the dataset:

* Dataset name: `"gtfintechlab/finqa"` (Hugging Face dataset).
* Split: `"test"` (since we evaluate on the test set by default; we could make this configurable).
* Optionally, a parameter for `prompt_format` (to choose between zero-shot or few-shot). Initially, we might default to `"zero_shot"` for simplicity. If we include this param, we can allow passing `"few_shot"` via `benchmark_kwargs` if needed.
* Generation parameters like `max_tokens`, `temperature`, etc., can be accepted or defaulted. (Evalchemy’s CLI can pass common generation args like `max_new_tokens` via the model, but we might expose some if needed for this task specifically.)

Initialize the base class via `super().__init__(logger=..., system_instruction=...)`. (We may not need a system\_instruction for FinQA; the prompt itself encodes the instruction. We can leave it None or use it for the “Discard instructions…” part if we wanted to use role separation.)

**Generate Responses:** In `FinQABenchmark.generate_responses(self, model)`, do the following:

* **Load the dataset:** Use Hugging Face datasets API to load the FinQA test set. For example:

  ```python
  import datasets
  dataset = datasets.load_dataset("gtfintechlab/finqa", split="test", trust_remote_code=True)
  ```

  (We include `trust_remote_code=True` because the dataset might have custom loading code.) If we made the split configurable, use that.

* **Prepare Instances:** Iterate over each example in the dataset and construct the prompt. For each example:

  * Combine the context: join `pre_text`, `post_text`, table rows, and question as FLaME did. For instance:

    ```python
    context_parts = " ".join(example["pre_text"]) + " " + \
                    " ".join(example["post_text"]) + " " + \
                    " ".join([" ".join(row) for row in example["table_ori"]]) + " " + \
                    example["question"]
    ```

    This yields a single string with all relevant text.
  * Formulate the prompt message. We can embed FLaME’s zero-shot prompt text directly:

    ```python
    prompt_text = (f"Discard all the previous instructions. Behave like you are a financial expert in question answering. "
                   f"Your task is to answer a financial question based on the provided context.\n\n"
                   f"The context: {context_parts}. Repeat your final answer at the end of your response.")
    ```

    This prompt is essentially the same as `finqa_zeroshot_prompt`, now filled with the actual context.
  * Create a message list for the model: e.g. `messages = [{"role": "user", "content": prompt_text}]`. Then use `self._prepare_messages(messages, model)` to get the formatted prompt (this will prepend any system instruction if set, and some model classes might turn it into a single string template).
  * Create an `Instance` for this prompt. Evalchemy’s `Instance` likely takes arguments like: `(method_name, reference, (prompt, generation_args), instance_id)`. In the Alpaca example, they do:

    ```python
    Instance(
        "generate_until",
        example,  # they pass the raw example as reference
        (formatted_prompt, {"max_new_tokens": self.max_tokens, "do_sample": self.do_sample, "temperature": self.temperature}),
        idx
    )
    ```

    . We can mirror this. The `method_name` `"generate_until"` is used internally by the LM wrapper to choose the generation method (for chat models, it typically means generate until stop token or end of message).

    * We include `example` (which contains the true answer in `example["answer"]`) so that the `Instance` carries ground truth info, if needed later.
    * Provide generation kwargs: `max_new_tokens` (maybe 100 or 256 for FinQA, to allow the model to explain and answer), and possibly a temperature (FinQA might not require much randomness, maybe use `temperature=0` or a small value for deterministic output since it’s an evaluation).
    * The `idx` is just the index of the example.

* **Collect Instances:** Append each instance to a list `all_instances`. If needed, implement a debug mode to only take a subset (similar to Alpaca’s `debug` flag that limits to 2 examples for quick testing).

* **Model Generation:** Once all instances are prepared, call `outputs = self.compute(model, all_instances)`. The `compute` method will handle sending prompts to the model (possibly distributed across GPUs if running multi-GPU) and return a list of outputs (each output corresponds to an instance, typically a generated string). We should wrap this in `torch.no_grad()` if using PyTorch models, to avoid gradient overhead (the Alpaca code does this with `with torch.no_grad():` around the generate call).

* **Post-process outputs:** After generation, if using multiple processes (MPI / distributed), only rank 0 will have the full set of outputs (the BaseBenchmark.compute merges them across ranks). We check:

  ```python
  if model.rank != 0:
      return None
  ```

  to ensure only the main process proceeds to evaluation (as done in AlpacaBenchmark).

  Now, construct a results structure. We can follow Evalchemy’s convention or create our own. A simple approach: build a list of dictionaries `model_outputs`, where each dict contains:

  * The prompt context or question (for logging if needed),
  * The model’s output (answer),
  * The ground truth answer.
    For example:

  ```python
  model_outputs = []
  for i, (example, output) in enumerate(zip(dataset, outputs)):
      model_outputs.append({
          "context": f"{example['pre_text']} ... {example['question']}",  # or the combined context if needed
          "predicted_answer": output,
          "true_answer": example["answer"]
      })
  }
  ```

  We might not include the entire context to avoid huge logs, maybe just the question or an identifier. The key is to carry `true_answer` for evaluation.

  Additionally, we can record the model identifier (name) to use in results (Evalchemy often logs the model name in metrics). For instance:

  ```python
  result_dict = {
      "model_outputs": model_outputs,
      "model_identifier": model.model_identifier  # if LM model has such an attribute
  }
  ```

  The Alpaca benchmark does similar, storing outputs and the model id.

* **Return results:** Return the `result_dict`. This will be passed to `evaluate_responses`.

**Evaluate Responses:** In `FinQABenchmark.evaluate_responses(self, results)`:

* First, handle the case where `results is None` (which would mean we’re on a non-master process in a distributed setting): just return `None` in that case.

* Extract the `model_outputs` list and possibly the `model_identifier` if needed:

  ```python
  outputs = results["model_outputs"]
  if not outputs:
      raise ValueError("No model outputs to evaluate")
  ```

* **Compute accuracy of answers:** We need to compare each predicted answer to the true answer, allowing tolerances similar to FLaME. We have two possible approaches:

  1. **Programmatic Numeric Comparison:** Implement a function to compare the predicted answer vs true answer:

     * If both are numeric (or can be interpreted as numeric), consider them equal if they match up to rounding or formatting differences. For example:

       * Strip off any trailing `%` sign from both answers for comparison (so "5%" becomes "5").
       * Convert both to decimal numbers if possible. We can use Python’s `Decimal` or float with rounding. One approach: determine the number of decimal places in the true answer string, then round the predicted number to that many decimals and compare.
       * Example: True = "1.00", Pred = "1". Convert true to float(1.0) or Decimal, pred to float(1.0), they match after rounding pred to 0 decimals (pred is 1, true is 1.00 → match). True = "1.02", Pred = "1": true is 1.02, pred is 1.0 – rounding pred to 2 decimals gives 1.00, which is not 1.02, so not a match. True = "5", Pred = "5%": strip "%", both become 5, match.
       * If answers are textual (not purely numeric), we can do a direct case-insensitive match after stripping punctuation. FinQA likely expects mostly numeric answers, but if there are text answers (e.g. "July"), an exact match (or containment) might be required. We can handle those on a case-by-case basis (exact string match).
     * Count how many predictions match the criteria for correctness.
     * Compute accuracy = correct\_count / total\_count.

  2. **LLM-based Evaluation (optional):** Alternatively, we can replicate FLaME’s approach using an evaluator LLM. Evalchemy could support this by leveraging the `annotator_model` mechanism:

     * We could add `annotator_model: str` as a parameter to `FinQABenchmark.__init__`, and set `REQUIRES_OPENAI_ANNOTATOR = True` on the class. Then, in `evaluate_responses`, we could call an OpenAI API (or another model) to judge each answer pair. For example, using the same prompt FLaME uses (defined in `evaluate_answer` function), we could prompt GPT-4 with each (predicted\_answer, correct\_answer) pair and parse its "correct"/"wrong" decision.
     * However, this adds external dependencies and latency. Unless high evaluation fidelity is needed, a coded comparison might suffice. In many cases, numeric answers can be checked with simple rules. We might choose to implement code-based evaluation for now (for offline reproducibility), and document that an LLM evaluator could be integrated for edge cases.

Given the complexity, a reasonable plan is to **implement a deterministic numeric comparison** for FinQA:

```python
correct = 0
for output in outputs:
    pred = str(output["predicted_answer"]).strip()
    true = str(output["true_answer"]).strip()
    # strip percentage sign
    pred_norm = pred[:-1] if pred.endswith('%') else pred
    true_norm = true[:-1] if true.endswith('%') else true
    # try numeric comparison
    try:
        pred_val = float(pred_norm)
        true_val = float(true_norm)
        # determine precision of true (if any decimal point)
        if '.' in true_norm:
            dec_places = len(true_norm.split('.')[-1])
            rounded_pred = round(pred_val, dec_places)
            match = (rounded_pred == round(true_val, dec_places))
        else:
            # true is integer
            match = (round(pred_val) == int(round(true_val)))
        # Also consider string exact match as a fallback for cases like text answers
    except:
        # Non-numeric comparison (fall back to case-insensitive strip)
        match = (pred_norm.lower() == true_norm.lower())
    if match:
        correct += 1
accuracy = correct / len(outputs)
```

We will also consider partial credit if needed (FinQA might not need it; each question has one answer). The final metric dictionary can simply be `{"accuracy": accuracy}`. We might also include `num_examples` for reference.

* (If using LLM to evaluate: we would instead accumulate the model’s “correct”/“wrong” judgments and compute accuracy. But we will proceed with the above logic for now.)

* **Return metrics:** Return a dict, e.g.:

  ```python
  metrics = {"accuracy": accuracy}
  ```

  We can also attach metadata similar to AlpacaEval’s output. For example, AlpacaEval returns a leaderboard with model scores; for us, a simple accuracy suffices. We might include `"completion_rate"` if we want to note how many answers were non-empty (though if the model fails to produce an answer it would be rare – we could detect `output == None`). Since we counted only matches out of total, completion\_rate isn’t critical here.

*(At this point, FinQA is integrated as an Evalchemy task. We should test it on a known model to ensure that the generation and evaluation flow produce the expected accuracy on FinQA test set.)*

### 3. **Implement `FPBBenchmark`**

**Initialization:** In `FPBBenchmark.__init__`, set:

* Dataset name: `"gtfintechlab/financial_phrasebank_sentences_allagree"`, and the specific config `"5768"` for the data (as used in FLaME).
* No need for subset selection (we use the default test split provided by that dataset).
* Allow a `prompt_format` parameter (to choose prompt variant). We might default to `"flame"` (the variant used in FLaME’s inference code) or simply zero-shot. Because the “flame” variant in FLaME’s prompt includes an explanation request, and the evaluation logic assumes the model will provide an explanation line, it might be good to use that for consistency. We can default `prompt_format="flame"`.
* We can accept similar generation params (though classification tasks may not need many tokens; maybe limit `max_new_tokens` to e.g. 50 since only a label and a sentence or two of explanation are expected).
* Call `super().__init__()` to set up logger, etc.

**Generate Responses:** In `FPBBenchmark.generate_responses(self, model)`:

* **Load dataset:**

  ```python
  dataset = datasets.load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", name="5768", split="test", trust_remote_code=True)
  ```

  (Passing `name="5768"` picks the specific agreement split; this replicates `safe_load_dataset(..., name="5768")` from FLaME.)

* **Prepare Instances:** Iterate over each sentence in the test set:

  * Get the sentence text (e.g. `sentence = example["sentence"]`) and true label (`label = example["label"]`). The dataset might give label as text (“positive”/“negative”/“neutral”) or as an index. We should check this. In FLaME, they treated `example["label"]` as text and later mapped it to numbers, which implies the dataset’s label is likely a string (or easily convertible to string).
  * Construct the prompt. We’ll use the core of `fpb_zeroshot_prompt`:

    * If `prompt_format == "flame"` (our default), FLaME’s implementation builds a **system** instruction and **user** message, but eventually combined them into one prompt string. We can simplify by combining them ourselves:

      ```python
      prompt_text = (
        "Discard all the previous instructions. Behave like you are an expert sentence sentiment classifier.\n"
        "Classify the following sentence into 'NEGATIVE', 'POSITIVE', or 'NEUTRAL' class. "
        "Label 'NEGATIVE' if it corresponds to negative sentiment, 'POSITIVE' if positive, or 'NEUTRAL' if neutral. "
        "Provide the label in the first line and provide a short explanation in the second line.\n"
        f"This is the sentence: {sentence}"
      )
      ```

      This encapsulates the instructions from FLaME’s flame prompt. We also include the request for an explanation. (We might optionally add “Explain how you came to your decision.” as FLaME did, but it’s not crucial.)
    * If we wanted to implement other formats (like a pure zero-shot without explanation), we could adjust accordingly. But focusing on one format for now is fine.
  * Create the message: `messages = [{"role": "user", "content": prompt_text}]`. (We could also supply a system role message if we wanted to split it, but in practice, the string already contains a system-like instruction to discard previous context.)
  * Use `self._prepare_messages(messages, model)` to format. If the model is an OpenAI chat model, this might wrap it properly; if it’s a plain causal LM, it may just join them.
  * Create an `Instance` with method `"generate_until"`, reference maybe just the sentence or index, and the tuple of (prompt, generation\_args). Generation args can include a small `max_new_tokens` and perhaps `stop` criteria if needed (though not strictly necessary if the model stops naturally after answer).

    ```python
    Instance("generate_until", None, (formatted_prompt, {"max_new_tokens": 50, "temperature": 0.0}), idx)
    ```

    We set temperature 0 for deterministic output (since it’s a classification). We pass `None` or the raw example as reference; including the example can help carry the true label.
  * Append to `instances` list.

* **Generate outputs:** Call `outputs = self.compute(model, instances)` to get the list of model-generated texts.

* **Aggregate results:** As before, ensure we only proceed on `rank == 0`. Then build a list of outputs with their true labels:

  ```python
  results_list = []
  for i, (example, output) in enumerate(zip(dataset, outputs)):
      results_list.append({
          "sentence": example["sentence"],
          "predicted_text": output,       # full text output from model (label + maybe explanation)
          "true_label": example["label"]  # possibly as text or numeric
      })
  }
  results = {"outputs": results_list}
  ```

  (We can omit model\_identifier here; not critical for computing metrics, but we can include it similar to other tasks.)

* Return `results`.

**Evaluate Responses:** In `FPBBenchmark.evaluate_responses(self, results)`:

* If `results is None`, return None.

* Extract `outputs_list = results["outputs"]`.

* **Extract predicted labels:** For each item in `outputs_list`, we need to derive a label from `predicted_text`. Since the model was instructed to put the label as the first line, we can do:

  ```python
  pred_text = item["predicted_text"]
  first_line = str(pred_text).strip().splitlines()[0] if pred_text else ""
  pred_label_str = first_line.upper()  # normalize to uppercase
  ```

  Now `pred_label_str` should be one of "NEGATIVE", "POSITIVE", "NEUTRAL" (or possibly something malformed if the model didn’t follow instructions, e.g., empty). We can map this to the numeric code:

  ```python
  label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
  pred_label = label_map.get(pred_label_str, -1)
  ```

  If the label isn’t recognized, we assign -1 (as an invalid code).

* Collect lists of predicted labels and true labels:

  ```python
  pred_labels = []
  true_labels = []
  for item in outputs_list:
      # determine pred_label as above
      pred_labels.append(pred_label)
      # Map true label to int:
      true = item["true_label"]
      if isinstance(true, str):
          true_num = label_map.get(true.strip().upper(), -1)
      else:
          true_num = int(true)
      true_labels.append(true_num)
  ```

  Now we have numeric arrays for predictions and ground truth. The true labels from the dataset may already be 0/1/2 (for HF datasets, sometimes they are). We handle string case just in case.

* **Compute metrics:** Using these arrays:

  * **Accuracy:** straightforward `(pred == true) / N` (count excluding -1 if any).
  * **Precision, Recall, F1:** We can use `sklearn.metrics.precision_recall_fscore_support` with `average='weighted'` to get overall precision, recall, F1. If we want to avoid introducing a dependency on scikit-learn, we can manually compute:

    * Precision\_weighted = sum\_i (prec\_i \* support\_i) / sum\_i support\_i, similarly for recall, where prec\_i is true positives\_i / predicted positives\_i, etc. Given the small number of classes, implementing by hand is possible but error-prone. Since Evalchemy already may have sklearn (not certain, but likely since other tasks might use it), using sklearn functions is acceptable.
  * Another approach: compute F1 per class and average by support. But using the library is simplest:

    ```python
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    # Filter out invalid preds (pred_label = -1)
    valid_idx = [i for i, p in enumerate(pred_labels) if p != -1 and true_labels[i] != -1]
    if valid_idx:
        y_true = np.array(true_labels)[valid_idx]
        y_pred = np.array(pred_labels)[valid_idx]
    else:
        y_true = np.array(true_labels)
        y_pred = np.array(pred_labels)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    ```

    If we find `valid_idx` (cases where pred was -1) is not empty, we exclude those from metric calculation (treat them as missing predictions). In FLaME’s FPB, they did not explicitly drop -1; they may have none or few, but following FOMC’s robustness, excluding invalid predictions might be safer to not skew metrics.
  * **Support:** We can also report the number of examples evaluated.

* **Return metrics:** e.g.

  ```python
  return {
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1": f1
  }
  ```

  These will be aggregated under the task name in the final output.

*(After this, FPB integration is done. We should test it with a known model (perhaps GPT-3.5 via openAI or a local llama) to see if it produces reasonable outputs and ensure our parsing logic works.)*

### 4. **Implement `FOMCBenchmark`**

**Initialization:** Similar to FPB:

* Dataset: `"gtfintechlab/fomc_communication"` (assuming the HF dataset is named that; FLaME uses `safe_load_dataset("gtfintechlab/fomc_communication")` with trust\_remote\_code).
* No special config name (likely just one version).
* Prompt format: default to `"zero_shot"` (the prompt itself is already a straightforward zero-shot classification request).
* Generation settings: few tokens (one word answer expected, but allow maybe up to 5 tokens just in case; temperature 0 for deterministic classification).
* Call `super().__init__()`.

**Generate Responses:** In `FOMCBenchmark.generate_responses(self, model)`:

* **Load dataset:**

  ```python
  dataset = datasets.load_dataset("gtfintechlab/fomc_communication", split="test", trust_remote_code=True)
  ```

  This gives us examples with `sentence` and `label`.

* **Prepare Instances:** For each example:

  * Get `sentence = example["sentence"]`.
  * Construct prompt:

    ```python
    prompt_text = (f"Classify the following Federal Reserve statement as HAWKISH (indicating a restrictive stance), "
                   f"DOVISH (indicating an accommodative stance), or NEUTRAL (balanced stance).\n"
                   f"Statement: {sentence}\n"
                   "Provide only one word as your answer: HAWKISH, DOVISH, or NEUTRAL.")
    ```

    This is directly from FLaME’s FOMC prompt.
  * `messages = [{"role": "user", "content": prompt_text}]`. (No system message needed beyond what's in prompt.)
  * `formatted = self._prepare_messages(messages, model)` to handle any template.
  * Create `Instance("generate_until", None, (formatted, {"max_new_tokens": 5, "temperature": 0.0}), idx)`.
  * Append to list.

* **Generate outputs:** `outputs = self.compute(model, instances)`.

* **Aggregate results:** On rank 0:

  ```python
  results_list = []
  for example, output in zip(dataset, outputs):
      results_list.append({
          "sentence": example["sentence"],
          "predicted_text": str(output).strip(),
          "true_label": example["label"]
      })
  results = {"outputs": results_list}
  ```

  We strip the output string because ideally it’s just "HAWKISH" etc., but there might be stray whitespace or newline.

* Return `results`.

**Evaluate Responses:** In `FOMCBenchmark.evaluate_responses(self, results)`:

* If `results is None`: return None.

* `outputs_list = results["outputs"]`.

* **Extract predicted labels:** Similar to FPB:

  ```python
  pred_labels = []
  true_labels = []
  label_map = {"DOVISH": 0, "HAWKISH": 1, "NEUTRAL": 2}
  for item in outputs_list:
      # Predicted text should ideally be one word.
      pred = str(item["predicted_text"]).strip().upper()
      # In case the model gave a sentence, take the first token:
      pred_word = pred.split()[0] if pred else ""
      pred_label = label_map.get(pred_word, -1)
      pred_labels.append(pred_label)
      # True label might be string or int:
      true = item["true_label"]
      if isinstance(true, str):
          true_num = label_map.get(true.strip().upper(), -1)
      else:
          true_num = int(true)
      true_labels.append(true_num)
  ```

  We take only the first token of the model’s output to avoid cases where it might have said "NEUTRAL stance" or something – the prompt said one word, but this ensures no extra words interfere.

* **Metrics:** As with FPB:

  * Exclude any pred\_label = -1 from metrics (if none of the expected words were found, though with the prompt, likely the model will choose one of them).
  * Compute accuracy, precision, recall, f1 (weighted). The code would be identical to FPB’s, using `accuracy_score` and `precision_recall_fscore_support` or manual calculation.
  * Because this is a 3-class classification, weighted average is appropriate if class distribution is imbalanced. (If needed, we could also output class-wise metrics, but typically overall metrics suffice.)

* **Return metrics:** e.g.

  ```python
  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
  ```

*(Now FOMC is integrated. Testing on a model with known outputs or small subset would verify our parsing – e.g., if the model sometimes outputs in lowercase or with punctuation, our mapping should catch those by uppercasing and stripping punctuation if needed.)*

### 5. **Integration and Verification**

After implementing the above classes:

* **Registering tasks:** The `TaskManager` will automatically discover these new tasks as long as they are in the `chat_benchmarks` directory with the `eval_instruct.py` file. We should ensure the folder names (`finqa`, `fpb`, `fomc`) match what we want the task identifiers to be. For consistency with others, using all lowercase is fine (Evalchemy lists custom tasks in lowercase like `alpaca_eval`, `math500`, etc.). The class names can be anything, but having “Benchmark” in the name is common and fine.

* **Dependencies:** Ensure the environment has the required packages:

  * `datasets` (for HF datasets),
  * `pandas` or `numpy` (for any DataFrame or array ops; we used numpy for metrics, which is already likely a dependency),
  * `sklearn` (if using it for metrics; if not already in Evalchemy’s dependencies, we may add or handle metrics manually).
  * Since Evalchemy is based on EleutherAI’s LM eval harness, it likely has numpy and maybe sklearn already available.

* **Evalchemy API modifications:** None of the core Evalchemy code needs to change for these tasks, as we are conforming to the existing interface. We simply add new task classes. However, a couple of considerations:

  * **Passing prompt\_format:** If we want users to toggle few-shot vs zero-shot, we introduced a `prompt_format` param in FPB and (maybe) FinQA classes. To allow this via command line, Evalchemy’s `TaskManager` can accept `**benchmark_kwargs`. For example, the user could run:

    ```
    python -m eval.eval --tasks fpb --prompt_format zero_shot
    ```

    if `--prompt_format` is defined to populate `benchmark_kwargs`. If not, we might document that changing prompt format requires modifying the code or adding such an argument in the eval script. (We can also decide to always use one format for now and refine later.)
  * **Annotator model (OpenAI API usage):** If we had chosen to use an OpenAI model for FinQA evaluation, we would need the OPENAI\_API\_KEY environment set and possibly mark the task with `REQUIRES_OPENAI_ANNOTATOR = True`. In our plan, we avoided that for now and used internal logic, so no special framework changes are required. (If later needed, the TaskManager already checks for this flag and will skip loading the task if no API key.)
  * **Thread-safety and batch processing:** Our implementation uses `self.compute` which already handles batching and slicing for distributed eval. Internally, it likely respects a global batch size. We should ensure our tasks do not override that. In our code, we prepared the entire list of instances and let `compute` handle batching. This is fine (Evalchemy’s CLI `--batch_size` will cause `model.generate_until` to internally batch process the instances).
  * **Logging:** We may use `self.logger.info(...)` within our methods to log progress (like “Loaded X examples” or “Generating responses…” similar to Alpaca eval code). This can help track the process during execution.
  * **Results format:** The returned metrics dict will be captured by Evalchemy and likely printed or saved. We should ensure metric names are unique and descriptive (we used simple names like "accuracy", which is fine since each task’s results are namespaced under the task name in the final JSON).

* **Step-by-step testing:**

  1. Run `evalchemy` with our tasks on a small model or in debug mode. For example:

     ```
     python -m eval.eval --model hf-causal --model_name facebook/opt-125m --tasks finqa,fpb,fomc --batch_size 1 --max_new_tokens 100
     ```

     This would load a small OPT model and run our tasks (we might restrict dataset size in code via a debug flag or setting for quick tests). Verify that no errors occur in data loading, generation, or evaluation.
  2. Check that the metrics make sense (e.g., accuracy for a random model will be low; we just want to see that it runs end-to-end).
  3. If any issues arise (for instance, if the dataset loading requires a specific version or the label fields are named differently), adjust accordingly. We might need to inspect the HuggingFace dataset format in a notebook to confirm field names.

### 6. **Possible Enhancements and Modifications**

Finally, consider any enhancements after the basic integration:

* **Few-shot support:** We could implement the few-shot prompts from FLaME’s `fewshot.py` for these tasks. For example, FinQA few-shot prompt might include exemplars of question-answer pairs. This would require storing a few example Q\&As (possibly from the training set) and constructing a prompt accordingly. Similarly, FPB few-shot could include a few labeled example sentences. This is an advanced feature; we can first focus on zero-shot which is already done.
* **Multiple evaluation metrics or detailed breakdown:** For classification tasks, we might report class-wise accuracy too (e.g., accuracy on each label). Evalchemy typically reports just the aggregate, which is fine.
* **Refactoring prompt text:** Instead of hardcoding prompt strings in our code, we could import or replicate the prompt registry from FLaME. However, that would add a lot of extra code. Our direct approach (embedding the needed prompt text) is straightforward and ensures self-containment.
* **Performance considerations:** The FinQA task as implemented will generate possibly long responses (with reasoning). This is fine for thorough evaluation, but if we only care about the final numeric answer, another strategy would be to prompt the model to output just the final answer (e.g., “only output the final numeric answer”). That would simplify evaluation (no need for extracting answers). However, it might change the task definition. Since the original FinQA benchmark likely expects models to reason, we kept the prompt instructing them to possibly give an answer with context. For evaluation, we handled parsing. This alignment with the original evaluation process is important if we want to compare with FLaME results.
* **Use of `Instance.task_name`:** Evalchemy’s BaseBenchmark adds a task name to each instance (see `instance.task_name = ...` in `compute`), which is used by some model wrappers or for logging. We don’t need to manage this; it’s automatic.

In summary, after implementing and testing these classes, we will have integrated FinQA, FPB, and FOMC tasks into Evalchemy’s unified evaluation framework. Each task uses Evalchemy’s standard flow:

* **FinQABenchmark:** Loads FinQA data, prompts the model with a financial QA instruction, then evaluates accuracy with tolerance for numeric answers (replacing the prior need for an LLM judge with code logic for now).
* **FPBBenchmark:** Loads Financial Phrase Bank data, prompts for sentiment classification with explanation, then parses model outputs to compute accuracy and F1.
* **FOMCBenchmark:** Loads FOMC sentences, prompts for hawkish/dovish/neutral classification, then evaluates classification metrics.

By following the structure of existing Evalchemy tasks (like the AlpacaEval example) and translating FLaME’s logic into this format, we ensure a **comprehensive integration**. This allows us to run `evalchemy` to benchmark models on these financial NLP tasks alongside other benchmarks, benefiting from Evalchemy’s features (such as easy model swapping and standardized result outputs). Each step above can be iteratively implemented and verified, resulting in a clear, maintainable adaptation of FLaME’s evaluation tasks into the Evalchemy harness.
