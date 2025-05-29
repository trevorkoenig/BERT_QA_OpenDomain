
# BERT-Based Open-Domain Question Answering Pipeline

This project implements a full open-domain question answering (QA) system using BERT fine-tuned on the SQuAD v2.0 dataset. The system processes user questions, extracts relevant context from Wikipedia, and identifies the most likely answer spans using a trained deep learning model. It also benchmarks BERT responses against ChatGPT outputs for evaluation.

## Overview

- Fine-tunes a BERT model for extractive QA using Hugging Face Transformers.
- Uses Wikipedia as the external knowledge base for real-world questions.
- Applies a custom scoring system for span selection from long documents.
- Compares BERT predictions against ChatGPT for performance analysis.
- Designed to run in Google Colab with persistent model saving to Drive.

## Installation

```bash
pip install transformers datasets evaluate
pip install --upgrade transformers
pip install wikipedia wikipedia-api spacy openai
python -m spacy download en_core_web_sm
```

## Training Pipeline

### 1. Load and Prepare Dataset
- Dataset: [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/)
- Preprocesses each example by tokenizing and converting character-based answer spans into token-based positions.
- Handles unanswerable cases by assigning (0, 0) span positions.

### 2. Define Model
- Model: `bert-base-uncased`
- Tokenizer and architecture loaded via Hugging Face.

### 3. Training Configuration
- Optimized with:
  - Learning rate: 2e-5
  - Epochs: 3
  - Batch size: 16
  - Weight decay: 0.01
- Evaluation and saving happen at the end of each epoch.

### 4. Metrics
- Computes simple token-level accuracy for start and end position predictions.

### 5. Save Trained Model
- Persistently saved to Google Drive under `/content/drive/MyDrive/NLP_BERT_Model`
- Any previous model in the path is overwritten.

## Inference and Evaluation Pipeline

After training, the model is used to extract answers from Wikipedia articles in response to a curated set of factoid questions.

### Inference Workflow

- Loads the trained BERT model and tokenizer from Drive.
- Uses `wikipedia` and `wikipedia-api` to retrieve article content relevant to a given question.
- Processes large article texts using a sliding window approach.
  - Chunks are passed through the model to predict start and end probabilities.
  - A custom scoring function selects the best answer span with a confidence score.
  - Results are wrapped in an `AnswerResult` class for traceability and future expansion.

### AnswerResult Class

Encapsulates:
- Answer text
- Confidence score
- Optional reference sentence (via spaCy parsing)

### Wikipedia Search and Span Extraction

- Top-N search results are retrieved using `wikipedia.search()`.
- The best span is extracted from each full article using `transformers` tokenization with context overflow control.
- A softmax is applied over all candidate spans to normalize and weight scores across sources.

### ChatGPT Comparison

- Each question is also submitted to OpenAI’s GPT (via API).
- Outputs from both BERT and ChatGPT are recorded.
- The results are written to a CSV file for manual review:
  ```
  | question | BERT_Ans | ChatGPT_Ans |
  ```

## Example Use Case

- 50+ general knowledge questions are evaluated.
- For each question, BERT attempts to extract a span from real Wikipedia articles.
- ChatGPT is used as a benchmark to assess span precision and information coverage.

## File Exports

- Final model is saved at:
  ```
  /content/drive/MyDrive/NLP_BERT_Model/
  ```
- Evaluation results are written to:
  ```
  /content/drive/MyDrive/CISC 688 - Intro to NLP/Project - CISC 688 - Intro to NLP/answeredQs.csv
  ```

## Dependencies

- Python 3.8+
- `transformers`, `datasets`, `evaluate`
- `spacy`, `wikipedia`, `wikipedia-api`
- `openai` (for ChatGPT integration)

## Notes

- Developed in Google Colab.
- Model is trained on extractive QA but applied to open-domain fact retrieval via Wikipedia.
- Long context processing and span scoring logic is custom-built for transparent evaluation.
