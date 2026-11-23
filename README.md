# Effects of Hybrid Retrieval and Recursive Chunking on RAG Performance

[![Course](https://img.shields.io/badge/Course-YZM5109%20LLM-blue.svg)](#overview)
[![Environment](https://img.shields.io/badge/Environment-Kaggle%20Notebook-20BEFF.svg)](#how-to-run-in-kaggle)
[![Dataset](https://img.shields.io/badge/Dataset-SQuAD%20v2-orange.svg)](#dataset-and-data-split)
[![Model](https://img.shields.io/badge/Generator-Qwen2.5--1.5B--Instruct-green.svg)](#models-and-settings)
[![Python](https://img.shields.io/badge/Python-3-yellow.svg)](#how-to-run-in-kaggle)

This is a small experiment for **YZM5109 - Large Language Models (Assignment II, Option C)**.

We test two ideas and compare them with a **Naive RAG** baseline:

- **Hybrid Retrieval** (Dense search + BM25, joined with RRF)
- **Recursive Chunking** (split by paragraph, then by sentence)

We check retrieval quality, answer quality, faithfulness, abstention (saying "I don't know"), and run time.

---

## Table of Contents

- [Overview](#overview)
- [The Four Configurations](#the-four-configurations)
- [Dataset and Data Split](#dataset-and-data-split)
- [Models and Settings](#models-and-settings)
- [Metrics](#metrics)
- [Results](#results)
- [How to Run in Kaggle](#how-to-run-in-kaggle)
- [Project Files](#project-files)
- [Implementation Notes](#implementation-notes)
- [References](#references)

---

## Overview

A language model can give wrong answers when the answer is not in the context it sees. **Retrieval-Augmented Generation (RAG)** helps with this. It first finds useful passages in a document store. Then it puts them in the prompt.

In this project we compare four systems, **C1 to C4**. We use the same test set for all of them. We do **not** fine-tune the generator model.

---

## The Four Configurations

| Name | Chunking | Chunks | Retriever | Top-k to LLM |
| :--- | :--- | :---: | :--- | :---: |
| **C1_Naive** | Fixed size: 350 characters, stride 300 | 5,262 | Dense (`all-MiniLM-L6-v2` + FAISS) | 3 |
| **C2_Hybrid** | Fixed size: 350 characters, stride 300 | 5,262 | Dense (top-15) + BM25 (top-15), joined with RRF (`k = 60`) | 3 |
| **C3_Recursive** | Recursive: paragraphs, then sentences, up to 750 characters | 3,429 | Dense (`all-MiniLM-L6-v2` + FAISS) | 3 |
| **C4_Combined** | Recursive: paragraphs, then sentences, up to 750 characters | 3,429 | Dense (top-15) + BM25 (top-15), joined with RRF (`k = 60`) | 3 |

In short:

- **C1** is the baseline.
- **C2** changes only the retriever.
- **C3** changes only the chunking.
- **C4** uses both changes.

---

## Dataset and Data Split

- **Dataset:** `rajpurkar/squad_v2` (`train` split).
- **Corpus:** We group the unique paragraphs by Wikipedia title. We take the first **50** articles that have at least **15** paragraphs. We join the paragraphs of each article with `\n\n`. This gives 50 documents (`doc_0` to `doc_49`). We also save the start position of each paragraph.
- **Question split (`seed = 42`):** We shuffle answerable and unanswerable questions separately.
  - **Dev set:** 50 questions (40 answerable, 10 unanswerable). We keep it aside and do not use it for tuning.
  - **Test set:** 120 questions (100 answerable, 20 unanswerable). All four systems use the same test set.
- **Gold evidence:** For an answerable question, a retrieved chunk counts as a hit when both of these are true:
  1. It comes from the same document as the question.
  2. Its character span `[start, end)` overlaps the gold answer span.

---

## Models and Settings

| Part | Value |
| :--- | :--- |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (L2-normalized, `faiss.IndexFlatIP`) |
| Lexical retriever | `BM25Okapi` (lowercase words, punctuation removed with regex) |
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` (Hugging Face `pipeline`, `torch.float16`, `device_map="auto"`) |
| Decoding | Greedy (`do_sample=False`), `max_new_tokens=30` |
| Seeds | `42` for `random`, `numpy`, and `torch` |

**System prompt:**

```text
Answer concisely using ONLY the context. If not in context, output ONLY: UNANSWERABLE
```

**User prompt:**

```text
Context:
{ctx}

Question: {question}
```

---

## Metrics

| Metric | Used on | Meaning |
| :--- | :--- | :--- |
| **Recall@3** | 100 answerable | Share of questions where at least one of the top-3 chunks overlaps the gold answer span. |
| **MRR** | 100 answerable | Mean of `1 / rank` of the first correct chunk in the top 3. It is `0.0` if there is none. |
| **EM** | 100 answerable | Exact match after lowercasing, removing punctuation, and splitting into words. |
| **F1** | 100 answerable | Word-level F1 between the predicted answer and the gold answer. |
| **Faithfulness** | Answerable, not abstained | Share of answer words that also appear in the top-3 chunks. Abstained answers are left out. |
| **FalseAbst(ans)** | 100 answerable | Share of answerable questions where the model abstains by mistake. |
| **Abstention(unans)** | 20 unanswerable | Share of unanswerable questions where the model abstains correctly. |
| **Time (s)** | 120 test questions | Total time for retrieval and generation. |

The model counts as "abstaining" when its output has one of these words: `unanswerable`, `not mentioned`, `no information`, `cannot`.

---

## Results

Test set: 100 answerable and 20 unanswerable questions.

| Condition | Chunks | Recall@3 (%) | MRR (%) | EM (%) | F1 (%) | Faithfulness (%) | FalseAbst(ans) (%) | Abstention(unans) (%) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1_Naive** | 5,262 | 73.00 | 65.17 | 20.00 | 38.80 | 86.94 | 7.00 | 40.00 | 91.90 |
| **C2_Hybrid** | 5,262 | 80.00 | 70.50 | 16.00 | 36.04 | 88.55 | 4.00 | 35.00 | 101.40 |
| **C3_Recursive** | 3,429 | 79.00 | 68.67 | 18.00 | 36.73 | 86.97 | 4.00 | 20.00 | 106.70 |
| **C4_Combined** | 3,429 | **85.00** | **77.17** | 20.00 | 37.95 | **89.10** | **1.00** | 35.00 | 108.40 |

Main points:

- **C4** has the best Recall@3, MRR, and Faithfulness. It also has the lowest false abstention rate.
- Both changes alone (C2 and C3) improve retrieval over C1.
- Better retrieval does not always give better answers. C1 still has the best F1.
- C1 is the fastest. C4 is the slowest.

---

## How to Run in Kaggle

This project was run in a **Kaggle Notebook**.

### 1. Create a new notebook

1. Sign in to [Kaggle](https://www.kaggle.com/) and click **Code -> New Notebook**.
2. Open the **Settings** panel on the right side:
   - Turn on **Internet**. The notebook needs it to download `datasets`, `faiss-cpu`, `rank_bm25`, `sentence-transformers`, and the model weights.
   - Choose a GPU, for example **GPU T4 x2**.

### 2. Add the code

Upload `llm-assignment-2.ipynb` (**File -> Upload Notebook**), or copy the code cells into a new notebook. The first cell installs the extra packages:

```python
!pip install -q faiss-cpu rank_bm25 datasets sentence-transformers
```

### 3. Run all cells

Click **Run All**, or run cells `1` to `7` in order. The notebook will:

1. Load `rajpurkar/squad_v2` and build the 50-document corpus, the dev set, and the test set.
2. Build the basic chunks (5,262) and the recursive chunks (3,429), with FAISS and BM25 indexes.
3. Load `Qwen/Qwen2.5-1.5B-Instruct` and test C1, C2, C3, and C4 on all 120 test questions.
4. Save the predictions, config, prompts, summary metrics, a plot, and a zip file.

### 4. Download the outputs

All files are saved in `/kaggle/working/rag_outputs/`. They are also zipped to `/kaggle/working/rag_outputs.zip`. Download the zip from the `FileLink` output or from the **Output** tab on the right side.

---

## Project Files

```text
.
├── README.md                     # This file
├── llm-assignment-2.ipynb        # Main notebook (C1-C4 pipeline and evaluation)
├── report.pdf                    # IEEE two-column report
└── rag_outputs/                  # Created when you run the notebook
    ├── config_and_prompts.json   # Settings, model IDs, and prompts
    ├── C1_Naive.csv / .json      # Predictions and scores for C1
    ├── C2_Hybrid.csv / .json     # Predictions and scores for C2
    ├── C3_Recursive.csv / .json  # Predictions and scores for C3
    ├── C4_Combined.csv / .json   # Predictions and scores for C4
    ├── summary.csv / .json       # Summary metrics for all four conditions
    └── fig1_scores.png           # Chart of retrieval and generation scores
```

---

## Implementation Notes

- **Punctuation is removed.** If we split only on spaces, punctuation stays on the words and BM25 matches worse. The `words()` function lowercases the text and removes punctuation first.
- **Articles are kept.** We do not remove `a`, `an`, and `the`. So a prediction without the first article can get `EM = 0.0`, even if the other words are correct.
- **Evidence uses character spans.** We check if a chunk overlaps the gold span in the same document. We do not only check if the answer text is inside the chunk.
- **Recursive chunking.** A paragraph up to 750 characters stays whole. A longer paragraph is split into sentences, and the sentences are packed up to 750 characters. A sentence that is still too long is cut at 750 characters. Chunks with 30 characters or fewer are dropped.
- **Faithfulness skips abstentions.** If the model abstains, that question is not counted. This way, abstentions do not raise the score.

---

## References

1. P. Rajpurkar, R. Jia, and P. Liang, "Know What You Don't Know: Unanswerable Questions for SQuAD," *Proc. 56th Annu. Meeting Assoc. Comput. Linguist. (ACL)*, pp. 784-792, 2018. [Dataset](https://huggingface.co/datasets/rajpurkar/squad_v2)
2. Qwen Team, "Qwen2.5-1.5B-Instruct." [Model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
