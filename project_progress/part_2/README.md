# Part 2 — Indexing and Evaluation

## What is here

| File | Content |
|---|---|
| `part_2.ipynb` | PART 1: Indexing (inverted index, test queries, TF-IDF ranking) and PART 2: Evaluation (metrics, `validation_labels.csv`, test query judgments) |
| `test_queries_judgments.csv` | Top 20 results of each test query, with the binary relevance label given by the team |
| `Evaluation_stud.ipynb`, `test_predictions.csv` | Practical lab on evaluation metrics (the metric functions of `part_2.ipynb` start from it) |
| `IRWA-2026-uXXXXXX-uXXXXXX-uXXXXXX-part-2.pdf` | The report (add it here when it is ready) |

## How to run (step by step)

1. **Run Part 1 first.** `part_2.ipynb` loads `data/processed_corpus.pkl`, which is written by `project_progress/part_1/part_1.ipynb`. `data/validation_labels.csv` must also be in `data/`.
2. **Activate the environment** (from the repository root):
   ```bash
   irwa_venv\Scripts\activate          # Windows
   source irwa_venv/bin/activate       # macOS/Linux
   ```
3. **Run the notebook:**
   ```bash
   jupyter notebook project_progress/part_2/part_2.ipynb
   ```
   Then *Kernel → Restart & Run All*. It takes less than a minute.

## Test query judgments

The first run creates `test_queries_judgments.csv` with the top 20 results of each test query and an empty `label` column.
Fill it with 1 (relevant) or 0 (not relevant) and run the last cell again to get the evaluation of the test queries.
The file is not overwritten once it exists.

## Search engine (summary)

- Index: inverted index `{term: [pid, ...]}` built on `all_terms` (all the processed fields of Part 1).
- Queries: same pre-processing as the documents; conjunctive (AND): a document must contain every query term.
- Ranking: TF-IDF cosine similarity. Document tf = term count / L2 norm of the document counts; idf = log(N / df).

## Evaluation (summary)

- Metrics: Precision@K, Recall@K, Average Precision@K, F1-Score@K, MAP, MRR and NDCG, for K = 5 and 10.
- Only the judged documents are ranked. Judged documents that do not contain all the query terms are not retrieved,
  but they still count as relevant for Recall and for the ideal ranking of NDCG.
