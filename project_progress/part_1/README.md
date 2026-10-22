# Part 1 — Text Processing and Exploratory Data Analysis

## What is here

| File | Content |
|---|---|
| `part_1.ipynb` | The whole Part 1: pre-processing decisions (statement points 1–4) and the exploratory data analysis |
| `../../myapp/preprocessing/text_processing.py` | Reusable, documented pre-processing functions (also used for queries in Parts 2–4) |
| `figures/` | All figures produced by the notebook (used in the report) |
| `IRWA-2026-uXXXXXX-uXXXXXX-uXXXXXX-part-1.pdf` | The report (add it here when it is ready) |

## How to run (step by step)

1. **Get the data.** Copy `fashion_products_dataset.json` and `validation_labels.csv` into the `data/` folder at the repository root.
2. **Create and activate the environment** (from the repository root):
   ```bash
   python -m venv irwa_venv
   # Windows:      irwa_venv\Scripts\activate
   # macOS/Linux:  source irwa_venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Download the language resources** (only once):
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   python -m spacy download en_core_web_sm      # optional, only for the NER section
   ```
4. **Run the notebook:**
   ```bash
   jupyter notebook project_progress/part_1/part_1.ipynb
   ```
   Then *Kernel → Restart & Run All*. It takes about 1 minute.

## Outputs

- `data/processed_corpus.pkl` — the processed corpus (one row per product, indexed by `pid`). **Part 2 loads this file.**
- `project_progress/part_1/figures/*.png` — figures for the report.

## Using the pre-processing in other code

```python
from myapp.preprocessing.text_processing import preprocess_corpus, preprocess_query

corpus = preprocess_corpus(records)                        # list of raw JSON records -> DataFrame
preprocess_query("women full sleeve sweatshirt cotton")    # ['women', 'full', 'sleev', 'sweatshirt', 'cotton']
```

Columns of the processed corpus:

| Column | Meaning |
|---|---|
| `pid`, `title`, `description`, `brand`, `category`, `sub_category`, `product_details`, `seller`, `url` | Original values, kept for display (`product_details` flattened into a dict) |
| `brand_full` | Brand name repaired from the URL (the dataset truncates brands: "REEB" → "reebok") |
| `title_terms`, `description_terms`, `details_terms`, `category_terms`, `sub_category_terms`, `brand_terms` | Processed terms per field (for a field-based index) |
| `all_terms` | Concatenation of all the term fields above (single-field index) |
| `seller_token` | Normalised seller name (stored, not indexed in `all_terms`) |
| `out_of_stock`, `selling_price`, `actual_price`, `discount`, `average_rating` | Typed metadata (bool / float, `NaN` when missing) for filtering and ranking |

## Pre-processing pipeline (summary)

lower-case + HTML/accents/whitespace normalisation → join inner hyphens/apostrophes (`T-Shirt` → `tshirt`)
→ tokenisation (words and punctuation as separate tokens) → remove punctuation → NLTK English stop words → drop numbers and 1-character tokens
→ Snowball stemming. Brands/sellers: one exact token, no stemming. Numeric fields: parsed, not indexed as text.
