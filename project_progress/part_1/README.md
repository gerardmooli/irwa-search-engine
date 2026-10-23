# Part 1 — Text Processing and Exploratory Data Analysis

## What is here

| File | Content |
|---|---|
| `part_1.ipynb` | PART 1: Data preparation (points 0–4) and PART 2: Exploratory Data Analysis |
| `IRWA-2026-uXXXXXX-uXXXXXX-uXXXXXX-part-1.pdf` | The report (add it here when it is ready) |

The pre-processing functions are in the notebook, in section 1 of PART 1.

## How to run (step by step)

1. **Get the data.** Copy `fashion_products_dataset.json` and `validation_labels.csv` into the `data/` folder at the repository root.
2. **Create and activate the environment** (from the repository root):
   ```bash
   python -m venv irwa_venv
   # Windows:      irwa_venv\Scripts\activate
   # macOS/Linux:  source irwa_venv/bin/activate
   pip install -r requirements.txt
   pip install matplotlib seaborn wordcloud spacy notebook
   ```
3. **Download the language resources** (only once):
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   python -m spacy download en_core_web_sm
   ```
   The spaCy model is only used in the named-entity section.
4. **Run the notebook:**
   ```bash
   jupyter notebook project_progress/part_1/part_1.ipynb
   ```
   Then *Kernel → Restart & Run All*. It takes about 1 minute.

## Outputs

- `data/processed_corpus.pkl` — one row per product, indexed by `pid`. Part 2 can load this file.
- `project_progress/part_1/figures/*.png` — figures for the report. This folder is created when the notebook runs and is not committed.

## Columns of the processed corpus

| Column | Meaning |
|---|---|
| `pid`, `title`, `description`, `brand`, `category`, `sub_category`, `product_details`, `seller`, `url` | Original values, kept for display (`product_details` flattened into a dict) |
| `brand_full` | Brand name repaired from the URL (the dataset truncates brands: "REEB" → "reebok") |
| `title_terms`, `description_terms`, `details_terms`, `category_terms`, `sub_category_terms`, `brand_terms` | Processed terms per field |
| `all_terms` | Concatenation of the term fields above |
| `seller_token` | Normalised seller name (stored, not included in `all_terms`) |
| `out_of_stock`, `selling_price`, `actual_price`, `discount`, `average_rating` | Typed metadata (bool / float, `NaN` when missing) |

## Pre-processing pipeline (summary)

lower-case + HTML/accents/whitespace normalisation → join inner hyphens/apostrophes (`T-Shirt` → `tshirt`)
→ tokenisation (words and punctuation as separate tokens) → remove punctuation → NLTK English stop words → drop numbers and 1-character tokens
→ Snowball stemming. Brands/sellers: one exact token, no stemming. Numeric fields: parsed, not indexed as text.
