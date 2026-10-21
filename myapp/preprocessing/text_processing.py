"""
Text pre-processing for the IRWA fashion-products search engine (Part 1).

Everything that turns raw product records into index-ready terms lives here,
so Parts 2-4 (indexing, ranking, web app) can import the *same* functions and
process queries exactly like documents.

Main entry points
-----------------
- build_terms(text, stem=True)      -> list[str]   one text field -> terms
- normalize_entity(text)            -> str         brand / seller -> one exact token
- repair_brand(brand, url)          -> str         recover truncated brand names from the URL
- flatten_product_details(details)  -> dict        list of {k: v} -> {k: v}
- details_to_text(details)          -> str         descriptive attributes as text
- parse_price / parse_discount / parse_rating      numeric fields -> float | None
- preprocess_document(record)       -> dict        one raw record -> processed record
- preprocess_corpus(records)        -> pd.DataFrame
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def _load_stopwords() -> set:
    """Load the NLTK English stop-word list, downloading it on first use."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


STOP_WORDS: set = _load_stopwords()
STEMMER = SnowballStemmer("english")

# Tokens: runs of letters/digits. Everything else (punctuation, symbols) is a separator.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Hyphen or apostrophe *inside* a word: "t-shirt" -> "tshirt", "men's" -> "mens".
_INNER_JOIN_RE = re.compile(r"(?<=[a-z0-9])[-'’](?=[a-z0-9])")

# product_details keys that are identifiers or logistics, not product descriptors.
# They add noise terms ("1005combo2", "india", "machine wash") that never match a fashion query.
DETAILS_EXCLUDED_KEYS = {
    "style code", "model name", "fabric care", "care instructions", "sales package",
    "pack of", "number of contents in sales package", "size", "country of origin",
    "domestic warranty", "reversible", "",
}


# ---------------------------------------------------------------------------
# Text fields
# ---------------------------------------------------------------------------

def normalize_text(text: Optional[str]) -> str:
    """
    Normalise raw text before tokenisation.

    - HTML entities are decoded ("&amp;" -> "&").
    - Unicode is normalised (NFKD) and accents are dropped, so "café" == "cafe".
    - Non-breaking spaces and other odd whitespace become plain spaces.
    - Everything is lower-cased.
    - Hyphens/apostrophes inside words are removed, so "T-Shirt" and "tshirt" become the same term.
    """
    if not text:
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("\xa0", " ")
    text = _INNER_JOIN_RE.sub("", text)
    return text


def tokenize(text: str) -> List[str]:
    """Split normalised text into alphanumeric tokens (punctuation is dropped here)."""
    return _TOKEN_RE.findall(text)


def build_terms(text: Optional[str], stem: bool = True, keep_numbers: bool = False) -> List[str]:
    """
    Turn one text field (or a query) into a list of index terms.

    Steps: normalise -> tokenise (removes punctuation) -> drop stop words
    -> drop pure numbers (optional) -> drop 1-character tokens -> stem (optional).

    :param text: raw text (title, description, query, ...)
    :param stem: apply the Snowball (Porter2) stemmer
    :param keep_numbers: keep purely numeric tokens such as "100" or "32"
    :return: list of terms, in original order (duplicates kept, needed for tf later)
    """
    terms = []
    for tok in tokenize(normalize_text(text)):
        if tok in STOP_WORDS:
            continue
        if not keep_numbers and tok.isdigit():
            continue
        if len(tok) < 2:
            continue
        terms.append(STEMMER.stem(tok) if stem else tok)
    return terms


def normalize_entity(text: Optional[str]) -> str:
    """
    Normalise a brand or seller name into ONE exact-match token.

    Names are proper nouns: stemming or splitting them would create false matches
    ("Levi's" vs "level"), so we only lower-case and remove spaces/punctuation:
    "Shyam Enterprises" -> "shyamenterprises", "Levi's" -> "levis".
    """
    return "".join(tokenize(normalize_text(text)))


def repair_brand(brand: Optional[str], url: Optional[str]) -> str:
    """
    Recover the full brand name, which is truncated in the dataset ("REEB", "Pu", "ECKO Unl").

    The product URL starts with a slug that begins with the full brand name
    (".../reebok-full-sleeve-solid-men-sweatshirt/p/..."). We join slug words one by one
    until they start with the truncated brand: "reeb" -> "reebok", "eckounl" -> "ecko unltd".
    If no prefix matches (or there is no brand), the original value is returned unchanged.
    """
    if not brand:
        return ""
    b = "".join(tokenize(normalize_text(brand)))
    if not b or not url:
        return brand
    path = urlparse(url).path.strip("/")
    slug_words = path.split("/")[0].split("-") if path else []
    acc = ""
    for i, word in enumerate(slug_words[:6]):
        acc += word
        if acc.startswith(b):
            return " ".join(slug_words[: i + 1])
    return brand


# ---------------------------------------------------------------------------
# product_details
# ---------------------------------------------------------------------------

def flatten_product_details(details: Any) -> Dict[str, str]:
    """Convert the raw list of one-key dicts into a single dict {attribute: value}."""
    flat: Dict[str, str] = {}
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                for k, v in item.items():
                    flat[str(k).strip()] = str(v).strip()
    elif isinstance(details, dict):
        flat = {str(k).strip(): str(v).strip() for k, v in details.items()}
    return flat


def details_to_text(details: Dict[str, str]) -> str:
    """
    Keep only descriptive attributes (fabric, pattern, fit, sleeve, colour, ideal for...)
    and return them as plain text ready for build_terms.

    Yes/No attributes become the attribute name when "Yes" ("Hooded: Yes" -> "hooded")
    and are dropped when "No", because a bare "yes" carries no meaning.
    """
    parts = []
    for key, value in details.items():
        k = key.lower().strip()
        if k in DETAILS_EXCLUDED_KEYS:
            continue
        v = value.lower().strip()
        if v in {"yes", "no", "na", "n/a", "none", ""}:
            if v == "yes":
                parts.append(key)
            continue
        parts.append(value)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Numeric / boolean fields (kept as typed metadata, NOT as index terms)
# ---------------------------------------------------------------------------

def parse_price(value: Any) -> Optional[float]:
    """'2,999' -> 2999.0 ; '' / None / garbage -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    v = str(value).replace(",", "").strip()
    try:
        return float(v) if v else None
    except ValueError:
        return None


def parse_discount(value: Any) -> Optional[float]:
    """'69% off' -> 69.0 (percentage) ; '' / None -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
    return float(m.group(1)) if m else None


def parse_rating(value: Any) -> Optional[float]:
    """'3.9' -> 3.9 ; '' / None -> None. Ratings outside [0, 5] are treated as missing."""
    r = parse_price(value)
    return r if r is not None and 0 <= r <= 5 else None


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def preprocess_document(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process one raw product record.

    Output keeps:
      * the original display fields required by the statement (pid, title, description,
        brand, category, sub_category, product_details, seller, out_of_stock, prices,
        discount, average_rating, url) - untouched, to show them in results;
      * one list of terms per searchable field (field-based index for Parts 2-3);
      * all_terms: the concatenation of every field (single-field baseline index);
      * typed numeric metadata for filtering / ranking.
    """
    details = flatten_product_details(rec.get("product_details"))
    brand_full = repair_brand(rec.get("brand"), rec.get("url"))
    brand_tok = normalize_entity(brand_full)
    seller_tok = normalize_entity(rec.get("seller"))

    title_terms = build_terms(rec.get("title"))
    description_terms = build_terms(rec.get("description"))
    category_terms = build_terms(rec.get("category"))
    sub_category_terms = build_terms(rec.get("sub_category"))
    details_terms = build_terms(details_to_text(details))
    brand_terms = [brand_tok] if brand_tok else []

    return {
        # identifier + display fields (unchanged)
        "pid": rec.get("pid"),
        "title": rec.get("title"),
        "description": rec.get("description") or "",
        "brand": rec.get("brand") or "",
        "brand_full": brand_full,          # repaired name (see repair_brand)
        "category": rec.get("category") or "",
        "sub_category": rec.get("sub_category") or "",
        "product_details": details,
        "seller": rec.get("seller") or "",
        "url": rec.get("url"),
        # searchable fields
        "title_terms": title_terms,
        "description_terms": description_terms,
        "brand_terms": brand_terms,
        "category_terms": category_terms,
        "sub_category_terms": sub_category_terms,
        "details_terms": details_terms,
        "seller_token": seller_tok,  # stored, not merged into all_terms (see notebook)
        "all_terms": (title_terms + description_terms + brand_terms
                      + category_terms + sub_category_terms + details_terms),
        # typed metadata
        "out_of_stock": bool(rec.get("out_of_stock", False)),
        "selling_price": parse_price(rec.get("selling_price")),
        "actual_price": parse_price(rec.get("actual_price")),
        "discount": parse_discount(rec.get("discount")),
        "average_rating": parse_rating(rec.get("average_rating")),
    }


def preprocess_corpus(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Process every record and return a DataFrame indexed by pid."""
    df = pd.DataFrame([preprocess_document(r) for r in records])
    return df.set_index("pid", drop=False)


def preprocess_query(query: str) -> List[str]:
    """Process a query with exactly the same pipeline as document text fields."""
    return build_terms(query)
