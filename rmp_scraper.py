from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE = Path(__file__).resolve().parent
CACHE_PATH = BASE / "rmp_cache.csv"

RMP_COLUMNS = [
    "instructor_clean",
    "rmp_rating",
    "rmp_difficulty",
    "rmp_would_take_again",
    "rmp_tags",
    "rmp_snippet",
    "source_url",
    "rmp_status",
    "rmp_page_status",
]


def _real_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def _normalize_name(name: str) -> str:
    """
    Normalize professor names enough for cache matching.
    Keeps 'Last, First' format stable but removes extra whitespace.
    """
    text = _real_text(name)
    text = " ".join(text.replace("(Primary)", "").split())
    return text.strip(" ,;")


def create_empty_cache() -> pd.DataFrame:
    df = pd.DataFrame(columns=RMP_COLUMNS)
    df.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
    return df


def load_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return create_empty_cache()

    try:
        cache = pd.read_csv(CACHE_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        cache = pd.read_csv(CACHE_PATH, encoding="cp1252")

    for col in RMP_COLUMNS:
        if col not in cache.columns:
            cache[col] = ""

    cache["instructor_clean"] = cache["instructor_clean"].apply(_normalize_name)

    return cache[RMP_COLUMNS]


def save_cache(cache: pd.DataFrame) -> None:
    for col in RMP_COLUMNS:
        if col not in cache.columns:
            cache[col] = ""

    cache = cache[RMP_COLUMNS].drop_duplicates(subset=["instructor_clean"], keep="last")
    cache.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")


def enrich_with_rmp(courses: pd.DataFrame, use_web=False, max_lookups=15) -> pd.DataFrame:
    """
    Cache-first RMP enrichment.

    For now:
    - reads rmp_cache.csv
    - merges ratings/tags/snippets onto courses
    - never crashes if cache is blank
    - live web scraping is intentionally deferred until cache works
    """
    courses = courses.copy()

    if "instructor_clean" not in courses.columns:
        courses["instructor_clean"] = ""

    courses["instructor_clean"] = courses["instructor_clean"].apply(_normalize_name)

    for col in [
        "rmp_rating",
        "rmp_difficulty",
        "rmp_would_take_again",
        "rmp_tags",
        "rmp_snippet",
        "source_url",
        "rmp_status",
        "rmp_page_status",
    ]:
        if col not in courses.columns:
            courses[col] = ""

    cache = load_cache()

    if cache.empty:
        courses["rmp_status"] = "no cache"
        return courses

    merged = courses.merge(
        cache,
        on="instructor_clean",
        how="left",
        suffixes=("", "_cache"),
    )

    for col in [
        "rmp_rating",
        "rmp_difficulty",
        "rmp_would_take_again",
        "rmp_tags",
        "rmp_snippet",
        "source_url",
        "rmp_status",
        "rmp_page_status",
    ]:
        cache_col = f"{col}_cache"

        if cache_col in merged.columns:
            merged[col] = merged[cache_col].combine_first(merged[col])
            merged = merged.drop(columns=[cache_col])

    merged["rmp_status"] = merged["rmp_status"].fillna("")
    merged.loc[
        merged["rmp_rating"].astype(str).str.strip().isin(["", "nan", "None"]),
        "rmp_status",
    ] = "not in cache"

    if use_web:
        # We will add live scraping here only after cache enrichment is stable.
        missing = merged["rmp_status"].eq("not in cache")
        merged.loc[missing, "rmp_status"] = "web lookup not yet enabled"

    return merged