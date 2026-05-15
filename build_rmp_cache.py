from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE = Path(__file__).resolve().parent
COURSES_CSV = BASE / "clean_schedule_sections.csv"
CACHE_CSV = BASE / "rmp_cache.csv"

SCHOOL_ID = "355"  # Georgetown University on RMP
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
}

CACHE_COLUMNS = [
    "instructor_clean",
    "rmp_rating",
    "rmp_tags",
    "rmp_snippet",
    "source_url",
    "rmp_status",
]


def real_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def normalize_name(name: str) -> str:
    text = real_text(name)
    text = text.replace("(Primary)", "")
    text = " ".join(text.split())
    return text.strip(" ,;")


def split_professors(raw: str) -> list[str]:
    text = normalize_name(raw)
    if not text:
        return []

    parts = re.split(r";|\n", text)
    return [normalize_name(p) for p in parts if normalize_name(p)]


def read_existing_cache() -> pd.DataFrame:
    if not CACHE_CSV.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)

    cache = pd.read_csv(CACHE_CSV, encoding="utf-8-sig")

    for col in CACHE_COLUMNS:
        if col not in cache.columns:
            cache[col] = ""

    cache["instructor_clean"] = cache["instructor_clean"].apply(normalize_name)
    return cache[CACHE_COLUMNS]


def save_cache(cache: pd.DataFrame) -> None:
    for col in CACHE_COLUMNS:
        if col not in cache.columns:
            cache[col] = ""

    cache = cache[CACHE_COLUMNS]
    cache = cache.drop_duplicates(subset=["instructor_clean"], keep="last")
    cache.to_csv(CACHE_CSV, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def professor_names_from_courses() -> list[str]:
    courses = pd.read_csv(COURSES_CSV, encoding="utf-8-sig")

    name_col = "instructor_clean" if "instructor_clean" in courses.columns else "instructor_raw"
    if name_col not in courses.columns:
        raise ValueError("Could not find instructor_clean or instructor_raw in course CSV.")

    names = set()

    for raw in courses[name_col].dropna():
        for prof in split_professors(raw):
            if prof:
                names.add(prof)

    return sorted(names)


def name_variants(name: str) -> list[str]:
    """
    RMP pages usually display First Last, while our CSV often has Last, First.
    Try both.
    """
    name = normalize_name(name)
    variants = [name]

    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        if first and last:
            variants.append(f"{first} {last}")

    return list(dict.fromkeys(variants))


def search_rmp_professor(name: str) -> dict:
    """
    Conservative HTML search. This does not bypass logins/CAPTCHA.
    If RMP blocks or changes markup, returns not found.
    """
    for variant in name_variants(name):
        url = f"https://www.ratemyprofessors.com/search/professors/{SCHOOL_ID}?q={quote_plus(variant)}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except requests.RequestException:
            continue

        if resp.status_code != 200:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        # Basic not-found detection
        if "No professors found" in text or "Don't see the professor" in text and variant.lower() not in text.lower():
            continue

        # Try to pull a rating near the professor name.
        rating = ""
        tags = ""
        snippet = ""

        professor_link = None
        for a in soup.find_all("a", href=True):
            if "/professor/" in a["href"]:
                professor_link = a["href"]
                break

        source_url = ""
        if professor_link:
            source_url = "https://www.ratemyprofessors.com" + professor_link

        # RMP search pages often include text like QUALITY 4.5 near result cards.
        rating_match = re.search(r"QUALITY\s+(\d(?:\.\d)?)", text, flags=re.IGNORECASE)
        if rating_match:
            rating = rating_match.group(1)

        if source_url:
            snippet = f"Matched RMP search result for {variant}."

        if rating or source_url:
            return {
                "instructor_clean": name,
                "rmp_rating": rating,
                "rmp_tags": tags,
                "rmp_snippet": snippet,
                "source_url": source_url,
                "rmp_status": "scraped_search",
            }

    return {
        "instructor_clean": name,
        "rmp_rating": "",
        "rmp_tags": "",
        "rmp_snippet": "",
        "source_url": "",
        "rmp_status": "not found",
    }


def main():
    if not COURSES_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {COURSES_CSV}. Put clean_schedule_sections.csv in this folder."
        )

    cache = read_existing_cache()
    cached_names = set(cache["instructor_clean"].dropna().astype(str))

    professors = professor_names_from_courses()
    missing = [p for p in professors if p not in cached_names]

    print(f"Found {len(professors)} unique professor names in course CSV.")
    print(f"{len(cached_names)} already in cache.")
    print(f"{len(missing)} missing from cache.")
    print()

    new_rows = []

    for i, professor in enumerate(missing, start=1):
        print(f"[{i}/{len(missing)}] Searching {professor}...")
        row = search_rmp_professor(professor)
        print(f"    -> {row['rmp_status']} {row['rmp_rating']} {row['source_url']}")
        new_rows.append(row)

        # Be polite and avoid hammering the site.
        time.sleep(2.5)

        # Save as we go, so progress is not lost.
        combined = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        save_cache(combined)

    final_cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    save_cache(final_cache)

    print()
    print(f"Done. Wrote {len(final_cache)} rows to {CACHE_CSV}")


if __name__ == "__main__":
    main()