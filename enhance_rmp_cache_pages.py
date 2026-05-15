from __future__ import annotations

import csv
import html
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE = Path(__file__).resolve().parent
CACHE_CSV = BASE / "rmp_cache.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0 Safari/537.36"
    )
}

NEW_COLUMNS = [
    "rmp_difficulty",
    "rmp_would_take_again",
    "rmp_page_tags",
    "rmp_teaching_signal",
    "rmp_page_status",
]


def real_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s

def read_cache() -> pd.DataFrame:
    if not CACHE_CSV.exists():
        raise FileNotFoundError("Could not find rmp_cache.csv")

    df = pd.read_csv(CACHE_CSV, encoding="utf-8-sig")

    for col in NEW_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Important: keep enhanced columns as strings so pandas does not crash
    # when writing scraped text/numbers into previously numeric columns.
    for col in NEW_COLUMNS:
        df[col] = df[col].astype("object").fillna("")

    return df

# def read_cache() -> pd.DataFrame:
#     if not CACHE_CSV.exists():
#         raise FileNotFoundError("Could not find rmp_cache.csv")

#     df = pd.read_csv(CACHE_CSV, encoding="utf-8-sig")

#     for col in NEW_COLUMNS:
#         if col not in df.columns:
#             df[col] = ""

#     return df


def save_cache(df: pd.DataFrame) -> None:
    df.to_csv(CACHE_CSV, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def find_number(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def extract_tags(text: str) -> str:
    tags = []

    # Common JSON-ish patterns that appear on RMP pages.
    patterns = [
        r'"tagName"\s*:\s*"([^"]+)"',
        r'"legacyId"\s*:\s*\d+\s*,\s*"name"\s*:\s*"([^"]+)"',
        r'"teacherRatingTags"\s*:\s*\[(.*?)\]',
    ]

    for pattern in patterns[:2]:
        tags.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    # Clean and deduplicate.
    cleaned = []
    for tag in tags:
        tag = html.unescape(str(tag)).strip()
        tag = re.sub(r"\\u002F", "/", tag)
        tag = re.sub(r"\\+", "", tag)
        if tag and tag.lower() not in {"null", "none"}:
            cleaned.append(tag)

    # Keep order, remove duplicates.
    unique = []
    seen = set()
    for tag in cleaned:
        key = tag.lower()
        if key not in seen:
            unique.append(tag)
            seen.add(key)

    return ", ".join(unique[:8])


def extract_teaching_signal(rating: str, difficulty: str, tags: str, would_take_again: str) -> str:
    pieces = []

    try:
        r = float(rating)
        if r >= 4.5:
            pieces.append("strong professor rating")
        elif r >= 4.0:
            pieces.append("solid professor rating")
        elif r <= 2.5:
            pieces.append("lower professor rating")
    except Exception:
        pass

    try:
        d = float(difficulty)
        if d <= 2.5:
            pieces.append("students seem to find the course relatively manageable")
        elif d >= 4.0:
            pieces.append("students seem to find the course demanding")
        elif d:
            pieces.append("moderate difficulty signal")
    except Exception:
        pass

    if would_take_again:
        try:
            w = float(would_take_again)
            if w >= 80:
                pieces.append("many students would take the professor again")
            elif w <= 50:
                pieces.append("mixed would-take-again signal")
        except Exception:
            pass

    tag_text = tags.lower()
    if any(x in tag_text for x in ["inspirational", "amazing lectures", "clear grading", "gives good feedback", "respected"]):
        pieces.append("review tags suggest an engaging or supportive teaching style")
    if any(x in tag_text for x in ["tough grader", "lots of homework", "test heavy", "lecture heavy"]):
        pieces.append("review tags suggest a heavier workload or tougher grading")

    return "; ".join(pieces)


def scrape_professor_page(url: str, current_rating: str = "") -> dict:
    url = real_text(url)
    if not url:
        return {
            "rmp_difficulty": "",
            "rmp_would_take_again": "",
            "rmp_page_tags": "",
            "rmp_teaching_signal": "",
            "rmp_page_status": "no source_url",
        }

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
    except requests.RequestException as e:
        return {
            "rmp_difficulty": "",
            "rmp_would_take_again": "",
            "rmp_page_tags": "",
            "rmp_teaching_signal": "",
            "rmp_page_status": f"request failed: {type(e).__name__}",
        }

    if resp.status_code != 200:
        return {
            "rmp_difficulty": "",
            "rmp_would_take_again": "",
            "rmp_page_tags": "",
            "rmp_teaching_signal": "",
            "rmp_page_status": f"http {resp.status_code}",
        }

    soup = BeautifulSoup(resp.text, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    raw = resp.text + " " + page_text

    difficulty = find_number(
        [
            r'"avgDifficulty"\s*:\s*([0-9.]+)',
            r'"avgDifficultyRating"\s*:\s*([0-9.]+)',
            r'"difficultyRating"\s*:\s*([0-9.]+)',
            r'Difficulty\s*([0-9.]+)',
        ],
        raw,
    )

    would_take_again = find_number(
        [
            r'"wouldTakeAgainPercent"\s*:\s*([0-9.]+)',
            r'"wouldTakeAgain"\s*:\s*([0-9.]+)',
            r'Would Take Again\s*([0-9.]+)\s*%',
        ],
        raw,
    )

    tags = extract_tags(raw)

    teaching_signal = extract_teaching_signal(
        rating=current_rating,
        difficulty=difficulty,
        tags=tags,
        would_take_again=would_take_again,
    )

    status = "page_scraped"
    if not difficulty and not would_take_again and not tags:
        status = "page scraped but details not found"

    return {
        "rmp_difficulty": difficulty,
        "rmp_would_take_again": would_take_again,
        "rmp_page_tags": tags,
        "rmp_teaching_signal": teaching_signal,
        "rmp_page_status": status,
    }


def main():
    df = read_cache()

    has_url = df["source_url"].apply(real_text).astype(bool)

    # Only scrape rows with real RMP pages.
    candidates = df[has_url].copy()

    # Skip rows already enhanced.
    needs = candidates[
        candidates["rmp_page_status"].apply(real_text).eq("")
    ].copy()

    print(f"Rows in cache: {len(df)}")
    print(f"Rows with source_url: {len(candidates)}")
    print(f"Rows still needing page scrape: {len(needs)}")
    print()

    # First run limit. Increase later if it works.
    limit = 2600
    needs = needs.head(limit)

    for count, (idx, row) in enumerate(needs.iterrows(), start=1):
        name = real_text(row.get("instructor_clean"))
        url = real_text(row.get("source_url"))
        rating = real_text(row.get("rmp_rating"))

        print(f"[{count}/{len(needs)}] Enhancing {name}...")
        result = scrape_professor_page(url, current_rating=rating)
        print(
            f"    -> {result['rmp_page_status']} | "
            f"difficulty={result['rmp_difficulty']} | "
            f"again={result['rmp_would_take_again']} | "
            f"tags={result['rmp_page_tags'][:60]}"
        )

        for col, value in result.items():
            df.at[idx, col] = "" if value is None else str(value)

        save_cache(df)
        time.sleep(2.0)

    print()
    print("Done. Saved enhanced RMP cache.")


if __name__ == "__main__":
    main()