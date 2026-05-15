from __future__ import annotations

import itertools
import re
from datetime import datetime

import pandas as pd


DAY_ORDER = {"M": 0, "T": 1, "W": 2, "Th": 3, "F": 4}


def _real_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def parse_days(days_raw: str) -> set[str]:
    s = str(days_raw)
    days = set()

    if "Th" in s or "Thursday" in s:
        days.add("Th")
        s = s.replace("Thursday", "").replace("Th", "")

    for full, code in [
        ("Monday", "M"),
        ("Tuesday", "T"),
        ("Wednesday", "W"),
        ("Friday", "F"),
    ]:
        if full in s:
            days.add(code)

    for c in ["M", "T", "W", "F"]:
        if re.search(rf"\b{c}\b", s):
            days.add(c)

    return days


def minutes(t: str) -> int | None:
    if not str(t).strip():
        return None

    for fmt in ["%I:%M %p", "%H:%M", "%I %p"]:
        try:
            dt = datetime.strptime(str(t).strip(), fmt)
            return dt.hour * 60 + dt.minute
        except Exception:
            pass

    return None


def date_overlap(a_start, a_end, b_start, b_end) -> bool:
    if pd.isna(a_start) or pd.isna(a_end) or pd.isna(b_start) or pd.isna(b_end):
        return True
    return max(a_start, b_start) <= min(a_end, b_end)


def time_conflict(a: pd.Series, b: pd.Series) -> bool:
    if not (parse_days(a.get("days_raw", "")) & parse_days(b.get("days_raw", ""))):
        return False

    if not date_overlap(
        a.get("start_date"),
        a.get("end_date"),
        b.get("start_date"),
        b.get("end_date"),
    ):
        return False

    a1, a2 = minutes(a.get("start_time", "")), minutes(a.get("end_time", ""))
    b1, b2 = minutes(b.get("start_time", "")), minutes(b.get("end_time", ""))

    if None in [a1, a2, b1, b2]:
        return False

    return max(a1, b1) < min(a2, b2)


def hard_filter(df: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    out = df.copy()

    avoid = set(prefs.get("avoid_days", []))
    if avoid:
        out = out[~out["days_raw"].apply(lambda x: bool(parse_days(x) & avoid))]

    earliest = minutes(prefs.get("earliest_time")) if prefs.get("earliest_time") else None
    latest = minutes(prefs.get("latest_time")) if prefs.get("latest_time") else None

    if earliest is not None:
        out = out[
            out["start_time"].apply(
                lambda x: (minutes(x) is None) or minutes(x) >= earliest
            )
        ]

    if latest is not None:
        out = out[
            out["end_time"].apply(
                lambda x: (minutes(x) is None) or minutes(x) <= latest
            )
        ]

    return out.reset_index(drop=True)


def course_interest_score(row: pd.Series, prefs: dict) -> float:
    text = " ".join(
        _real_text(row.get(c))
        for c in ["course_id", "title", "description", "attribute"]
    ).lower()

    score = 0.0

    for interest in prefs.get("interests", []):
        if str(interest).lower() in text:
            score += 1.5

    for goal in prefs.get("goals", []):
        for token in re.findall(r"[a-z]{4,}", str(goal).lower()):
            if token in text:
                score += 0.2

    return score


def row_score(row: pd.Series, prefs: dict, completed: set[str]) -> float:
    weights = prefs.get("weights", {})
    score = 0.0

    # Only boost if it's actually tied to a requirement
    if row.get("course_id") not in completed and _real_text(row.get("requirement_group")):
        score += weights.get("requirement_fit", 2)

    score += course_interest_score(row, prefs) * weights.get("interest_fit", 3)

    # RMP rating
    rating = row.get("rmp_rating")
    try:
        if pd.notna(rating) and str(rating).strip().lower() not in {"", "nan", "none"}:
            score += float(rating) * 0.8 * weights.get("professor", 2)
    except Exception:
        pass

    difficulty = row.get("rmp_difficulty")
    try:
        if pd.notna(difficulty) and str(difficulty).strip().lower() not in {"", "nan", "none"}:
            d = float(difficulty)

            # Lower difficulty is better, especially if the student asks for manageable/easy classes.
            ease_weight = weights.get("easy", 1)
            score += max(0, 5 - d) * 0.5 * ease_weight
    except Exception:
        pass

    again = row.get("rmp_would_take_again")
    try:
        if pd.notna(again) and str(again).strip().lower() not in {"", "nan", "none"}:
            score += (float(again) / 100) * 2.0 * weights.get("professor", 2)
    except Exception:
        pass

    # General engagement signal (light)
    tags = _real_text(row.get("rmp_tags")).lower()
    if any(x in tags for x in ["inspiring", "engaging", "passionate", "interesting"]):
        score += 0.6 * weights.get("professor", 2)

    # Professor preferences from NLP
    prof_prefs = prefs.get("professor_preferences", [])

    rmp_text = " ".join([
        _real_text(row.get("rmp_tags")),
        _real_text(row.get("rmp_snippet"))
    ]).lower()

    if "easy" in prof_prefs:
        try:
            difficulty = float(row.get("rmp_difficulty"))
            score += max(0, 5 - difficulty) * weights.get("professor", 2)
        except Exception:
            pass

    if "supportive" in prof_prefs and any(x in rmp_text for x in ["caring", "helpful", "supportive", "kind"]):
        score += 2.5 * weights.get("professor", 2)

    if "engaging" in prof_prefs and any(x in rmp_text for x in ["engaging", "inspiring", "passionate"]):
        score += 2.5 * weights.get("professor", 2)

    return score


def schedule_features(rows: list[pd.Series]) -> dict:
    all_days = set().union(
        *[parse_days(r.get("days_raw", "")) for r in rows]
    ) if rows else set()

    starts = [
        minutes(r.get("start_time", ""))
        for r in rows
        if minutes(r.get("start_time", "")) is not None
    ]

    ends = [
        minutes(r.get("end_time", ""))
        for r in rows
        if minutes(r.get("end_time", "")) is not None
    ]

    return {
        "days_count": len(all_days),
        "has_friday": "F" in all_days,
        "earliest": min(starts) if starts else None,
        "latest": max(ends) if ends else None,
    }

def bucket_limit_ok(rows: list[pd.Series], new_row: pd.Series) -> bool:
    group = _real_text(new_row.get("requirement_group"))

    if not group:
        return True

    same_bucket = [
        r for r in rows
        if _real_text(r.get("requirement_group")) == group
    ]

    credit_limit = new_row.get("bucket_limit_credits")
    course_limit = new_row.get("bucket_limit_courses")

    try:
        if pd.notna(credit_limit) and str(credit_limit).strip() not in {"", "nan", "None"}:
            limit = float(credit_limit)
            current_credits = sum(float(r.get("credit_hours") or 0) for r in same_bucket)
            new_credits = float(new_row.get("credit_hours") or 0)
            if current_credits + new_credits > limit:
                return False
    except Exception:
        pass

    try:
        if pd.notna(course_limit) and str(course_limit).strip() not in {"", "nan", "None"}:
            limit = int(float(course_limit))
            if len(same_bucket) + 1 > limit:
                return False
    except Exception:
        pass

    return True

def compatible(rows: list[pd.Series], new_row: pd.Series) -> bool:
    if any(r.get("course_id") == new_row.get("course_id") for r in rows):
        return False

    return not any(time_conflict(r, new_row) for r in rows)


def generate_schedules(
    df: pd.DataFrame,
    prefs: dict,
    completed: set[str],
    min_credits=15.0,
    max_credits=17.5,
    max_schedules=200,
):
    df = hard_filter(df, prefs)
    df = df[~df["course_id"].isin(completed)].copy()

    if df.empty:
        return []

    df["base_score"] = df.apply(lambda r: row_score(r, prefs, completed), axis=1)
    df = df.sort_values("base_score", ascending=False).head(80)

    schedules = []
    rows = [r for _, r in df.iterrows()]

    def backtrack(start_idx, chosen, credits):
        if len(schedules) >= max_schedules:
            return

        if min_credits <= credits <= max_credits:
            schedules.append(list(chosen))

        if credits >= max_credits:
            return

        for i in range(start_idx, len(rows)):
            r = rows[i]
            c = float(r.get("credit_hours") or 0)

            if c <= 0 or credits + c > max_credits:
                continue

            if compatible(chosen, r):
                chosen.append(r)
                backtrack(i + 1, chosen, credits + c)
                chosen.pop()

    backtrack(0, [], 0.0)

    return rank_schedules(schedules, prefs, completed)[:10]


def rank_schedules(
    schedules: list[list[pd.Series]],
    prefs: dict,
    completed: set[str],
):
    ranked = []

    for rows in schedules:
        base = sum(row_score(r, prefs, completed) for r in rows)
        feats = schedule_features(rows)
        reasons = []

        if prefs.get("weights", {}).get("compact", 1) >= 3:
            base -= feats["days_count"] * 1.5
            reasons.append(
                f"keeps classes to {feats['days_count']} day(s), which fits your compact-schedule preference"
            )

        if not feats["has_friday"]:
            base += prefs.get("weights", {}).get("no_friday", 1) * 1.5
            reasons.append("keeps Fridays free")

        if prefs.get("latest_time"):
            reasons.append(f"avoids classes after {prefs['latest_time']}")

        if prefs.get("earliest_time"):
            reasons.append(f"avoids classes before {prefs['earliest_time']}")

        ranked.append((base, reasons, rows))

    return sorted(ranked, key=lambda x: x[0], reverse=True)


def explain_course(row: pd.Series, prefs: dict) -> str:
    pieces = []

    course_id = _real_text(row.get("course_id"))
    subject = course_id.split("-")[0] if "-" in course_id else ""

    title = _real_text(row.get("title")).lower()
    description = _real_text(row.get("description")).lower()
    attribute = _real_text(row.get("attribute")).lower()

    text = " ".join([title, description, attribute])

    interests = [
        interest
        for interest in prefs.get("interests", [])
        if str(interest).lower() in text
    ]

    if interests:
        pieces.append(f"matches your interest in {', '.join(interests[:3])}")

    if subject == "FINC":
        pieces.append(
            "supports your finance path with practical investing, markets, or valuation skills"
        )
    elif subject == "MGMT":
        pieces.append(
            "supports your management path and connects to leadership, organizations, or entrepreneurship"
        )
    elif subject == "OPAN":
        pieces.append(
            "connects to analytics, data, and applied problem-solving"
        )
    elif subject == "COSC":
        pieces.append(
            "builds technical fluency that can support your interest in AI and software"
        )
    elif subject in {"GOVT", "INAF"}:
        pieces.append(
            "connects to government, policy, and institutions"
        )
    elif subject in {"LING", "ANTH", "CCTP"}:
        pieces.append(
            "connects to language, culture, communication, or society"
        )

    if "entrepreneur" in text or "startup" in text:
        pieces.append("connects directly to entrepreneurship and startup interests")

    if "creativity" in text or "imagination" in text:
        pieces.append(
            "may appeal to your creative side rather than feeling like a purely technical requirement"
        )

    if "sports" in text and any(x in text for x in ["data", "analytics", "viz", "visualization"]):
        pieces.append("blends analytics with a more fun, applied topic")

    if "real estate" in text:
        pieces.append("adds a tangible, deal-oriented finance angle")

    if "private equity" in text:
        pieces.append("aligns with investing, venture, and high-growth company interests")

    if "valuation" in text:
        pieces.append("builds a core skill for finance, investing, and venture work")

    if "derivatives" in text:
        pieces.append("adds a more technical markets skill set")

    if "global" in text or "international" in text:
        pieces.append("connects to your international perspective and interest in global systems")

    rating = row.get("rmp_rating")
    try:
        if pd.notna(rating) and str(rating).strip().lower() not in {"", "nan", "none"}:
            r = float(rating)
            if r >= 4.7:
                pieces.append(f"excellent professor rating ({r:.1f}/5)")
            elif r >= 4.2:
                pieces.append(f"strong professor rating ({r:.1f}/5)")
            else:
                pieces.append(f"professor rating signal: {r:.1f}/5")
    except Exception:
        pass

    difficulty = row.get("rmp_difficulty")
    try:
        if pd.notna(difficulty) and str(difficulty).strip().lower() not in {"", "nan", "none"}:
            d = float(difficulty)
            wants_easy = str(prefs.get("notes", "")).lower()
            wants_easy = wants_easy + " " + " ".join(str(x).lower() for x in prefs.get("goals", []))

            if d <= 2.3:
                pieces.append(f"students rate this professor as very manageable ({d:.1f}/5 difficulty)")
            elif d <= 3.2:
                pieces.append(f"difficulty looks moderate ({d:.1f}/5)")
            elif d >= 4.0:
                pieces.append(f"students rate this as demanding ({d:.1f}/5 difficulty)")
    except Exception:
        pass

    again = row.get("rmp_would_take_again")
    try:
        if pd.notna(again) and str(again).strip().lower() not in {"", "nan", "none"}:
            a = float(again)
            if a >= 85:
                pieces.append(f"{a:.0f}% of students would take this professor again")
            elif a >= 70:
                pieces.append(f"a solid {a:.0f}% of students would take this professor again")
            else:
                pieces.append(f"{a:.0f}% would-take-again signal")
    except Exception:
        pass

    rmp_tags = _real_text(row.get("rmp_tags"))
    rmp_snippet = _real_text(row.get("rmp_snippet"))

    if rmp_tags:
        pieces.append(f"student review themes: {rmp_tags}")

    if rmp_snippet:
        pieces.append(f"student review signal: {rmp_snippet}")

    if not pieces:
        pieces.append("fits your academic plan and schedule constraints")

    return "; ".join(dict.fromkeys(pieces))


def rows_to_df(rows: list[pd.Series], prefs: dict) -> pd.DataFrame:
    cols = [
        "course_id",
        "title",
        "section",
        "crn",
        "credit_hours",
        "instructor_clean",
        "days_raw",
        "start_time",
        "end_time",
        "start_date",
        "end_date",
        "rmp_rating",
        "rmp_difficulty",
        "rmp_would_take_again",
        "rmp_tags",
        "source_url",
    ]

    data = []

    for r in rows:
        item = {c: r.get(c, "") for c in cols}
        item["why_you_might_like_it"] = explain_course(r, prefs)
        data.append(item)

    return pd.DataFrame(data)
