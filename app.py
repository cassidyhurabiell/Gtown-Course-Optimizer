from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import pandas as pd
import streamlit as st
import html
import streamlit.components.v1 as components

from config import GEORGETOWN_BLUE, GEORGETOWN_GRAY, SOFT_PINK, LIGHT_PINK, BACKGROUND, CARD_BACKGROUND, TEXT_DARK, BORDER
from data_cleaning import load_courses, load_requirements
from nlp_profile import parse_student_profile
from transcript_parser import extract_text_from_uploaded_file, parse_completed_courses
from rmp_scraper import enrich_with_rmp
from optimizer import generate_schedules, rows_to_df, parse_days
from requirements_logic import calculate_unmet_requirements, attach_requirement_metadata


BASE = Path(__file__).resolve().parent


st.set_page_config(page_title="Georgetown Smart Schedule Optimizer", layout="wide")

st.markdown(
    f"""
<style>
.stApp {{
    background: {BACKGROUND};
    color: {TEXT_DARK};
}}

.block-container {{
    padding-top: 1.5rem;
    max-width: 1250px;
}}

h1, h2, h3 {{
    color: {GEORGETOWN_BLUE};
    letter-spacing: -0.02em;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {GEORGETOWN_BLUE}, #12335F);
}}

[data-testid="stSidebar"] * {{
    color: white !important;
}}

.stButton>button {{
    background-color: {SOFT_PINK};
    color: {GEORGETOWN_BLUE};
    border: 0;
    font-weight: 800;
    border-radius: 999px;
    padding: 0.65rem 1.4rem;
}}

.stDownloadButton>button {{
    background-color: {GEORGETOWN_BLUE};
    color: white;
    border: 0;
    border-radius: 999px;
}}

div[data-testid="stMetric"] {{
    background: {CARD_BACKGROUND};
    border-left: 5px solid {SOFT_PINK};
    padding: 0.8rem;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(4, 30, 66, 0.08);
}}

.pretty-card {{
    background: {CARD_BACKGROUND};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 6px 20px rgba(4, 30, 66, 0.07);
    margin-bottom: 1rem;
}}

.pink-pill {{
    display: inline-block;
    background: {LIGHT_PINK};
    color: {GEORGETOWN_BLUE};
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-right: 0.35rem;
}}

.calendar {{
    display: grid;
    grid-template-columns: 70px repeat(5, 1fr);
    border: 1px solid {BORDER};
    border-radius: 18px;
    overflow: hidden;
    background: white;
    margin-top: 0.8rem;
}}

.calendar-header {{
    background: {GEORGETOWN_BLUE};
    color: white;
    font-weight: 800;
    text-align: center;
    padding: 0.55rem;
    font-size: 0.85rem;
}}

.calendar-time {{
    background: #F1F3F6;
    color: {GEORGETOWN_BLUE};
    font-size: 0.75rem;
    padding: 0.35rem;
    border-top: 1px solid {BORDER};
}}

.calendar-cell {{
    min-height: 52px;
    border-top: 1px solid {BORDER};
    border-left: 1px solid {BORDER};
    padding: 0.25rem;
    position: relative;
}}

.class-block {{
    background: {LIGHT_PINK};
    border-left: 4px solid {SOFT_PINK};
    color: {GEORGETOWN_BLUE};
    border-radius: 10px;
    padding: 0.3rem 0.4rem;
    font-size: 0.72rem;
    line-height: 1.1;
    margin-bottom: 0.25rem;
    box-shadow: 0 2px 6px rgba(4, 30, 66, 0.10);
}}

.class-block.mod-a {{
    width: 46%;
    float: left;
}}

.class-block.mod-b {{
    width: 46%;
    float: right;
}}

.class-block.full-term {{
    width: 95%;
    clear: both;
}}

.small-muted {{
    color: #667085;
    font-size: 0.85rem;
}}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Georgetown Smart Schedule Optimizer")
st.caption(
    "Local NLP, transcript parsing, credit-based schedule generation, requirement-aware planning, "
    "and optional RMP scraping/cache. No LLM API required."
)


@st.cache_data(show_spinner=False)
def cached_courses(uploaded_name, uploaded_bytes):
    if uploaded_bytes is not None:
        tmp = BASE / "_uploaded_courses.csv"
        tmp.write_bytes(uploaded_bytes)
        return load_courses(tmp)

    for name in [
        "clean_schedule_sections.csv",
        "courses.csv",
        "schedule.csv",
        "aiMgmtSpring26SoC.csv",
        "aiMgmtSpring26SoC(1).csv",
    ]:
        p = BASE / name
        if p.exists():
            return load_courses(p)

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def cached_requirements(uploaded_name, uploaded_bytes):
    if uploaded_bytes is not None:
        tmp = BASE / "_uploaded_requirements.csv"
        tmp.write_bytes(uploaded_bytes)
        return load_requirements(tmp)

    for name in [
        "requirements.csv",
        "georgetown_msb_majors_flat.csv",
        "georgetown_msb_majors_flat(2).csv",
    ]:
        p = BASE / name
        if p.exists():
            return load_requirements(p)

    return pd.DataFrame()


def normalize_course_code(code: str) -> str:
    text = str(code).strip().upper()
    text = text.replace(":", "")
    text = " ".join(text.split())

    if "-" in text:
        return text

    parts = text.split()
    if len(parts) >= 2 and parts[1].isdigit():
        return f"{parts[0]}-{parts[1]}"

    return text


def build_minor_requirements(selected_minors: list[str]) -> pd.DataFrame:
    rows = []

    if "Computer Science" in selected_minors:
        rows.extend(
            [
                {
                    "major": "Minor: Computer Science",
                    "category": "Minor",
                    "requirement_group": "I. Required Computer Science Minor Courses",
                    "course_code": "COSC 1020",
                    "course_id": "COSC-1020",
                    "course_title": "Computer Science I",
                    "credits": 3,
                    "notes": "Required for CS minor",
                    "source": "User-provided curriculum text",
                },
                {
                    "major": "Minor: Computer Science",
                    "category": "Minor",
                    "requirement_group": "I. Required Computer Science Minor Courses",
                    "course_code": "COSC 1030",
                    "course_id": "COSC-1030",
                    "course_title": "Computer Science II",
                    "credits": 3,
                    "notes": "Required for CS minor",
                    "source": "User-provided curriculum text",
                },
                {
                    "major": "Minor: Computer Science",
                    "category": "Minor",
                    "requirement_group": "I. Required Computer Science Minor Courses",
                    "course_code": "COSC 1110",
                    "course_id": "COSC-1110",
                    "course_title": "Mathematical Methods for Computer Science",
                    "credits": 3,
                    "notes": "MATH-2800 may substitute for math majors/minors",
                    "source": "User-provided curriculum text",
                },
                {
                    "major": "Minor: Computer Science",
                    "category": "Minor",
                    "requirement_group": "I. Required Computer Science Minor Courses",
                    "course_code": "COSC 2010",
                    "course_id": "COSC-2010",
                    "course_title": "Data Structures",
                    "credits": 3,
                    "notes": "Required for CS minor",
                    "source": "User-provided curriculum text",
                },
            ]
        )

        # These are handled as a broad bucket later by schedule-course matching.
        rows.append(
            {
                "major": "Minor: Computer Science",
                "category": "Minor",
                "requirement_group": "II. Two COSC Electives Numbered 2000-4999",
                "course_code": "Any 2000-4999 COSC Course",
                "course_id": "COSC-ELECTIVE-2000-4999",
                "course_title": "Any two undergraduate COSC electives numbered 2000-4999",
                "credits": 6,
                "notes": "Do not use graduate-level 5000+ courses",
                "source": "User-provided curriculum text",
            }
        )

    if "Entrepreneurship" in selected_minors:
        rows.extend(
            [
                {
                    "major": "Minor: Entrepreneurship",
                    "category": "Minor",
                    "requirement_group": "I. Three Required Entrepreneurship Courses",
                    "course_code": "MGMT 2220",
                    "course_id": "MGMT-2220",
                    "course_title": "Foundations of Entrepreneurship",
                    "credits": 3,
                    "notes": "Required for entrepreneurship minor",
                    "source": "User-provided curriculum text",
                },
                {
                    "major": "Minor: Entrepreneurship",
                    "category": "Minor",
                    "requirement_group": "I. Three Required Entrepreneurship Courses",
                    "course_code": "MGMT 2221",
                    "course_id": "MGMT-2221",
                    "course_title": "Entrepreneurial Changemakers for the Common Good",
                    "credits": 3,
                    "notes": "Required for entrepreneurship minor",
                    "source": "User-provided curriculum text",
                },
                {
                    "major": "Minor: Entrepreneurship",
                    "category": "Minor",
                    "requirement_group": "II. One Entrepreneurship Capstone Course",
                    "course_code": "MGMT 3224",
                    "course_id": "MGMT-3224",
                    "course_title": "Launching Entrepreneurial Ventures",
                    "credits": 3,
                    "notes": "Choose MGMT-3224 or MGMT-3225",
                    "source": "User-provided curriculum text",
                },
                {
                    "major": "Minor: Entrepreneurship",
                    "category": "Minor",
                    "requirement_group": "II. One Entrepreneurship Capstone Course",
                    "course_code": "MGMT 3225",
                    "course_id": "MGMT-3225",
                    "course_title": "Growing Entrepreneurial Businesses",
                    "credits": 3,
                    "notes": "Choose MGMT-3224 or MGMT-3225",
                    "source": "User-provided curriculum text",
                },
            ]
        )

        elective_courses = [
            ("ACCT 3103", "Accounting and Management Strategy", 3),
            ("ECON 4416", "Market Design", 3),
            ("FINC 3101", "Corporate Valuation", 1.5),
            ("FINC 3265", "Private Equity", 1.5),
            ("FINC 3266", "Venture Capital", 1.5),
            ("GBUS 4972", "C-Lab: Startup Studio", 3),
            ("GBUS 4492", "Law, Business, and Entrepreneurship", 3),
            ("MARK 3101", "Marketing Intelligence", 3),
            ("MARK 3227", "Branding", 3),
            ("MARK 3235", "Social and Digital Media Marketing", 1.5),
            ("MGMT 3224", "Launching Entrepreneurial Ventures", 3),
            ("MGMT 3225", "Growing Entrepreneurial Businesses", 3),
            ("MGMT 3277", "Imagination and Creativity", 3),
            ("OPAN 3243", "Intro Bus App Development in Python", 1.5),
            ("OPAN 3244", "Mgmt Bus App Development in Python", 1.5),
            ("OPAN 3256", "Digital Technologies and Analytics", 3),
            ("OPAN 3271", "Environmental Sustainability Operations and Business Models", 3),
            ("STIA 3005", "Science Tech in Global Arena", 3),
        ]

        for code, title, credits in elective_courses:
            rows.append(
                {
                    "major": "Minor: Entrepreneurship",
                    "category": "Minor",
                    "requirement_group": "III. Three Credits of Entrepreneurship Electives",
                    "course_code": code,
                    "course_id": normalize_course_code(code),
                    "course_title": title,
                    "credits": credits,
                    "notes": "Only three credits of the minor can overlay with an MSB major requirement",
                    "source": "User-provided curriculum text",
                }
            )

    return pd.DataFrame(rows)


def add_broad_requirement_matches(requirements: pd.DataFrame, courses: pd.DataFrame) -> pd.DataFrame:
    if requirements.empty:
        return requirements

    extra_rows = []

    for _, req in requirements.iterrows():
        course_id = str(req.get("course_id", ""))

        if course_id == "COSC-ELECTIVE-2000-4999":
            matches = courses[
                (courses["course_id"].astype(str).str.startswith("COSC-"))
                & (
                    courses["course_number"]
                    .astype(str)
                    .str.extract(r"(\d+)", expand=False)
                    .fillna("0")
                    .astype(int)
                    .between(2000, 4999)
                )
            ]

            for _, course in matches.iterrows():
                new_row = req.copy()
                new_row["course_id"] = course["course_id"]
                new_row["course_code"] = course["course_id"]
                new_row["course_title"] = course.get("title", "")
                new_row["credits"] = str(course.get("credit_hours", ""))
                extra_rows.append(new_row)

        course_code = str(req.get("course_code", "")).lower()
        course_title = str(req.get("course_title", "")).lower()

        if "any 3000-level finc" in course_code or "any 3000-level finc" in course_title:
            matches = courses[
                (courses["course_id"].astype(str).str.startswith("FINC-"))
                & (
                    courses["course_number"]
                    .astype(str)
                    .str.extract(r"(\d+)", expand=False)
                    .fillna("0")
                    .astype(int)
                    .between(3000, 3999)
                )
            ]

            for _, course in matches.iterrows():
                new_row = req.copy()
                new_row["course_id"] = course["course_id"]
                new_row["course_code"] = course["course_id"]
                new_row["course_title"] = course.get("title", "")
                new_row["credits"] = str(course.get("credit_hours", ""))
                extra_rows.append(new_row)

    if extra_rows:
        requirements = pd.concat([requirements, pd.DataFrame(extra_rows)], ignore_index=True)

    return requirements


def apply_manual_preferences(prefs: dict, no_friday: bool, start_after: str, end_before: str, study_abroad: bool, study_abroad_term: str | None, language_needed: str | None, selected_minors: list[str]) -> dict:
    prefs = deepcopy(prefs)
    prefs.setdefault("weights", {})
    # Let the checkbox control Friday preference.
    # If no_friday is not checked, remove any Friday avoidance inferred from text.
    if not no_friday:
        prefs["avoid_days"] = [d for d in prefs.get("avoid_days", []) if d != "F"]
        prefs["weights"]["no_friday"] = 0

    if no_friday:
        prefs.setdefault("avoid_days", [])
        if "F" not in prefs["avoid_days"]:
            prefs["avoid_days"].append("F")
        prefs["weights"]["no_friday"] = max(prefs["weights"].get("no_friday", 1), 4)

    if start_after and start_after != "No preference":
        prefs["earliest_time"] = start_after

    if end_before and end_before != "No preference":
        prefs["latest_time"] = end_before

    prefs["study_abroad"] = study_abroad
    prefs["study_abroad_term"] = study_abroad_term
    prefs["language_needed"] = language_needed
    prefs["selected_minors"] = selected_minors

    if study_abroad:
        prefs["weights"]["requirement_fit"] = max(prefs["weights"].get("requirement_fit", 2), 5)
        prefs["planning_note"] = "Study abroad selected, so the scheduler should prioritize major/minor progress earlier."

    return prefs


def build_candidate_pool(courses: pd.DataFrame, required_ids: set[str], prefs: dict, mode: str) -> pd.DataFrame:
    courses = courses.copy()

    if not required_ids:
        return courses

    required = courses[courses["course_id"].isin(required_ids)].copy()

    text_cols = [c for c in ["title", "description", "attribute", "course_id"] if c in courses.columns]
    search_text = courses[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

    interest_terms = [str(x).lower() for x in prefs.get("interests", []) if str(x).strip()]
    career_terms = []
    for goal in prefs.get("goals", []):
        career_terms.extend([x.lower() for x in str(goal).split() if len(x) >= 4])

    extra_terms = set(interest_terms + career_terms)
    if prefs.get("language_needed") and prefs.get("language_needed") not in [None, "None / not sure"]:
        extra_terms.add(str(prefs.get("language_needed")).lower())

    if extra_terms:
        pattern = "|".join([term.replace("+", r"\+") for term in extra_terms])
        interest = courses[search_text.str.contains(pattern, case=False, na=False, regex=True)].copy()
    else:
        interest = courses.iloc[0:0].copy()

    if mode == "Requirement-heavy":
        pool = required

    elif mode == "Balanced":
        pool = pd.concat([required, interest], ignore_index=True)

    elif mode == "Interest/career-heavy":
        pool = pd.concat([required, interest], ignore_index=True)

        # Let a few high-interest technical/policy/data classes compete, but do not replace requirements entirely.
        subject_boosts = ["COSC", "OPAN", "CCTP", "STIA", "GOVT", "LING"]
        extra = courses[courses["course_id"].astype(str).str.split("-").str[0].isin(subject_boosts)]
        pool = pd.concat([pool, extra], ignore_index=True)

    else:
        pool = required

    pool = pool.drop_duplicates(subset=["course_id", "section", "crn"])
    return pool


def mode_prefs(base_prefs: dict, mode: str, class_year: str, study_abroad: bool) -> dict:
    prefs = deepcopy(base_prefs)
    prefs.setdefault("weights", {})

    if mode == "Requirement-heavy":
        prefs["weights"]["requirement_fit"] = 7
        prefs["weights"]["interest_fit"] = 1
        prefs["weights"]["professor"] = max(1, prefs["weights"].get("professor", 1))

    elif mode == "Balanced":
        prefs["weights"]["requirement_fit"] = 5
        prefs["weights"]["interest_fit"] = 3

    elif mode == "Interest/career-heavy":
        prefs["weights"]["requirement_fit"] = 3
        prefs["weights"]["interest_fit"] = 6
        prefs["weights"]["professor"] = max(2, prefs["weights"].get("professor", 2))

    if class_year in ["Junior", "Senior"] or study_abroad:
        prefs["weights"]["requirement_fit"] = max(prefs["weights"].get("requirement_fit", 2), 6)

    return prefs


def explain_schedule_overall(rows, prefs: dict, mode_name: str, selected_majors: list[str], selected_minors: list[str]) -> str:
    course_ids = [str(r.get("course_id", "")) for r in rows]
    subjects = [cid.split("-")[0] for cid in course_ids if "-" in cid]

    all_days = set()
    for r in rows:
        all_days |= parse_days(r.get("days_raw", ""))

    pieces = [f"This is the {mode_name.lower()} option."]

    if selected_majors:
        pieces.append(f"It is built around progress in {', '.join(selected_majors)}.")

    if selected_minors:
        pieces.append(f"It also accounts for the {', '.join(selected_minors)} minor path.")

    if subjects.count("FINC") >= 3:
        pieces.append("It makes strong Finance progress without relying only on unrelated interest matches.")

    if subjects.count("MGMT") >= 2:
        pieces.append("It keeps Management meaningfully represented instead of letting Finance dominate.")

    if any(s in {"COSC", "OPAN"} for s in subjects):
        pieces.append("It adds technical or analytics fluency, which fits fintech, AI, or project-management goals.")

    if prefs.get("study_abroad"):
        pieces.append("Because study abroad is selected, this schedule favors requirements that are safer to complete before being off campus.")

    if prefs.get("language_needed") and prefs.get("language_needed") not in [None, "None / not sure"]:
        pieces.append(f"It keeps language planning visible for {prefs.get('language_needed')}.")

    if prefs.get("avoid_days") and "F" in prefs.get("avoid_days", []) and "F" not in all_days:
        pieces.append("It keeps Fridays free, matching your preference.")

    if prefs.get("earliest_time"):
        pieces.append(f"It avoids starts before {prefs.get('earliest_time')}.")

    if prefs.get("latest_time"):
        pieces.append(f"It avoids ending after {prefs.get('latest_time')}.")

    return " ".join(pieces)

def _time_to_minutes(t: str) -> int | None:
    if not t or str(t).strip().lower() in {"nan", "none", ""}:
        return None

    for fmt in ["%I:%M %p", "%H:%M"]:
        try:
            dt = pd.to_datetime(str(t).strip(), format=fmt)
            return int(dt.hour * 60 + dt.minute)
        except Exception:
            pass

    try:
        dt = pd.to_datetime(str(t).strip())
        return int(dt.hour * 60 + dt.minute)
    except Exception:
        return None


def _term_class(start_date, end_date) -> str:
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")

    if pd.isna(start) or pd.isna(end):
        return "full-term"

    days = (end - start).days

    if days >= 90:
        return "full-term"

    if end.month <= 10:
        return "mod-a"

    return "mod-b"

def render_schedule_calendar(rows) -> str:
    days = ["M", "T", "W", "Th", "F"]
    day_labels = {"M": "Mon", "T": "Tue", "W": "Wed", "Th": "Thu", "F": "Fri"}

    start_hour = 8
    end_hour = 20

    def safe(x):
        return html.escape(str(x if x is not None else ""))

    def block_for_course(r):
        term_class = _term_class(r.get("start_date"), r.get("end_date"))
        course_id = safe(r.get("course_id", ""))
        title = safe(str(r.get("title", ""))[:30])
        time_label = safe(f"{r.get('start_time', '')}–{r.get('end_time', '')}")

        return (
            f'<div class="class-block {term_class}">'
            f'<strong>{course_id}</strong><br>'
            f'{title}<br>'
            f'<span>{time_label}</span>'
            f'</div>'
        )

    body = ['<div class="calendar">']
    body.append('<div class="calendar-header">Time</div>')

    for d in days:
        body.append(f'<div class="calendar-header">{day_labels[d]}</div>')

    for hour in range(start_hour, end_hour):
        label = (
            pd.Timestamp(year=2026, month=1, day=1, hour=hour)
            .strftime("%I:%M %p")
            .lstrip("0")
        )

        body.append(f'<div class="calendar-time">{label}</div>')

        for d in days:
            cell_blocks = []

            for r in rows:
                course_days = parse_days(r.get("days_raw", ""))
                if d not in course_days:
                    continue

                start_min = _time_to_minutes(r.get("start_time", ""))
                end_min = _time_to_minutes(r.get("end_time", ""))

                if start_min is None or end_min is None:
                    continue

                if hour * 60 <= start_min < (hour + 1) * 60:
                    cell_blocks.append(block_for_course(r))

            body.append(f'<div class="calendar-cell">{"".join(cell_blocks)}</div>')

    body.append("</div>")

    return f"""
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: transparent;
        }}

        .calendar {{
            display: grid;
            grid-template-columns: 72px repeat(5, 1fr);
            border: 1px solid #D8DEE8;
            border-radius: 18px;
            overflow: hidden;
            background: white;
            width: 100%;
            box-sizing: border-box;
        }}

        .calendar-header {{
            background: #041E42;
            color: white;
            font-weight: 800;
            text-align: center;
            padding: 9px 5px;
            font-size: 13px;
        }}

        .calendar-time {{
            background: #F1F3F6;
            color: #041E42;
            font-size: 11px;
            padding: 6px;
            border-top: 1px solid #D8DEE8;
            min-height: 58px;
            box-sizing: border-box;
        }}

        .calendar-cell {{
            min-height: 58px;
            border-top: 1px solid #D8DEE8;
            border-left: 1px solid #D8DEE8;
            padding: 4px;
            box-sizing: border-box;
            overflow: hidden;
        }}

        .class-block {{
            background: #FBE7F0;
            border-left: 4px solid #F4A7C5;
            color: #041E42;
            border-radius: 10px;
            padding: 5px 6px;
            font-size: 11px;
            line-height: 1.15;
            margin-bottom: 4px;
            box-shadow: 0 2px 6px rgba(4, 30, 66, 0.10);
            box-sizing: border-box;
            overflow-wrap: break-word;
        }}

        .class-block.mod-a {{
            width: 47%;
            float: left;
            background: #FBE7F0;
        }}

        .class-block.mod-b {{
            width: 47%;
            float: right;
            background: #E8EEF9;
            border-left-color: #3B4A7C;
        }}

        .class-block.full-term {{
            width: 96%;
            clear: both;
            background: #F4F7FB;
            border-left-color: #041E42;
        }}

        .class-block span {{
            color: #667085;
            font-size: 10px;
        }}
    </style>
    </head>
    <body>
        {''.join(body)}
    </body>
    </html>
    """


def display_schedule_table(rows, prefs):
    df = rows_to_df(rows, prefs).copy()

    if "instructor_clean" in df.columns:
        df["instructor"] = df["instructor_clean"]

    # Add term part from the actual row data.
    term_parts = []
    for r in rows:
        term_parts.append(_term_class(r.get("start_date"), r.get("end_date")).replace("-", "_"))

    df["term"] = term_parts

    display_cols = [
        "course_id",
        "title",
        "crn",
        "instructor",
        "rmp_rating",
        "rmp_difficulty",
        "rmp_would_take_again",
        "why_you_might_like_it",
        "days_raw",
        "start_time",
        "end_time",
        "term",
    ]

    for col in display_cols:
        if col not in df.columns:
            df[col] = ""

    display_df = df[display_cols].copy()

    display_df = display_df.rename(
        columns={
            "course_id": "Course ID",
            "title": "Title",
            "crn": "CRN",
            "instructor": "Instructor",
            "rmp_rating": "RMP",
            "rmp_difficulty": "Difficulty",
            "rmp_would_take_again": "Would take again",
            "why_you_might_like_it": "Why you might like it",
            "days_raw": "Days",
            "start_time": "Start",
            "end_time": "End",
            "term": "Term",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Why you might like it": st.column_config.TextColumn(
                "Why you might like it",
                width="large",
            ),
            "Title": st.column_config.TextColumn("Title", width="medium"),
        },
    )

    return df

with st.sidebar:
    st.markdown("### Setup")

    with st.expander("Data files", expanded=True):
        course_file = st.file_uploader("Schedule of Classes CSV", type=["csv"])
        req_file = st.file_uploader("Optional requirements CSV", type=["csv"])
        transcript_file = st.file_uploader(
            "Optional transcript / degree audit PDF or text",
            type=["pdf", "txt", "csv"],
        )

    with st.expander("Student plan", expanded=True):
        class_year = st.selectbox(
            "Class year",
            ["First-year", "Sophomore", "Junior", "Senior", "Other"],
            index=1,
        )

        min_credits = st.number_input(
            "Minimum credits",
            0.0,
            21.0,
            15.0,
            0.5,
        )

        max_credits = st.number_input(
            "Maximum credits",
            0.0,
            21.0,
            17.5,
            0.5,
        )

    with st.expander("Course eligibility", expanded=False):
        allow_grad_courses = st.checkbox(
            "Allow graduate-level courses",
            value=False,
        )

        allow_non_main_campus = st.checkbox(
            "Allow non-main-campus / online courses",
            value=False,
        )

    time_options = [
        "No preference",
        "08:00 AM",
        "08:30 AM",
        "09:00 AM",
        "09:30 AM",
        "10:00 AM",
        "10:30 AM",
        "11:00 AM",
        "12:00 PM",
        "01:00 PM",
        "02:00 PM",
        "03:00 PM",
        "04:00 PM",
        "05:00 PM",
        "06:00 PM",
        "07:00 PM",
        "08:00 PM",
    ]

    with st.expander("Schedule preferences", expanded=False):
        no_friday = st.checkbox("Prefer no Friday classes", value=False)

        start_after = st.selectbox(
            "Do not start before",
            time_options,
            index=0,
        )

        end_before = st.selectbox(
            "Prefer to end by",
            time_options,
            index=0,
        )

    with st.expander("Study abroad", expanded=False):
        study_abroad = st.checkbox("Planning to study abroad", value=False)
        study_abroad_term = None
        language_needed = None

        if study_abroad:
            study_abroad_term = st.selectbox(
                "Likely study abroad term",
                ["Fall junior year", "Spring junior year", "Not sure yet"],
            )

            language_needed = st.selectbox(
                "Language needed?",
                [
                    "None / not sure",
                    "Arabic",
                    "French",
                    "Spanish",
                    "Italian",
                    "German",
                    "Chinese",
                    "Other",
                ],
            )

    with st.expander("RMP", expanded=False):
        use_rmp_web = st.toggle(
            "Attempt live RMP scraping",
            value=False,
            help="Creates/uses rmp_cache.csv. Leave off while testing to avoid slow or blocked requests.",
        )

        max_rmp = st.slider(
            "Max live professor lookups",
            0,
            75,
            15,
        )

    with st.expander("Display", expanded=False):
        show_details = st.checkbox(
            "Show technical details",
            value=False,
        )


course_bytes = course_file.getvalue() if course_file is not None else None
req_bytes = req_file.getvalue() if req_file is not None else None

courses = cached_courses(course_file.name if course_file else None, course_bytes)
requirements = cached_requirements(req_file.name if req_file else None, req_bytes)

if courses.empty:
    st.error(
        "No course rows were loaded. Upload a Schedule of Classes CSV, or put "
        "clean_schedule_sections.csv, courses.csv, or schedule.csv in this folder."
    )
    st.stop()

minor_options = ["Entrepreneurship", "Computer Science"]

selected_majors = []
selected_minors = []

if not requirements.empty:
    major_options = sorted(
        [
            m
            for m in requirements["major"].dropna().astype(str).unique()
            if m not in ["", "All MSB Students"] and not m.startswith("Minor:")
        ]
    )

    with st.sidebar:
        selected_majors = st.multiselect(
            "Major(s) / concentration(s)",
            major_options,
            default=[],
        )

        selected_minors = st.multiselect(
            "Minor(s)",
            minor_options,
            default=[],
        )
else:
    with st.sidebar:
        selected_minors = st.multiselect(
            "Minor(s)",
            minor_options,
            default=[],
        )

minor_requirements = build_minor_requirements(selected_minors)

if not minor_requirements.empty:
    requirements = pd.concat([requirements, minor_requirements], ignore_index=True)

requirements = add_broad_requirement_matches(requirements, courses)


st.subheader("Student profile")
profile_text = st.text_area(
    "Paste anything about the student: personality, interests, preferred days/times, goals, learning style, constraints, etc.",
    height=170,
    value=(
        "I am curious, adventurous, and interested in AI, mythology, linguistics, "
        "government, fintech, startups, and project management. I prefer a compact "
        "schedule, strong professors, and classes that are useful but also genuinely interesting."
    ),
)

prefs = parse_student_profile(profile_text, class_year=class_year)
prefs = apply_manual_preferences(
    prefs,
    no_friday=no_friday,
    start_after=start_after,
    end_before=end_before,
    study_abroad=study_abroad,
    study_abroad_term=study_abroad_term,
    language_needed=language_needed,
    selected_minors=selected_minors,
)

easy_words = ["easy", "manageable", "not too hard", "lighter workload", "low stress", "chill"]
if any(word in profile_text.lower() for word in easy_words):
    prefs.setdefault("weights", {})["easy"] = 4
else:
    prefs.setdefault("weights", {})["easy"] = 1

if show_details:
    with st.expander("Parsed student profile", expanded=False):
        st.json(prefs)


st.subheader("Completed or in-progress courses")
completed = set()

completed_df = pd.DataFrame()

if transcript_file is not None:
    text = extract_text_from_uploaded_file(transcript_file)
    completed_df = parse_completed_courses(text, courses)
    completed = set(completed_df["course_id"].dropna().astype(str))
else:
    labels = courses[["course_id", "title", "course_label"]].drop_duplicates().sort_values("course_label")
    selected_labels = st.multiselect("Choose courses already taken", labels["course_label"].tolist())
    label_to_id = dict(zip(labels["course_label"], labels["course_id"]))
    completed = {label_to_id[x] for x in selected_labels if x in label_to_id}

required_course_ids = set()

if not requirements.empty:
    relevant_requirement_names = ["All MSB Students"] + selected_majors + [f"Minor: {m}" for m in selected_minors]

    relevant_requirements = requirements[
        requirements["major"].isin(relevant_requirement_names)
    ].copy()

    unmet_requirements = calculate_unmet_requirements(relevant_requirements, completed)

    required_course_ids = set(unmet_requirements["course_id"].dropna().astype(str))

    if show_details:
        with st.expander("Relevant unmet requirements", expanded=False):        
            display_cols = [
                c
                for c in [
                    "major",
                    "category",
                    "requirement_group",
                    "course_id",
                    "course_title",
                    "credits",
                    "requirement_status",
                    "bucket_limit_credits",
                    "bucket_limit_courses",
                    "notes",
                ]
                if c in unmet_requirements.columns
        ]

        st.dataframe(unmet_requirements[display_cols], use_container_width=True)
else:
    unmet_requirements = pd.DataFrame()

st.divider()

if use_rmp_web:
    with st.spinner("Checking professor fit through RMP cache/live scraping..."):
        courses_enriched = enrich_with_rmp(courses, use_web=True, max_lookups=max_rmp)
else:
    courses_enriched = enrich_with_rmp(courses, use_web=False, max_lookups=max_rmp)

courses_enriched = attach_requirement_metadata(courses_enriched, unmet_requirements)

if not allow_grad_courses:
    course_numbers = (
        courses_enriched["course_number"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("0")
        .astype(int)
    )
    courses_enriched = courses_enriched[course_numbers.lt(5000)]

if not allow_non_main_campus and "campus" in courses_enriched.columns:
    courses_enriched = courses_enriched[
        courses_enriched["campus"]
        .fillna("")
        .astype(str)
        .str.contains("Main Campus", case=False, na=False)
    ]


if show_details:
    with st.expander("Course data preview", expanded=False):
        show_cols = [
            c
            for c in [
                "course_id",
                "title",
                "credit_hours",
                "instructor_clean",
                "days_raw",
                "start_time",
                "end_time",
                "requirement_group",
                "requirement_rule",
                "bucket_limit_credits",
                "rmp_rating",
                "rmp_status",
            ]
            if c in courses_enriched.columns
        ]
        st.dataframe(courses_enriched[show_cols].head(300), use_container_width=True)

st.subheader("Schedule options")

schedule_modes = ["Balanced", "Requirement-heavy", "Interest/career-heavy"]

if st.button("Generate schedules", type="primary"):
    used_signatures = set()

    for mode in schedule_modes:
        mode_specific_prefs = mode_prefs(prefs, mode, class_year, study_abroad)
        candidate_pool = build_candidate_pool(courses_enriched, required_course_ids, mode_specific_prefs, mode)

        if candidate_pool.empty:
            st.warning(f"{mode}: no eligible course pool found.")
            continue

        ranked = generate_schedules(
            candidate_pool,
            mode_specific_prefs,
            completed,
            min_credits=min_credits,
            max_credits=max_credits,
            max_schedules=300,
        )

        if not ranked:
            st.warning(f"{mode}: no schedules found. Try widening credits or relaxing hard restrictions.")
            continue

        chosen = None
        for score, reasons, rows in ranked:
            signature = tuple(sorted(str(r.get("course_id", "")) + "-" + str(r.get("section", "")) for r in rows))
            if signature not in used_signatures:
                chosen = (score, reasons, rows)
                used_signatures.add(signature)
                break

        if chosen is None:
            chosen = ranked[0]

        score, reasons, rows = chosen
        total = sum(float(r.get("credit_hours") or 0) for r in rows)

        with st.expander(f"{mode} schedule: {total:.1f} credits, score {score:.2f}", expanded=(mode == "Balanced")):
            overall = explain_schedule_overall(
                rows,
                mode_specific_prefs,
                mode,
                selected_majors,
                selected_minors,
            )

            st.write("Why this overall schedule works:")
            st.write(overall)

            if reasons:
                st.write("Schedule structure:")
                st.write("; ".join(reasons))

            st.markdown("#### Calendar view")
            components.html(render_schedule_calendar(rows), height=760, scrolling=True)

            with st.expander("Class details and explanations", expanded=True):
                full_df = display_schedule_table(rows, mode_specific_prefs)

            st.download_button(
                f"Download {mode} schedule CSV",
                full_df.to_csv(index=False),
                file_name=f"{mode.lower().replace('/', '_').replace(' ', '_')}_schedule.csv",
            )

            # df = rows_to_df(rows, mode_specific_prefs)

            # with st.expander("Class details and explanations", expanded=False):
            #     st.dataframe(df, use_container_width=True)

            # st.download_button(
            #     f"Download {mode} schedule CSV",
            #     df.to_csv(index=False),
            #     file_name=f"{mode.lower().replace('/', '_').replace(' ', '_')}_schedule.csv",
            # )

st.divider()

with st.expander("App summary and parsed transcript", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Courses loaded", len(courses_enriched))
    col2.metric("Completed courses", len(completed))
    col3.metric("Parser confidence", prefs.get("parser_confidence", 0))

    if transcript_file is not None and not completed_df.empty:
        st.subheader("Parsed from transcript")
        st.dataframe(completed_df, use_container_width=True)

