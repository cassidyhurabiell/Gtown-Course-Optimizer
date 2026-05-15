"""
Georgetown Schedule CSV Cleaner

Takes a raw Georgetown Schedule of Classes CSV and creates:
1. clean_schedule_sections.csv  - one row per course section
2. clean_schedule_meetings.csv  - one row per meeting block

Usage in PowerShell:
    python clean_georgetown_schedule.py "your_raw_schedule.csv"

Optional output folder:
    python clean_georgetown_schedule.py "your_raw_schedule.csv" --out cleaned_output
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DAY_MAP = {
    "Monday": "M",
    "Tuesday": "T",
    "Wednesday": "W",
    "Thursday": "Th",
    "Friday": "F",
    "Saturday": "Sa",
    "Sunday": "Su",
}

POSSIBLE_COLUMNS = {
    "title": ["Title", "Course Title", "Class Title"],
    "subject": ["Subject", "Subj", "Department"],
    "course_number": ["Course Number", "Catalog Number", "Number", "Course"],
    "section": ["Section", "Section Number", "Sec"],
    "crn": ["CRN", "Class Number"],
    "credit_hours": ["Credit Hours", "Credits", "Units"],
    "instructor": ["Instructor", "Instructors", "Faculty"],
    "meeting_details": ["Meeting Details", "Meetings", "Meeting Pattern"],
    "campus": ["Campus"],
    "instructional_methods": ["Instructional Methods", "Instruction Method", "Modality"],
    "enrollment_status": ["Enrollment Status", "Enrollment"],
    "reserved_seats": ["Reserved Seats", "Reserved"],
    "attribute": ["Attribute", "Attributes"],
    "linked_sections": ["Linked Sections", "Linked Section"],
}


def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error: Exception | None = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e

    raise last_error or ValueError(f"Could not read {path}")


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for name in possible_names:
        key = name.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def clean_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_number_string(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return text


def clean_float(value: Any) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return 0.0

    try:
        return float(value)
    except ValueError:
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else 0.0


def clean_instructor(raw: Any) -> str:
    text = clean_string(raw)
    if not text:
        return ""

    text = text.replace("(Primary)", "")
    parts = [p.strip() for p in re.split(r"\n|;", text) if p.strip()]
    return "; ".join(parts)


def parse_meeting_details(raw: Any) -> list[dict[str, Any]]:
    text = clean_string(raw)
    if not text:
        return []

    text = re.sub(
        r"\n?S\s*\n+\s*M\s*\n+\s*T\s*\n+\s*W\s*\n+\s*T\s*\n+\s*F\s*\n+\s*S\s*\n?",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    day_names = "|".join(DAY_MAP.keys())

    block_pattern = re.compile(
        rf"(?P<days>(?:{day_names})(?:,(?:{day_names}))*|None)\s+"
        rf"(?P<time>\d{{1,2}}:\d{{2}}\s*[AP]M\s*-\s*\d{{1,2}}:\d{{2}}\s*[AP]M|-\s*)\s*"
        rf"Type:\s*(?P<type>.*?)\s+"
        rf"Building:\s*(?P<building>.*?)\s+"
        rf"Room:\s*(?P<room>.*?)\s+"
        rf"Start Date:\s*(?P<start>\d{{2}}/\d{{2}}/\d{{4}})\s+"
        rf"End Date:\s*(?P<end>\d{{2}}/\d{{2}}/\d{{4}})",
        flags=re.IGNORECASE | re.DOTALL,
    )

    meetings: list[dict[str, Any]] = []

    for match in block_pattern.finditer(text):
        days_raw = match.group("days").strip()
        time_raw = match.group("time").strip()

        if days_raw.lower() == "none":
            day_codes: list[str] = []
        else:
            day_codes = [DAY_MAP[d.strip()] for d in days_raw.split(",") if d.strip() in DAY_MAP]

        start_time = ""
        end_time = ""
        if "-" in time_raw and time_raw.strip() != "-":
            pieces = [p.strip() for p in time_raw.split("-", 1)]
            if len(pieces) == 2:
                start_time, end_time = pieces

        start_date = pd.to_datetime(match.group("start"), errors="coerce")
        end_date = pd.to_datetime(match.group("end"), errors="coerce")

        meetings.append(
            {
                "days_raw": days_raw if days_raw.lower() != "none" else "",
                "day_codes": ",".join(day_codes),
                "start_time": start_time,
                "end_time": end_time,
                "meeting_type": clean_string(match.group("type")),
                "building": clean_string(match.group("building")),
                "room": clean_string(match.group("room")),
                "start_date": "" if pd.isna(start_date) else start_date.date().isoformat(),
                "end_date": "" if pd.isna(end_date) else end_date.date().isoformat(),
                "has_meeting": bool(day_codes and start_time and end_time),
            }
        )

    return meetings


def classify_term_part(start_date: str, end_date: str) -> str:
    if not start_date or not end_date:
        return ""

    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")

    if pd.isna(start) or pd.isna(end):
        return ""

    duration_days = (end - start).days

    if duration_days >= 100:
        return "full_term"

    if start.month in {8, 9} and end.month in {10}:
        return "mod_a"
    if start.month in {10} and end.month in {12}:
        return "mod_b"

    if start.month in {1} and end.month in {2, 3}:
        return "mod_a"
    if start.month in {3} and end.month in {4, 5}:
        return "mod_b"

    return "partial_term"


def normalize_courses(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped: dict[str, pd.Series] = {}

    for target, aliases in POSSIBLE_COLUMNS.items():
        col = find_column(raw_df, aliases)
        if col is None:
            mapped[target] = pd.Series([""] * len(raw_df), index=raw_df.index)
        else:
            mapped[target] = raw_df[col]

    section_rows: list[dict[str, Any]] = []
    meeting_rows: list[dict[str, Any]] = []

    for idx in raw_df.index:
        title = clean_string(mapped["title"].loc[idx])
        subject = clean_string(mapped["subject"].loc[idx]).upper()
        course_number = clean_number_string(mapped["course_number"].loc[idx])
        section = clean_number_string(mapped["section"].loc[idx])
        crn = clean_number_string(mapped["crn"].loc[idx])
        credit_hours = clean_float(mapped["credit_hours"].loc[idx])
        instructor_raw = clean_string(mapped["instructor"].loc[idx])
        instructor_clean = clean_instructor(instructor_raw)
        meeting_details = clean_string(mapped["meeting_details"].loc[idx])

        course_id = f"{subject}-{course_number}" if subject and course_number else ""
        section_id = f"{course_id}-{section}" if course_id and section else course_id

        meetings = parse_meeting_details(meeting_details)

        if not meetings:
            meetings = [
                {
                    "days_raw": "",
                    "day_codes": "",
                    "start_time": "",
                    "end_time": "",
                    "meeting_type": "",
                    "building": "",
                    "room": "",
                    "start_date": "",
                    "end_date": "",
                    "has_meeting": False,
                }
            ]

        all_day_codes = sorted(
            {code for m in meetings for code in str(m.get("day_codes", "")).split(",") if code}
        )
        all_days_raw = "; ".join([m["days_raw"] for m in meetings if m.get("days_raw")])
        start_dates = [m["start_date"] for m in meetings if m.get("start_date")]
        end_dates = [m["end_date"] for m in meetings if m.get("end_date")]

        section_start = min(start_dates) if start_dates else ""
        section_end = max(end_dates) if end_dates else ""
        term_part = classify_term_part(section_start, section_end)

        base = {
            "title": title,
            "subject": subject,
            "course_number": course_number,
            "section": section,
            "crn": crn,
            "credit_hours": credit_hours,
            "instructor_raw": instructor_raw,
            "instructor_clean": instructor_clean,
            "campus": clean_string(mapped["campus"].loc[idx]),
            "instructional_methods": clean_string(mapped["instructional_methods"].loc[idx]),
            "enrollment_status": clean_string(mapped["enrollment_status"].loc[idx]),
            "reserved_seats": clean_string(mapped["reserved_seats"].loc[idx]),
            "attribute": clean_string(mapped["attribute"].loc[idx]),
            "linked_sections": clean_string(mapped["linked_sections"].loc[idx]),
            "course_id": course_id,
            "section_id": section_id,
        }

        section_row = {
            **base,
            "meeting_count": len([m for m in meetings if m.get("has_meeting")]),
            "days_raw": all_days_raw,
            "day_codes": ",".join(all_day_codes),
            "start_date": section_start,
            "end_date": section_end,
            "term_part": term_part,
            "has_meeting": any(bool(m.get("has_meeting")) for m in meetings),
            "meetings_json": json.dumps(meetings, ensure_ascii=False),
            "search_text": " ".join(
                [
                    title,
                    subject,
                    course_number,
                    instructor_clean,
                    clean_string(mapped["attribute"].loc[idx]),
                    clean_string(mapped["instructional_methods"].loc[idx]),
                ]
            ).lower(),
        }
        section_rows.append(section_row)

        for meeting_index, meeting in enumerate(meetings, start=1):
            meeting_rows.append(
                {
                    **base,
                    "meeting_index": meeting_index,
                    **meeting,
                    "term_part": classify_term_part(
                        meeting.get("start_date", ""),
                        meeting.get("end_date", ""),
                    ),
                }
            )

    sections_df = pd.DataFrame(section_rows)
    meetings_df = pd.DataFrame(meeting_rows)

    return sections_df, meetings_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to raw Georgetown Schedule of Classes CSV")
    parser.add_argument(
        "--out",
        default="cleaned_schedule_output",
        help="Output folder. Default: cleaned_schedule_output",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_csv_flexible(input_path)
    sections_df, meetings_df = normalize_courses(raw_df)

    sections_path = output_dir / "clean_schedule_sections.csv"
    meetings_path = output_dir / "clean_schedule_meetings.csv"

    sections_df.to_csv(sections_path, index=False, encoding="utf-8-sig")
    meetings_df.to_csv(meetings_path, index=False, encoding="utf-8-sig")

    print(f"Read {len(raw_df)} raw rows")
    print(f"Wrote {len(sections_df)} section rows to: {sections_path}")
    print(f"Wrote {len(meetings_df)} meeting rows to: {meetings_path}")
    print()
    print("Preview columns:")
    print(", ".join(sections_df.columns))


if __name__ == "__main__":
    main()
