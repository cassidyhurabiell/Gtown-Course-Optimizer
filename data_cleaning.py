from __future__ import annotations

import json
import re
import pandas as pd


POSSIBLE_COLUMNS = {
    "course_id": ["course_id", "course", "course code", "subject_catalog", "subject + number"],
    "subject": ["subject", "subj"],
    "course_number": ["course_number", "course number", "catalog", "number", "catalog number"],
    "title": ["title", "course_title", "course title", "name"],
    "section": ["section", "sec"],
    "crn": ["crn"],
    "credit_hours": ["credit_hours", "credit hours", "credits", "credit", "hours"],
    "instructor": ["instructor", "instructors", "faculty", "professor", "instructor_clean", "instructor_raw"],
    "days_raw": ["days_raw", "day_codes", "days", "meeting days", "day"],
    "start_time": ["start_time", "start", "begin time", "start time"],
    "end_time": ["end_time", "end", "end time"],
    "start_date": ["start_date", "start date"],
    "end_date": ["end_date", "end date"],
    "building": ["building", "bldg"],
    "room": ["room"],
    "description": ["description", "instructional_methods", "desc", "course description"],
    "attribute": ["attribute", "attributes", "requirement", "core"],
}


def _norm_col(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip().lower().replace("_", " "))


def _find_col(df: pd.DataFrame, target: str) -> str | None:
    norm_to_real = {_norm_col(c): c for c in df.columns}
    for option in POSSIBLE_COLUMNS[target]:
        key = _norm_col(option)
        if key in norm_to_real:
            return norm_to_real[key]
    return None


def _read_csv(path_or_file) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path_or_file, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path_or_file, encoding="latin1")


def _clean_credits(x) -> float:
    if pd.isna(x):
        return 0.0
    m = re.search(r"\d+(?:\.\d+)?", str(x))
    return float(m.group(0)) if m else 0.0


def _clean_instructor(x) -> str:
    if pd.isna(x):
        return ""
    s = re.sub(r"\s+", " ", str(x)).strip()
    s = s.replace("(Primary)", "")
    s = s.replace("TBA", "").replace("TBD", "")
    return s.strip(" ,;")


def _parse_course_id(row: pd.Series) -> str:
    existing = row.get("course_id", "")
    if pd.notna(existing) and str(existing).strip():
        return str(existing).strip().upper().replace(" ", "-")

    subj = str(row.get("subject", "")).strip().upper()
    num = str(row.get("course_number", "")).strip()

    if re.fullmatch(r"\d+\.0", num):
        num = num[:-2]

    if subj and num and subj != "NAN" and num != "NAN":
        return f"{subj}-{num}".replace(" ", "")

    return ""


def _first_meeting_value(meetings_json, key: str) -> str:
    if pd.isna(meetings_json) or not str(meetings_json).strip():
        return ""

    try:
        meetings = json.loads(str(meetings_json))
    except Exception:
        return ""

    if not meetings:
        return ""

    value = meetings[0].get(key, "")
    return "" if pd.isna(value) else str(value).strip()


def _finish_course_frame(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()

    for col in [
        "course_id", "subject", "course_number", "title", "section", "crn",
        "credit_hours", "instructor", "days_raw", "start_time", "end_time",
        "start_date", "end_date", "building", "room", "description", "attribute"
    ]:
        if col not in out.columns:
            out[col] = ""

    out["course_id"] = out.apply(_parse_course_id, axis=1)
    out["title"] = out["title"].fillna("").astype(str).str.strip()
    out["credit_hours"] = out["credit_hours"].apply(_clean_credits)
    out["instructor_clean"] = out["instructor"].apply(_clean_instructor)
    out["course_label"] = out["course_id"] + " - " + out["title"]

    for col in ["days_raw", "start_time", "end_time", "description", "attribute", "building", "room"]:
        out[col] = out[col].fillna("").astype(str)

    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")

    out = out[out["course_id"].astype(str).str.strip().ne("")]
    return out.reset_index(drop=True)


def load_courses(path_or_file) -> pd.DataFrame:
    df = _read_csv(path_or_file)

    # Already-cleaned CSV from clean_georgetown_schedule.py
    if "course_id" in df.columns and "meetings_json" in df.columns:
        out = df.copy()

        if "instructor" not in out.columns:
            if "instructor_clean" in out.columns:
                out["instructor"] = out["instructor_clean"]
            elif "instructor_raw" in out.columns:
                out["instructor"] = out["instructor_raw"]
            else:
                out["instructor"] = ""

        if "days_raw" not in out.columns and "day_codes" in out.columns:
            out["days_raw"] = out["day_codes"]

        if "description" not in out.columns:
            if "instructional_methods" in out.columns:
                out["description"] = out["instructional_methods"]
            else:
                out["description"] = ""

        for col in ["start_time", "end_time", "building", "room"]:
            if col not in out.columns:
                out[col] = out["meetings_json"].apply(lambda x, key=col: _first_meeting_value(x, key))

        return _finish_course_frame(out)

    # Raw Georgetown CSV
    out = pd.DataFrame(index=df.index)

    for target in POSSIBLE_COLUMNS:
        col = _find_col(df, target)
        out[target] = df[col] if col else ""

    return _finish_course_frame(out)


def load_requirements(path_or_file) -> pd.DataFrame:
    if path_or_file is None:
        return pd.DataFrame(columns=["major", "class_year", "course_id", "course_title", "priority"])

    df = pd.read_csv(path_or_file)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "program": "major",
        "concentration": "major",
        "requirement_program": "major",
        "name": "major",
        "title": "course_title",
        "course": "course_id",
        "course_code": "course_id",
        "course_number": "course_id",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    for c in ["major", "class_year", "course_id", "course_title", "priority"]:
        if c not in df.columns:
            df[c] = ""

    df["major"] = df["major"].fillna("").astype(str).str.strip()
    df["course_id"] = (
        df["course_id"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(" ", "-", regex=False)
    )

    return df






