from __future__ import annotations

import io
import re
import pandas as pd

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

COURSE_RE = re.compile(r"\b([A-Z]{2,5})\s*[- ]?\s*(\d{3,5}[A-Z]?)\b")


def extract_text_from_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".csv"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("Install pypdf to read PDFs: python -m pip install pypdf")
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def parse_completed_courses(text: str, course_catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    seen = []
    for m in COURSE_RE.finditer(text or ""):
        course_id = f"{m.group(1).upper()}-{m.group(2).upper()}"
        if course_id not in seen:
            seen.append(course_id)
    df = pd.DataFrame({"course_id": seen})
    if course_catalog is not None and not course_catalog.empty and "course_id" in course_catalog.columns:
        titles = course_catalog[["course_id", "title"]].drop_duplicates()
        df = df.merge(titles, on="course_id", how="left")
    else:
        df["title"] = ""
    df["course_label"] = df["course_id"] + " — " + df["title"].fillna("")
    return df
