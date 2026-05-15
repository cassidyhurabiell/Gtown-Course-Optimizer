import re
import pandas as pd


def infer_group_rule(group_name: str, rows: pd.DataFrame) -> dict:
    text = str(group_name).lower()

    credit_match = re.search(r"(\d+)\s*credit", text)
    if credit_match:
        return {"type": "credits", "n": float(credit_match.group(1))}

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }

    for word, n in number_words.items():
        if word in text and "course" in text:
            return {"type": "choose_n_courses", "n": n}

    return {"type": "all_required", "n": len(rows)}


def calculate_unmet_requirements(requirements: pd.DataFrame, completed: set[str]) -> pd.DataFrame:
    if requirements.empty:
        return pd.DataFrame()

    rows = []

    for col in ["major", "requirement_group", "course_id", "credits"]:
        if col not in requirements.columns:
            requirements[col] = ""

    for (major, group), group_df in requirements.groupby(["major", "requirement_group"], dropna=False):
        rule = infer_group_rule(group, group_df)

        exact = group_df[
            group_df["course_id"].astype(str).str.match(r"^[A-Z]{2,5}-\d{4}$", na=False)
        ].copy()

        completed_in_group = exact[exact["course_id"].isin(completed)]
        remaining_options = exact[~exact["course_id"].isin(completed)]

        if rule["type"] == "credits":
            completed_credits = (
                completed_in_group["credits"]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .sum()
            )
            still_needed = max(0, rule["n"] - completed_credits)

            if still_needed > 0:
                temp = remaining_options.copy()
                temp["requirement_status"] = f"choose {still_needed:g} more credit(s) from this group"
                temp["requirement_rule"] = "credits"
                temp["bucket_limit_credits"] = still_needed
                temp["bucket_limit_courses"] = None
                rows.append(temp)

        elif rule["type"] == "choose_n_courses":
            completed_count = completed_in_group["course_id"].nunique()
            still_needed = max(0, rule["n"] - completed_count)

            if still_needed > 0:
                temp = remaining_options.copy()
                temp["requirement_status"] = f"choose {still_needed:g} more course(s) from this group"
                temp["requirement_rule"] = "choose_n_courses"
                temp["bucket_limit_credits"] = None
                temp["bucket_limit_courses"] = still_needed
                rows.append(temp)

        else:
            unmet = remaining_options.copy()
            unmet["requirement_status"] = "required course"
            unmet["requirement_rule"] = "all_required"

            # Required courses are not a choose-one bucket.
            # Example: COSC-1020, COSC-1030, COSC-1110, COSC-2010 are all required.
            unmet["bucket_limit_credits"] = None
            unmet["bucket_limit_courses"] = None

            rows.append(unmet)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def attach_requirement_metadata(courses: pd.DataFrame, unmet_requirements: pd.DataFrame) -> pd.DataFrame:
    courses = courses.copy()

    for col in [
        "requirement_group",
        "requirement_rule",
        "bucket_limit_credits",
        "bucket_limit_courses",
        "requirement_status",
        "requirement_major",
    ]:
        courses[col] = ""

    if unmet_requirements.empty:
        return courses

    keep_cols = [
        c for c in [
            "major",
            "course_id",
            "requirement_group",
            "requirement_rule",
            "bucket_limit_credits",
            "bucket_limit_courses",
            "requirement_status",
        ]
        if c in unmet_requirements.columns
    ]

    meta = unmet_requirements[keep_cols].drop_duplicates(subset=["course_id", "requirement_group"])

    merged = courses.merge(
        meta,
        on="course_id",
        how="left",
        suffixes=("", "_req"),
    )

    if "major" in merged.columns:
        merged["requirement_major"] = merged["major"].fillna("")
        merged = merged.drop(columns=["major"])

    for col in [
        "requirement_group",
        "requirement_rule",
        "bucket_limit_credits",
        "bucket_limit_courses",
        "requirement_status",
    ]:
        req_col = f"{col}_req"
        if req_col in merged.columns:
            merged[col] = merged[req_col].fillna(merged[col])
            merged = merged.drop(columns=[req_col])

    return merged