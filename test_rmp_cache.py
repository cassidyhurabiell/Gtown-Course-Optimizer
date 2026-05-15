import pandas as pd
from rmp_scraper import enrich_with_rmp

test_courses = pd.DataFrame([
    {
        "course_id": "MGMT-2220",
        "title": "Foundations of Entrepreneurship",
        "instructor_clean": "Kanze, Dana",
    },
    {
        "course_id": "FINC-3104",
        "title": "Investments",
        "instructor_clean": "Liang, Claire",
    },
    {
        "course_id": "TEST-0000",
        "title": "Unknown Professor Test",
        "instructor_clean": "Professor, Notincache",
    },
    {
    "course_id": "TEST-0001",
    "title": "Scraped Professor Test",
    "instructor_clean": "Aas, Sean",
    },
])

enriched = enrich_with_rmp(test_courses, use_web=False)

print(enriched[[
    "course_id",
    "title",
    "instructor_clean",
    "rmp_rating",
    "rmp_tags",
    "rmp_snippet",
    "rmp_status",
]])