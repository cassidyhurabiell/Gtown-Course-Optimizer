from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from config import DEFAULT_PREFS, DAY_MAP

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # app still runs without sklearn
    TfidfVectorizer = None
    cosine_similarity = None

INTENT_EXAMPLES = {
    "avoid_late": [
        "I want to be done early", "no night classes", "avoid evening classes",
        "I hate late classes", "nothing after dinner", "finish before evening",
        "I do not want classes late in the day",
    ],
    "avoid_early": [
        "I hate early mornings", "no 8 am classes", "avoid morning classes",
        "I do not want to wake up early", "nothing before 10 am",
    ],
    "prefer_morning": [
        "I like morning classes", "I want early classes", "front load my day",
        "I focus best in the morning",
    ],
    "compact": [
        "I want a compact schedule", "put classes close together", "back to back classes",
        "classes on fewer days", "not spread out", "cluster my classes",
    ],
    "professor_quality": [
        "great professors matter most", "best rated teachers", "I care about professor quality",
        "inspiring professors", "good teaching is important",
    ],
    "exploratory": [
        "I want to explore new subjects", "I am open minded", "I want interesting electives",
        "I want classes that expand my perspective",
    ],
    "career": [
        "I want practical career skills", "classes useful for internships", "job relevant courses",
        "skills for consulting finance startups technology policy",
    ],
}

INTEREST_SEEDS = [
    "ai", "artificial intelligence", "ethics", "government", "politics", "public policy",
    "linguistics", "language", "mythology", "religion", "philosophy", "history",
    "finance", "venture capital", "startups", "entrepreneurship", "marketing",
    "data", "coding", "computer science", "psychology", "sociology", "international relations",
    "law", "health", "medicine", "environment", "climate", "art", "literature", "writing",
]

PERSONALITY_WORDS = [
    "curious", "creative", "analytical", "social", "collaborative", "independent",
    "practical", "ambitious", "adventurous", "organized", "visual", "discussion based",
    "hands on", "entrepreneurial", "reflective", "mission driven",
]


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?;\n]+", text) if s.strip()]


@dataclass
class LocalIntentMatcher:
    threshold: float = 0.18

    def match(self, text: str) -> Dict[str, float]:
        if not text.strip():
            return {}
        if TfidfVectorizer is None:
            return self._keyword_fallback(text)

        examples = []
        labels = []
        for label, phrases in INTENT_EXAMPLES.items():
            for p in phrases:
                examples.append(p)
                labels.append(label)
        docs = examples + [text]
        vect = TfidfVectorizer(ngram_range=(1, 3), stop_words="english").fit_transform(docs)
        sims = cosine_similarity(vect[-1], vect[:-1]).flatten()
        scores: Dict[str, float] = {}
        for label, score in zip(labels, sims):
            scores[label] = max(scores.get(label, 0.0), float(score))
        return {k: v for k, v in scores.items() if v >= self.threshold}

    def _keyword_fallback(self, text: str) -> Dict[str, float]:
        t = text.lower()
        scores = {}
        checks = {
            "avoid_late": ["late", "night", "evening", "done early", "finish early"],
            "avoid_early": ["early morning", "8 am", "8am", "before 10", "hate mornings"],
            "prefer_morning": ["morning", "early classes", "front load"],
            "compact": ["compact", "back to back", "close together", "fewer days", "not spread out"],
            "professor_quality": ["professor", "teacher", "rated", "inspiring"],
        }
        for label, words in checks.items():
            if any(w in t for w in words):
                scores[label] = 0.5
        return scores


def _extract_days(text: str) -> Tuple[List[str], List[str]]:
    t = text.lower()
    avoid, prefer = set(), set()
    for word, code in DAY_MAP.items():
        if re.search(rf"\b(no|avoid|hate|free|off)\s+\w*\s*{re.escape(word)}s?\b", t) or re.search(rf"\b{re.escape(word)}s?\s+(off|free)\b", t):
            avoid.add(code)
        if re.search(rf"\b(prefer|like|want|best on)\s+\w*\s*{re.escape(word)}s?\b", t):
            prefer.add(code)
    if "fridays off" in t or "no fridays" in t or "avoid fridays" in t:
        avoid.add("F")
    return sorted(avoid), sorted(prefer)


def _extract_times(text: str) -> Tuple[str | None, str | None]:
    t = text.lower()
    earliest = None
    latest = None

    before_match = re.search(r"(?:nothing|no classes|avoid classes)?\s*(?:before|earlier than)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)
    after_match = re.search(r"(?:nothing|no classes|avoid classes)?\s*(?:after|later than)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t)

    def fmt(hour: str, minute: str | None, ampm: str | None) -> str:
        h = int(hour)
        m = minute or "00"
        if ampm is None:
            ampm = "PM" if h < 8 else "AM"
        return f"{h:02d}:{m} {ampm.upper()}"

    if before_match:
        earliest = fmt(*before_match.groups())
    if after_match:
        latest = fmt(*after_match.groups())

    if any(p in t for p in ["late", "night class", "evening class", "done early", "finish early"]):
        latest = latest or "05:00 PM"
    if any(p in t for p in ["early morning", "8 am", "8am", "hate mornings", "not early"]):
        earliest = earliest or "10:00 AM"

    return earliest, latest


def _extract_list_after_cues(text: str, cues: List[str]) -> List[str]:
    t = text.lower()
    items = []
    for cue in cues:
        pattern = rf"{cue}\s+([^.;\n]+)"
        m = re.search(pattern, t)
        if m:
            raw = re.split(r"\bbut\b|\balthough\b|\bwhile\b", m.group(1))[0]
            raw = raw.replace("&", ",").replace(" and ", ",")
            items.extend([x.strip(" ,") for x in raw.split(",") if x.strip(" ,")])
    return items


def _extract_interests(text: str) -> List[str]:
    t = text.lower()
    found = set()
    for seed in INTEREST_SEEDS:
        if re.search(rf"\b{re.escape(seed)}\b", t):
            found.add(seed)
    found.update(_extract_list_after_cues(t, ["interested in", "i like", "i enjoy", "curious about", "drawn to", "want to study"]))
    cleaned = []
    for item in found:
        item = re.sub(r"^(classes in|courses in|studying)\s+", "", item.strip())
        if 2 <= len(item) <= 45:
            cleaned.append(item)
    return sorted(set(cleaned))


def _extract_personality(text: str) -> List[str]:
    t = text.lower()
    found = [w for w in PERSONALITY_WORDS if w in t]
    found.extend(_extract_list_after_cues(t, ["i am", "i'm", "my personality is", "students describes me as"]))
    return sorted(set([x.strip() for x in found if 2 <= len(x.strip()) <= 45]))[:12]


def _extract_goals(text: str) -> List[str]:
    goals = _extract_list_after_cues(text, ["hoping to", "want to", "looking to", "my goal is", "i need to"])
    return sorted(set([g.strip() for g in goals if 3 <= len(g.strip()) <= 80]))[:10]


def parse_student_profile(text: str, class_year: str = "") -> Dict:
    prefs = copy.deepcopy(DEFAULT_PREFS)
    prefs["notes"] = text
    prefs["class_year"] = class_year

    avoid, prefer = _extract_days(text)
    prefs["avoid_days"] = avoid
    prefs["prefer_days"] = prefer

    earliest, latest = _extract_times(text)
    prefs["earliest_time"] = earliest
    prefs["latest_time"] = latest

    prefs["interests"] = _extract_interests(text)
    prefs["personality"] = _extract_personality(text)
    prefs["goals"] = _extract_goals(text)

    prefs.setdefault("professor_preferences", [])

    matches = LocalIntentMatcher().match(text)
    if "avoid_late" in matches:
        prefs["latest_time"] = prefs["latest_time"] or "05:00 PM"
    if "avoid_early" in matches:
        prefs["earliest_time"] = prefs["earliest_time"] or "10:00 AM"
        prefs["weights"]["morning"] = 0
    if "prefer_morning" in matches and "avoid_early" not in matches:
        prefs["weights"]["morning"] = 2
    if "compact" in matches:
        prefs["weights"]["compact"] = 3
    if "professor_quality" in matches:
        prefs["weights"]["professor"] = 3
    if "career" in matches:
        prefs["weights"]["requirement_fit"] = 3
    if "exploratory" in matches:
        prefs["weights"]["interest_fit"] = 3

    if "F" in prefs["avoid_days"]:
        prefs["weights"]["no_friday"] = 3
    
    text_lower = text.lower()

    if any(x in text_lower for x in ["easy professor", "easy grader", "not too hard", "low difficulty"]):
        prefs["professor_preferences"].append("easy")
        prefs["weights"]["professor"] = max(prefs["weights"].get("professor", 1), 4)

    if any(x in text_lower for x in ["cares", "caring", "supportive", "helpful", "kind", "understanding"]):
        prefs["professor_preferences"].append("supportive")
        prefs["weights"]["professor"] = max(prefs["weights"].get("professor", 1), 4)

    if any(x in text_lower for x in ["engaging", "interesting professor", "inspiring", "passionate"]):
        prefs["professor_preferences"].append("engaging")
        prefs["weights"]["professor"] = max(prefs["weights"].get("professor", 1), 4)

    prefs["professor_preferences"] = sorted(set(prefs["professor_preferences"]))

    confidence = min(1.0, 0.35 + 0.10 * len(matches) + 0.05 * len(prefs["interests"]))
    prefs["parser_confidence"] = round(confidence, 2)
    prefs["matched_intents"] = matches
    return prefs
