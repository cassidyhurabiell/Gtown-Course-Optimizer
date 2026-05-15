#I know the names of the colors are messed up, that's just cause I completely changed the colorscheme post-variable name & using
GEORGETOWN_BLUE = "#041E42"
GEORGETOWN_GRAY = "#8A8D8F"
SOFT_PINK = "#5FFF56"
LIGHT_PINK = "#5D2F72"
BACKGROUND = "#FF36C9D7"

CARD_BACKGROUND = "#680581"
TEXT_DARK = "#1F2933"
BORDER = "#D8DEE8"

DAY_MAP = {
    "monday": "M", "mon": "M", "m": "M",
    "tuesday": "T", "tue": "T", "tues": "T", "t": "T",
    "wednesday": "W", "wed": "W", "w": "W",
    "thursday": "Th", "thu": "Th", "thur": "Th", "thurs": "Th", "th": "Th",
    "friday": "F", "fri": "F", "f": "F",
}

DEFAULT_PREFS = {
    "avoid_days": [],
    "prefer_days": [],
    "earliest_time": None,
    "latest_time": None,
    "weights": {
        "professor": 2,
        "compact": 2,
        "no_friday": 1,
        "morning": 1,
        "requirement_fit": 2,
        "interest_fit": 3,
        "personality_fit": 2,
    },
    "interests": [],
    "goals": [],
    "personality": [],
    "notes": "",
}