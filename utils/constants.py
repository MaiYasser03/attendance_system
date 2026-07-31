import os
import re

CSV_PATH = os.getenv("ATTENDANCE_CSV", "attendance.csv")

NAME_RE = re.compile(
    r"(?:name|nama|nane|nom[e]?|nam[e]?|na me)[\s:\-]*([A-Z][A-Za-z ]{2,}?)(?=\s+(?:id|number|no|nr|nim|student|faculty|course|dob|date|birth)\b|\s*\d|$)",
    re.I,
)
ID_RE = re.compile(r"\b(\d{8,12})\b")
NAME_LABEL_CORRECTIONS = [
    (re.compile(r"\bNane\b", re.I), "Name"),
    (re.compile(r"\bNam[e]?\b", re.I), "Name"),
    (re.compile(r"\bDiumber\b", re.I), "Number"),
    (re.compile(r"\bNumebr\b", re.I), "Number"),
    (re.compile(r"\bNo\b", re.I), "No"),
]
