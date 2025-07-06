import re

CSV_PATH = 'attendance.csv'
NAME_RE = re.compile(r"name[:\s]*([a-zA-Z ]+?)(?=\s*id\b|$)", re.I)
ID_RE = re.compile(r'(\d{8,10})')
