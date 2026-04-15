from typing import Optional, Dict, List, Any
import difflib

CAMPUS_LOCATIONS = {
    "city campus library": {
        "name": "City Campus Library",
        "address": "Sandyford Road, Newcastle upon Tyne, NE1 8ST",
        "campus": "City Campus",
        "coordinates": {"lat": 54.9767, "lng": -1.6131},
        "directions_hint": "Enter via the main Sandyford Road entrance. Opposite the Ellison Building."
    },
    "ellison building": {
        "name": "Ellison Building",
        "address": "Ellison Place, Newcastle upon Tyne, NE1 8ST",
        "campus": "City Campus",
        "coordinates": {"lat": 54.9763, "lng": -1.6128},
        "directions_hint": "Main entrance on Ellison Place. Home to Business and Law faculty."
    },
    "northumberland building": {
        "name": "Northumberland Building",
        "address": "Northumberland Road, Newcastle upon Tyne, NE1 8ST",
        "campus": "City Campus",
        "coordinates": {"lat": 54.9771, "lng": -1.6135},
        "directions_hint": "On Northumberland Road, next to the Students Union."
    },
    "students union": {
        "name": "Students' Union",
        "address": "2 Sandyford Road, Newcastle upon Tyne, NE1 8SB",
        "campus": "City Campus",
        "coordinates": {"lat": 54.9765, "lng": -1.6140},
        "directions_hint": "Ground floor of the Northumberland Building complex."
    },
    "sport central": {
        "name": "Sport Central",
        "address": "Northumberland Road, Newcastle upon Tyne, NE1 8ST",
        "campus": "City Campus",
        "coordinates": {"lat": 54.9774, "lng": -1.6130},
        "directions_hint": "Large sports complex opposite the Northumberland Building."
    },
    "coach lane campus": {
        "name": "Coach Lane Campus",
        "address": "Coach Lane, Benton, Newcastle upon Tyne, NE7 7XA",
        "campus": "Coach Lane",
        "coordinates": {"lat": 55.0041, "lng": -1.5792},
        "directions_hint": "Accessible by Metro (Benton station). Home to Health and Education faculties."
    },
    "city campus east": {
        "name": "City Campus East",
        "address": "Sandyford Road, Newcastle upon Tyne, NE1 8QH",
        "campus": "City Campus",
        "coordinates": {"lat": 54.9760, "lng": -1.6115},
        "directions_hint": "Modern computing and engineering block east of main city campus."
    },
    "pandon building": {
        "name": "Pandon Building",
        "address": "Pandon, Newcastle upon Tyne, NE1 8ST",
        "campus": "City Campus",
        "coordinates": {"lat": 54.9755, "lng": -1.6108},
        "directions_hint": "Home to the Faculty of Arts, Design and Social Sciences."
    },
}

FACILITY_HOURS = {
    "city campus library": {
        "name": "City Campus Library",
        "hours": {
            "Monday": "8:00am - 11:00pm", "Tuesday": "8:00am - 11:00pm",
            "Wednesday": "8:00am - 11:00pm", "Thursday": "8:00am - 11:00pm",
            "Friday": "8:00am - 9:00pm", "Saturday": "10:00am - 6:00pm", "Sunday": "10:00am - 8:00pm"
        }
    },
    "sport central": {
        "name": "Sport Central",
        "hours": {
            "Monday": "6:30am - 10:00pm", "Tuesday": "6:30am - 10:00pm",
            "Wednesday": "6:30am - 10:00pm", "Thursday": "6:30am - 10:00pm",
            "Friday": "6:30am - 9:00pm", "Saturday": "8:00am - 8:00pm", "Sunday": "9:00am - 7:00pm"
        }
    },
    "students union": {
        "name": "Students' Union",
        "hours": {
            "Monday": "9:00am - 9:00pm", "Tuesday": "9:00am - 9:00pm",
            "Wednesday": "9:00am - 9:00pm", "Thursday": "9:00am - 9:00pm",
            "Friday": "9:00am - 11:00pm", "Saturday": "11:00am - 11:00pm", "Sunday": "Closed"
        }
    },
    "ellison building": {
        "name": "Ellison Building",
        "hours": {
            "Monday": "7:00am - 10:00pm", "Tuesday": "7:00am - 10:00pm",
            "Wednesday": "7:00am - 10:00pm", "Thursday": "7:00am - 10:00pm",
            "Friday": "7:00am - 9:00pm", "Saturday": "9:00am - 5:00pm", "Sunday": "Closed"
        }
    },
    "northumberland building": {
        "name": "Northumberland Building",
        "hours": {
            "Monday": "7:00am - 10:00pm", "Tuesday": "7:00am - 10:00pm",
            "Wednesday": "7:00am - 10:00pm", "Thursday": "7:00am - 10:00pm",
            "Friday": "7:00am - 9:00pm", "Saturday": "9:00am - 5:00pm", "Sunday": "Closed"
        }
    },
    "city campus east": {
        "name": "City Campus East",
        "hours": {
            "Monday": "7:00am - 10:00pm", "Tuesday": "7:00am - 10:00pm",
            "Wednesday": "7:00am - 10:00pm", "Thursday": "7:00am - 10:00pm",
            "Friday": "7:00am - 9:00pm", "Saturday": "9:00am - 5:00pm", "Sunday": "Closed"
        }
    },
    "coach lane campus": {
        "name": "Coach Lane Campus",
        "hours": {
            "Monday": "7:30am - 9:00pm", "Tuesday": "7:30am - 9:00pm",
            "Wednesday": "7:30am - 9:00pm", "Thursday": "7:30am - 9:00pm",
            "Friday": "7:30am - 7:00pm", "Saturday": "9:00am - 4:00pm", "Sunday": "Closed"
        }
    },
    "cafe": {
        "name": "Campus Cafe",
        "hours": {
            "Monday": "8:00am - 6:00pm", "Tuesday": "8:00am - 6:00pm",
            "Wednesday": "8:00am - 6:00pm", "Thursday": "8:00am - 6:00pm",
            "Friday": "8:00am - 5:00pm", "Saturday": "10:00am - 3:00pm", "Sunday": "Closed"
        }
    },
    "gym": {
        "name": "Campus Gym (Sport Central)",
        "hours": {
            "Monday": "6:30am - 10:00pm", "Tuesday": "6:30am - 10:00pm",
            "Wednesday": "6:30am - 10:00pm", "Thursday": "6:30am - 10:00pm",
            "Friday": "6:30am - 9:00pm", "Saturday": "8:00am - 8:00pm", "Sunday": "9:00am - 7:00pm"
        }
    },
    "library": {
        "name": "City Campus Library",
        "hours": {
            "Monday": "8:00am - 11:00pm", "Tuesday": "8:00am - 11:00pm",
            "Wednesday": "8:00am - 11:00pm", "Thursday": "8:00am - 11:00pm",
            "Friday": "8:00am - 9:00pm", "Saturday": "10:00am - 6:00pm", "Sunday": "10:00am - 8:00pm"
        }
    },
    "printing room": {
        "name": "Printing Room",
        "hours": {
            "Monday": "8:00am - 9:00pm", "Tuesday": "8:00am - 9:00pm",
            "Wednesday": "8:00am - 9:00pm", "Thursday": "8:00am - 9:00pm",
            "Friday": "8:00am - 7:00pm", "Saturday": "10:00am - 4:00pm", "Sunday": "Closed"
        }
    },
    "student services": {
        "name": "Student Services",
        "hours": {
            "Monday": "9:00am - 5:00pm", "Tuesday": "9:00am - 5:00pm",
            "Wednesday": "9:00am - 5:00pm", "Thursday": "9:00am - 5:00pm",
            "Friday": "9:00am - 4:30pm", "Saturday": "Closed", "Sunday": "Closed"
        }
    },
}

CAMPUS_EVENTS = [
    {"name": "Open Day", "time": "10:00am", "location": "Ellison Building", "date": "today", "type": "admissions"},
    {"name": "AI Research Seminar", "time": "2:00pm", "location": "City Campus East", "date": "today", "type": "academic"},
    {"name": "Freshers Fair", "time": "11:00am", "location": "Students' Union", "date": "tomorrow", "type": "social"},
    {"name": "Careers Workshop", "time": "3:00pm", "location": "Northumberland Building", "date": "today", "type": "careers"},
]

NEARBY_MAP = {
    "ellison building": ["City Campus Library", "Northumberland Building", "Sport Central", "Students' Union"],
    "city campus library": ["Ellison Building", "Northumberland Building", "Students' Union"],
    "northumberland building": ["Students' Union", "Ellison Building", "City Campus Library"],
}

def _fuzzy_match(query: str, keys: list):
    query = query.lower().strip()
    matches = difflib.get_close_matches(query, keys, n=1, cutoff=0.5)
    return matches[0] if matches else None

def get_location(name: str):
    key = _fuzzy_match(name, list(CAMPUS_LOCATIONS.keys()))
    return CAMPUS_LOCATIONS.get(key) if key else None

def get_facility_hours(name: str):
    key = _fuzzy_match(name, list(FACILITY_HOURS.keys()))
    return FACILITY_HOURS.get(key) if key else None

def get_nearby_facilities(reference, facility_type):
    if not reference:
        return []
    key = _fuzzy_match(reference, list(NEARBY_MAP.keys()))
    if not key:
        return []
    return [CAMPUS_LOCATIONS[_fuzzy_match(n, list(CAMPUS_LOCATIONS.keys()))] for n in NEARBY_MAP[key] if _fuzzy_match(n, list(CAMPUS_LOCATIONS.keys()))]

def get_events(time_ref: str, location):
    filtered = [e for e in CAMPUS_EVENTS if time_ref.lower() in e["date"] or e["date"] == "today"]
    if location:
        filtered = [e for e in filtered if location.lower() in e["location"].lower()] or filtered
    return filtered