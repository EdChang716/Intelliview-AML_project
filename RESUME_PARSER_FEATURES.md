# Resume Parser Features & Improvements

This document describes the enhanced resume parsing capabilities implemented in `parsers/resume_parser.py`.

## Overview

The resume parser uses rule-based logic with smart pattern recognition to extract structured data from single-column PDF resumes. It's been enhanced to handle various resume formats and edge cases.

## Key Features

### 1. Experience & Projects Parsing

**Smart Entry Detection:**
- Detects new job/project entries by analyzing the next line (date or bullet)
- Works with any job title format - no hardcoded keywords needed
- Supports multiple section types:
  - `WORK EXPERIENCE`
  - `EXPERIENCE`
  - `PROJECTS`
  - `ACADEMIC PROJECTS`
  - `LEADERSHIP EXPERIENCE`

**Date Range Detection:**
- Flexible date formats:
  - `Sep 2025 - Present`
  - `Jun 2024 - Aug 2024`
  - `Oct 2024`
  - `2024-2025`
  - `Dec 2026(Expected)`

**Multi-line Bullet Merging:**
- Automatically merges continuation lines into single bullet points
- Preserves complete bullet text across multiple PDF lines

### 2. Education Parsing

**Degree Formats Supported:**
- With periods: `M.S.`, `B.S.`, `M.A.`, `B.A.`, `Ph.D.`
- Without periods: `MS`, `BS`, `MA`, `BA`, `PhD`
- Full words: `Master`, `Bachelor`

**Multiple Degrees on One Line:**
```
BS in Environmental Engineering, BS in Psychology, GPA 3.8/4.0
```
Correctly extracts:
- Degree: "BS in Environmental Engineering, BS in Psychology"
- Major: "Environmental Engineering, Psychology"
- GPA: "3.8/4.0"

**Degree Field Cleanup:**
Automatically removes GPA and dates from degree field:
```
Input:  "B.S. in Data Science, GPA: 3.8/4.0 Sep 2020 - Dec 2024"
Output: "B.S. in Data Science"
```

**Major Extraction:**
- Extracts from degree line: `B.S. in Data Science` → Major: "Data Science"
- Extracts from separate line: `Major in Data Science, Minor in Business Administration`
- Handles multiple majors from degree line

**Coursework Detection (Case-Insensitive):**
Supports all common variations:
- `Relevant Coursework:`
- `Coursework:`
- `Courses:`
- `Relevant Courses:`
- `RELEVANT COURSEWORK:` (uppercase)

**Complete Education Structure:**
```json
{
  "school_name": "Boston University Boston, MA",
  "degree": "B.S. in Data Science",
  "major": "Data Science, Minor in Business Administration",
  "dates": "Sep 2020",
  "gpa": "3.8/4.0",
  "courses": [
    "Bayesian Statistics, Computer Systems, Database Design..."
  ]
}
```

### 3. Frontend Integration

**Education Editor UI:**
The frontend now displays ALL education fields:
- School Name
- Degree/Program
- **Major/Field of Study** (NEW)
- Dates
- GPA
- **Relevant Coursework** (NEW - textarea)

**Dynamic Section Rendering:**
Frontend automatically displays all parsed sections without hardcoded filtering:
- Work Experience
- Projects
- Leadership Experience
- Any other section detected by parser

## Testing

Run the test script to verify parsing:
```bash
source venv/bin/activate
python test_parser.py
```

## Example Output

### Your Resume (resume_ML)
```
📋 EXPERIENCE & PROJECTS
  WORK EXPERIENCE: 11 bullets
  PROJECTS: 9 bullets
  LEADERSHIP EXPERIENCE: 2 bullets
  Total bullets captured: 22

🎓 EDUCATION (Structured)
  School 1: Columbia University
    Degree: M.S. in Data Science
    Major:  Data Science
    GPA:    Not detected
    Courses: 1 lines

  School 2: Boston University
    Degree: B.S. in Data Science
    Major:  Data Science, Minor in Business Administration
    GPA:    3.8/4.0
    Courses: 1 lines
```

### NTU Format (Multiple Degrees)
```
Input:
  BS in Environmental Engineering, BS in Psychology, GPA 3.8/4.0
  Relevant Coursework: Environmental Science, Data Analysis

Output:
  Degree: "BS in Environmental Engineering, BS in Psychology"
  Major:  "Environmental Engineering, Psychology"
  GPA:    "3.8/4.0"
  Courses: ["Environmental Science, Data Analysis"]
```

## Implementation Details

### Key Functions

**`parse_resume_entries(text)`**
- Returns list of `{section, entry, text}` dicts
- Uses `is_date_range()` for smart entry detection
- Handles multi-line bullets with lookahead

**`extract_structured_education(text)`**
- Returns list of education dicts with all fields
- Regex-based field extraction with cleanup
- Handles edge cases (Major line, multiple degrees, etc.)

**`is_date_range(line)`**
- Comprehensive regex for date detection
- Supports month-year, year ranges, "Present", "Expected"

### Frontend JavaScript

**`groupEntries(entries)`**
- Dynamically groups entries by section
- No hardcoded section names - adapts to any resume

**`renderEducationEditor()`**
- Renders all education fields including major and courses
- Courses displayed as comma-separated textarea

**`collectEducationFromEditor()`**
- Converts coursework textarea back to array
- Preserves all education structure fields

## Configuration

No configuration needed - parser automatically adapts to resume format.

**Supported resume types:**
- Single-column standard resumes
- US-style resumes with uppercase section headers
- International formats (tested with Taiwan NTU format)

**Not supported:**
- Multi-column resumes
- Graphic-heavy resumes
- Tables (may work depending on PDF structure)

## Files Modified

1. **`parsers/resume_parser.py`** - Core parsing logic
2. **`app/templates/index.html`** - Frontend education editor
3. **`test_parser.py`** - Validation script

## Troubleshooting

**Issue: Coursework not showing**
- Solution: Parser now supports "Coursework:", "Courses:", "Relevant Coursework:", "Relevant Courses:" (case-insensitive)

**Issue: Multiple degrees on one line not parsed**
- Solution: Parser now handles both "B.S." and "BS" formats and extracts all majors

**Issue: Degree includes GPA and dates**
- Solution: Parser now strips GPA and dates from degree field

**Issue: Frontend only shows projects**
- Solution: Frontend now uses `Object.keys()` to display all sections dynamically
