# resume_parser.py

import os
import json
import re
import argparse
import pdfplumber


# ======================
#  PDF 文字抽取
# ======================

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extracts text from PDF.
    Using layout=False often works better for single-column resumes
    to preserve reading order, but you can toggle to True if needed.
    """
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # layout=True helps preserve visual spacing, which is sometimes useful
            # but for standard parsing, raw stream is often easier if columns aren't an issue.
            page_text = page.extract_text(layout=False)
            if page_text:
                text.append(page_text)
    return "\n".join(text)


# ======================
#  協助判斷的小函式
# ======================

def is_section_header(line: str) -> bool:
    """
    Detects if a line is a major section header (e.g., "EXPERIENCE", "PROJECTS").
    Criteria: Uppercase, reasonable length, not a bullet.
    """
    clean = line.strip()
    # "WORK EXPERIENCE" or "PROJECTS" match this
    if clean.isupper() and 3 <= len(clean) <= 40 and not is_bullet(clean):
        return True
    return False


def is_bullet(line: str) -> bool:
    """
    Detects standard bullet points.
    Handles: •, -, ●, and common PDF extraction artifacts like \uf0b7
    """
    clean = line.strip()
    if not clean:
        return False

    # Check for standard bullets
    if clean.startswith("•") or clean.startswith("- ") or clean.startswith("●"):
        return True

    # Check for PDF extraction bullets (private use Unicode characters)
    # Common bullet characters in PDFs: \uf0b7, \uf0a7, \uf0d8, etc.
    first_char = clean[0]
    if ord(first_char) in range(0xF000, 0xF100):  # Private Use Area
        return True

    # Check for indented lines that start with common bullet words (as backup)
    # This helps with PDFs where bullets are converted to spaces
    if line.startswith("  ") or line.startswith("\t"):
        # Indented line - check if it looks like a bullet
        words = clean.split()
        if words and words[0][0].isupper():  # Starts with capital letter
            return True

    return False


def is_date_range(line: str) -> bool:
    """
    Detects date ranges to identify the start of a new entry.
    Matches: "Jun 2025 - Present", "2024-2025", "Oct 2024", "Sep 2020-Dec 2024"
    """
    # Regex to catch standard Month Year formats or "Present"
    date_pattern = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|\d{4}\s*-\s*\d{4}|\d{4}"
    return bool(re.search(date_pattern, line, re.IGNORECASE))


# ======================
#  Experience / Projects 解析：section + entry + bullet
# ======================

def parse_resume_entries(text: str):
    """
    Parses experience-type bullets from resume text.
    Returns list of dict:
    {
        "section": "EXPERIENCE" / "PROJECTS" / etc.,
        "entry": "Job Title (Date Range)",
        "text":  "A single bullet point"
    }

    Improved logic from dfparser_updated.ipynb:
    - Better date range detection
    - Smarter entry detection (title + date or title + bullet)
    - More robust multi-line bullet merging
    """
    # Split lines and remove pure whitespace
    lines = [l.rstrip() for l in text.split("\n") if l.strip()]

    results = []
    current_section = "UNCATEGORIZED"
    current_entry = "General"

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # ---------------------------------------------------------
        # 1. Detect SECTION HEADER
        # ---------------------------------------------------------
        if is_section_header(line):
            current_section = line
            current_entry = None  # Reset entry on new section
            i += 1
            continue

        # ---------------------------------------------------------
        # 2. Detect NEW ENTRY (Job or Project Title)
        # Logic: If we are in Experience/Projects, and the line is NOT a bullet,
        # AND the NEXT line looks like a Date or a Bullet, this line is a Title.
        # ---------------------------------------------------------
        is_new_entry = False

        # Check constraints to avoid false positives in random text
        # Use substring matching to catch all variants (WORK EXPERIENCE, PROJECT EXPERIENCE, etc.)
        if "EXPERIENCE" in current_section or "PROJECT" in current_section:
            if not is_bullet(line):
                # Look ahead 1 line
                if i + 1 < len(lines):
                    next_line = lines[i+1].strip()

                    # If next line is a date, this line is definitely a Title
                    if is_date_range(next_line):
                        is_new_entry = True

                    # If next line is a bullet, this line is a Title (Project w/o date on next line)
                    elif is_bullet(next_line):
                        is_new_entry = True

        if is_new_entry:
            current_entry = line

            # If the next line was a date, grab it for metadata and skip it
            if i + 1 < len(lines) and is_date_range(lines[i+1]):
                date_str = lines[i+1].strip()
                current_entry = f"{line} ({date_str})"
                i += 2  # Skip Title and Date
            else:
                i += 1  # Just skip Title
            continue

        # ---------------------------------------------------------
        # 3. Capture BULLETS
        # ---------------------------------------------------------
        if is_bullet(line):
            # Clean the bullet marker - remove standard bullets and PDF artifacts
            bullet_text = line.strip()

            # Remove standard bullet characters
            bullet_text = re.sub(r"^[•\-●]\s*", "", bullet_text)

            # Remove PDF private use area characters (0xF000-0xF0FF)
            # These are common bullet characters in PDFs like \uf0b7, \uf0a7, \uf0d8
            if bullet_text and ord(bullet_text[0]) in range(0xF000, 0xF100):
                bullet_text = bullet_text[1:].strip()

            bullet_text = bullet_text.strip()

            # Handle multi-line bullets (look ahead for continuation)
            j = i + 1
            while j < len(lines):
                nxt = lines[j].strip()

                # BREAK conditions (End of bullet):
                # 1. Next line is a new bullet
                # 2. Next line is a section header
                # 3. Next line is a date (start of new entry)
                if is_bullet(nxt) or is_section_header(nxt) or is_date_range(nxt):
                    break

                # If none of the above, it's a continuation line. Merge it.
                bullet_text += " " + nxt
                j += 1

            # Save the result
            results.append({
                "section": current_section,
                "entry": current_entry or "General",
                "text": bullet_text
            })

            i = j  # Jump parsing index to where we stopped
            continue

        # If line fits no category, skip it (usually address, noise, etc.)
        i += 1

    return results


# ======================
#  Metadata (EDUCATION / SKILLS / COURSES)
# ======================

def extract_metadata_sections(text: str):
    """
    抽出 EDUCATION / SKILLS / COURSES 等 metadata 區塊。
    - EDUCATION / SKILLS 依全大寫標題分段
    - 在 EDUCATION 區塊中，額外把所有 'Courses:' 行收集到 metadata['COURSES']
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    metadata = {
        "EDUCATION": "",
        "SKILLS": "",
        "COURSES": "",
        "OTHER": ""
    }

    current = None

    for line in lines:
        # 全大寫 section header
        if line.isupper() and len(line) >= 3:
            up = line.upper()

            # 不把 EXPERIENCE/PROJECTS 納入 metadata (including variants)
            if "EXPERIENCE" in up or "PROJECT" in up:
                current = None
                continue

            if up in metadata:
                current = up
                continue

            # Use substring matching to catch variants like "TECHNICAL SKILLS", "CORE SKILLS"
            if up == "EDUCATION" or "SKILL" in up:
                current = "SKILLS" if "SKILL" in up else up
                continue

            current = "OTHER"
            continue

        # 在 EDUCATION 區塊內，特殊抓 Courses: 行
        if current == "EDUCATION" and line.lower().startswith("courses:"):
            if metadata["COURSES"]:
                metadata["COURSES"] += "\n" + line
            else:
                metadata["COURSES"] = line

        # 一般情況：此行歸屬當前 section
        if current:
            if metadata[current]:
                metadata[current] += "\n" + line
            else:
                metadata[current] = line

    return metadata


# ======================
#  EDUCATION 結構化解析
# ======================

def extract_structured_education(text: str):
    """
    從 EDUCATION 區塊中抓出：
    - school_name
    - degree (e.g., "M.S. in Data Science", "B.S. in Data Science")
    - major
    - dates
    - gpa
    - courses (list of str)
    回傳 list[dict]，每間學校一個 dict。

    Handles formats like:
    - "M.S. in Data Science Dec 2026(Expected)"
    - "B.S. in Data Science, GPA: 3.8/4.0 Sep 2020 - Dec 2024"
    - "Relevant Coursework: ..."
    - "Coursework: ..."
    """
    metadata = extract_metadata_sections(text)
    edu_text = metadata.get("EDUCATION", "")

    lines = [l.strip() for l in edu_text.split("\n") if l.strip()]

    schools = []
    current_school = {}

    # More flexible patterns
    date_pattern = re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}"
        r"|"
        r"\d{4}\s*-\s*\d{4}"
        r"|"
        r"(20\d{2}).*(20\d{2}|Expected|expected|Present)",
        re.IGNORECASE
    )
    gpa_pattern = re.compile(r"GPA[:\s]*([0-9]\.[0-9]+(?:/[0-9]\.[0-9]+)?)", re.IGNORECASE)
    # Updated pattern to match both "B.S." and "BS" formats
    # Also matches "MS, Data Science" format (comma-separated)
    # Must match degree at START of line or as complete word
    degree_pattern = re.compile(
        r"^(M\.?S\.?|M\.?A\.?|B\.?S\.?|B\.?A\.?|Ph\.?D\.?|Master|Bachelor)"
        r"[\s,]+"
        r"(in|of|,)?\s*"
        r"([A-Za-z\s&]+)",
        re.IGNORECASE
    )

    for line in lines:

        # 1) 判斷新學校：包含 University, College, Institute
        if any(word in line for word in ["University", "College", "Institute", "NTU"]):
            if current_school:
                schools.append(current_school)

            current_school = {
                "school_name": line,
                "degree": None,
                "major": None,
                "location": None,
                "dates": None,
                "gpa": None,
                "courses": []
            }
            continue

        # 2) Degree line (e.g., "M.S. in Data Science Dec 2026(Expected)" or "BS in Environmental Engineering, BS in Psychology, GPA 3.8/4.0")
        # Skip lines that start with "Major" - they'll be handled separately
        deg_match = degree_pattern.search(line)
        if deg_match and current_school and not line.startswith("Major"):
            # Extract only the degree part, stop before GPA or dates
            degree_text = line

            # Remove GPA and everything after it (both "GPA:" and "GPA " formats)
            degree_text = re.sub(r',?\s*GPA[:\s].*', '', degree_text, flags=re.IGNORECASE)

            # Remove dates (month + year pattern) and everything after
            degree_text = re.sub(
                r'\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}.*',
                '',
                degree_text,
                flags=re.IGNORECASE
            )

            # Remove year ranges like "Sep 2020 - Dec 2024"
            degree_text = re.sub(r'\s+\d{4}\s*-\s*\d{4}.*', '', degree_text)

            current_school["degree"] = degree_text.strip()

            # Extract major from the degree line
            # Try multiple patterns:

            # Pattern 1: "in/of [Major]" format (B.S. in Data Science)
            major_matches = re.findall(
                r"(?:in|of)\s+([A-Za-z\s]+?)(?=\s*,\s*(?:B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|Ph\.?D\.?)|,\s*GPA|$)",
                line,
                re.IGNORECASE
            )
            if major_matches:
                current_school["major"] = ", ".join([m.strip() for m in major_matches])
            else:
                # Pattern 2: Comma-separated format (MS, Data Science)
                comma_match = re.search(
                    r"^(?:M\.?S\.?|M\.?A\.?|B\.?S\.?|B\.?A\.?|Ph\.?D\.?),\s+([A-Za-z\s&]+?)(?:\s+(?:Expected|Sep|Oct|Nov|Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|\d{4})|$)",
                    line,
                    re.IGNORECASE
                )
                if comma_match:
                    current_school["major"] = comma_match.group(1).strip()
                else:
                    # Pattern 3: Fallback to "in/of" without strict boundaries
                    major_match = re.search(
                        r"(?:in|of)\s+([A-Za-z\s,]+?)(?:\s*,|\s*Dec|\s*Jan|\s*Feb|\s*Mar|\s*Apr|\s*May|\s*Jun|\s*Jul|\s*Aug|\s*Sep|\s*Oct|\s*Nov|\s*\d{4}|GPA|$)",
                        line,
                        re.IGNORECASE
                    )
                    if major_match:
                        current_school["major"] = major_match.group(1).strip()

            # Extract dates
            date_m = date_pattern.search(line)
            if date_m:
                current_school["dates"] = date_m.group(0)

            # Extract GPA (case insensitive)
            gpa_m = gpa_pattern.search(line)
            if gpa_m:
                current_school["gpa"] = gpa_m.group(1)

            continue

        # 3) Major line (if separate from degree)
        # Only process if we already have a degree (to avoid treating it as a degree)
        if current_school and current_school.get("degree") and line.startswith("Major"):
            # Extract major info from "Major in X, Minor in Y" format
            major_line = line.replace("Major in", "").replace("Major:", "").strip()
            # If it's already set from the degree line, don't overwrite unless this is more detailed
            if not current_school["major"] or len(major_line) > len(current_school["major"]):
                current_school["major"] = major_line
            continue

        # 4) Coursework lines (case insensitive check)
        # Matches: "Relevant Coursework:", "Coursework:", "Courses:", "Relevant Courses:"
        if current_school and re.search(r"(Relevant\s+)?(Coursework|Courses):", line, re.IGNORECASE):
            courses_str = re.sub(r"^(Relevant\s+)?(Coursework|Courses):\s*", "", line, flags=re.IGNORECASE).strip()
            if courses_str:
                current_school["courses"].append(courses_str)
            continue

    if current_school:
        schools.append(current_school)

    return schools


# ======================
#  CLI 執行入口
# ======================
def parse_resume_to_bullets(pdf_path: str):
    """
    給 FastAPI 後端使用的介面：
    輸入 PDF 路徑，回傳 experience/project 的 bullets list。
    格式：
    [
      {
        "section": "EXPERIENCE",
        "entry": "CAYIN Technology — AI Engineering Intern ...",
        "text": "某條 bullet"
      },
      ...
    ]
    """
    # 1) 讀 PDF 文字
    raw_text = extract_pdf_text(pdf_path)

    # 2) 解析經驗型 bullets (EXPERIENCE / PROJECTS)
    entries = parse_resume_entries(raw_text)

    return entries

def parse_resume_all(pdf_path: str):
    """
    給 FastAPI / 其他 Python code 用的入口：
    輸入一個 PDF 路徑，回傳一個 dict，包含：
    - entries: EXPERIENCE / PROJECTS 的 bullets
    - metadata: EDUCATION / SKILLS / COURSES / OTHER
    - education_structured: 結構化的學歷資訊
    """
    # 1) 讀 PDF 文字
    raw_text = extract_pdf_text(pdf_path)

    # 2) 解析經驗型 bullets
    entries = parse_resume_entries(raw_text)

    # 3) 解析 metadata
    metadata = extract_metadata_sections(raw_text)

    # 4) 結構化 EDUCATION
    education_structured = extract_structured_education(raw_text)

    return {
        "entries": entries,
        "metadata": metadata,
        "education_structured": education_structured,
        "raw_text": raw_text,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Parse resume PDF and export structured JSON for RAG."
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        required=True,
        help="Path to the resume PDF file."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save parsed JSON outputs."
    )

    args = parser.parse_args()

    pdf_path = args.pdf_path
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # 🔹 改成用你上面寫好的高階函式
    result = parse_resume_all(pdf_path)

    entries = result["entries"]
    metadata = result["metadata"]
    education_structured = result["education_structured"]

    # 5) 輸出 JSON 到 out_dir
    entries_path = os.path.join(out_dir, "experience_entries.json")
    metadata_path = os.path.join(out_dir, "metadata.json")
    edu_struct_path = os.path.join(out_dir, "education_structured.json")

    with open(entries_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(edu_struct_path, "w", encoding="utf-8") as f:
        json.dump(education_structured, f, ensure_ascii=False, indent=2)

    print(f"[OK] Parsed entries saved to: {entries_path}")
    print(f"[OK] Metadata saved to:      {metadata_path}")
    print(f"[OK] Education saved to:     {edu_struct_path}")

if __name__ == "__main__":
    main()
