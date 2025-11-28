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
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


# ======================
#  協助判斷的小函式
# ======================

def is_section_header(line: str) -> bool:
    clean = line.strip()
    return clean.isupper() and 3 <= len(clean) <= 40


def is_bullet(line: str) -> bool:
    clean = line.strip()
    return clean.startswith("•") or clean.startswith("- ")


def is_role_title(line: str) -> bool:
    # 依照你履歷常見職稱關鍵字
    keywords = [
        "Intern", "Assistant", "Research", "Engineer", "Scientist",
        "Developer", "Associate", "Fellow"
    ]
    return any(k in line for k in keywords)


# ======================
#  Experience / Projects 解析：section + entry + bullet
# ======================

def parse_resume_entries(text: str):
    """
    解析出經驗型的 bullets，輸出 list of dict：
    {
        "section": "EXPERIENCE" / "PROJECTS",
        "entry": "CAYIN Technology — AI Engineering Intern ...",
        "text":  "某一條 bullet"
    }
    """
    lines = [l.rstrip() for l in text.split("\n") if l.strip()]

    results = []
    current_section = None
    current_entry = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 1) SECTION header
        if is_section_header(line):
            current_section = line
            current_entry = None
            i += 1
            continue

        # 2) EXPERIENCE entry: company + role (兩行)
        if (
            current_section == "EXPERIENCE"
            and i + 1 < len(lines)
            and not is_bullet(line)
            and not is_section_header(line)
            and not is_bullet(lines[i+1])
            and is_role_title(lines[i+1])
        ):
            company = line
            role = lines[i+1].strip()
            current_entry = f"{company} — {role}"
            i += 2
            continue

        # 2b) PROJECTS entry: 一行標題，下一行是 bullet
        if (
            current_section == "PROJECTS"
            and current_entry is None
            and not is_bullet(line)
            and not is_section_header(line)
            and i + 1 < len(lines)
            and is_bullet(lines[i+1])
        ):
            current_entry = line  # e.g. "Financial Argument Mining with LLMs, NTU Spring 2024"
            i += 1
            continue

        # 3) BULLET + 多行合併
        if is_bullet(line):
            bullet = line.lstrip("•- ").strip()
            j = i + 1

            while j < len(lines):
                nxt = lines[j].strip()

                # 下一行是新的 bullet 或 section → 結束
                if is_bullet(nxt) or is_section_header(nxt):
                    break

                # 下一行長得像新的 EXPERIENCE entry → 結束
                if (
                    j + 1 < len(lines)
                    and not is_bullet(nxt)
                    and is_role_title(lines[j+1])
                ):
                    break

                # 其他情況：當作續行
                bullet += " " + nxt
                j += 1

            results.append({
                "section": current_section,
                "entry": current_entry,
                "text": bullet
            })

            i = j
            continue

        # 4) 其他普通行略過
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

            # 不把 EXPERIENCE/PROJECTS 納入 metadata
            if up in ["EXPERIENCE", "PROJECTS"]:
                current = None
                continue

            if up in metadata:
                current = up
                continue

            if up in ["EDUCATION", "SKILLS"]:
                current = up
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
    - degree (含年限)
    - gpa
    - courses (list of str)
    回傳 list[dict]，每間學校一個 dict。
    """
    metadata = extract_metadata_sections(text)
    edu_text = metadata.get("EDUCATION", "")

    lines = [l.strip() for l in edu_text.split("\n") if l.strip()]

    schools = []
    current_school = {}

    date_pattern = re.compile(r"(20\d{2}).*(20\d{2}|expected)", re.IGNORECASE)
    gpa_pattern = re.compile(r"GPA\s*([0-9]\.[0-9])", re.IGNORECASE)

    for line in lines:

        # 1) 判斷新學校：包含 University 或 NTU
        if "University" in line or "NTU" in line:
            if current_school:
                schools.append(current_school)

            current_school = {
                "school_name": line,
                "degree": None,
                "major": None,   # 目前先不拆 major，之後可以再加
                "location": None,
                "dates": None,
                "gpa": None,
                "courses": []
            }
            continue

        # 2) Degree + dates + GPA（例如 MS / BS 那行）
        if "Master" in line or "BS" in line:
            deg_line = line
            current_school["degree"] = deg_line

            m = date_pattern.search(line)
            if m:
                current_school["dates"] = m.group(0)

            gpa_m = gpa_pattern.search(line)
            if gpa_m:
                current_school["gpa"] = gpa_m.group(1)

            continue

        # 3) Courses: 行
        if line.startswith("Courses:"):
            courses_str = line.replace("Courses:", "").strip()
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
