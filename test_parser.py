#!/usr/bin/env python3
"""
Test script to verify resume parsing improvements
"""
import json
from pathlib import Path
from parsers.resume_parser import parse_resume_all

def test_resume_parsing():
    """Test the improved resume parser"""

    # Test with your resume
    pdf_path = "user_data/raw/resume_ML/resume.pdf"

    if not Path(pdf_path).exists():
        print(f"❌ Resume not found: {pdf_path}")
        return

    print("=" * 60)
    print("Testing Resume Parser with:", pdf_path)
    print("=" * 60)

    result = parse_resume_all(pdf_path)

    # 1. Check Experience/Projects parsing
    print("\n📋 EXPERIENCE & PROJECTS")
    print("-" * 60)
    entries = result["entries"]
    sections = {}
    for entry in entries:
        section = entry["section"]
        if section not in sections:
            sections[section] = 0
        sections[section] += 1

    for section, count in sections.items():
        print(f"  {section}: {count} bullets")

    print(f"\n  Total bullets captured: {len(entries)}")

    # Show first few entries from each section
    print("\n  Sample entries:")
    seen_sections = set()
    for entry in entries[:5]:
        section = entry["section"]
        if section not in seen_sections:
            print(f"\n  [{section}] {entry['entry']}")
            print(f"    → {entry['text'][:80]}...")
            seen_sections.add(section)

    # 2. Check Education parsing
    print("\n\n🎓 EDUCATION (Structured)")
    print("-" * 60)
    edu = result["education_structured"]

    if not edu:
        print("  ❌ No education data parsed!")
    else:
        for i, school in enumerate(edu, 1):
            print(f"\n  School {i}:")
            print(f"    Name:   {school['school_name']}")
            print(f"    Degree: {school['degree'] or 'Not detected'}")
            print(f"    Major:  {school['major'] or 'Not detected'}")
            print(f"    Dates:  {school['dates'] or 'Not detected'}")
            print(f"    GPA:    {school['gpa'] or 'Not detected'}")
            if school['courses']:
                print(f"    Courses: {len(school['courses'])} lines")
                for course_line in school['courses'][:2]:  # Show first 2
                    print(f"      - {course_line[:60]}...")

    # 3. Summary
    print("\n\n📊 SUMMARY")
    print("-" * 60)
    print(f"  ✅ Total bullets: {len(entries)}")
    print(f"  ✅ Work Experience bullets: {sections.get('WORK EXPERIENCE', 0)}")
    print(f"  ✅ Projects bullets: {sections.get('PROJECTS', 0)}")
    print(f"  ✅ Leadership bullets: {sections.get('LEADERSHIP EXPERIENCE', 0)}")
    print(f"  ✅ Schools parsed: {len(edu)}")

    # Check if improvements worked
    issues = []
    if sections.get('WORK EXPERIENCE', 0) == 0:
        issues.append("❌ WORK EXPERIENCE not captured")
    if sections.get('PROJECTS', 0) == 0:
        issues.append("❌ PROJECTS not captured")
    if len(edu) == 0:
        issues.append("❌ No education parsed")
    else:
        for school in edu:
            if not school.get('degree'):
                issues.append(f"❌ Degree not parsed for {school['school_name']}")
            if not school.get('gpa') and 'GPA' in result['metadata'].get('EDUCATION', ''):
                issues.append(f"⚠️  GPA not parsed for {school['school_name']}")

    if issues:
        print("\n\n⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n\n✨ ALL CHECKS PASSED!")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_resume_parsing()
