# core/questions.py
from typing import Optional, List, Dict, Set
import random

# 改成從 llm_client 匯入 ask_llm
from .llm_client import ask_llm

BEHAVIORAL_BANK = {
    "teamwork": [
        "Tell me about a time you worked on a team and faced a challenge.",
        "Describe a time you had to help a teammate who was struggling.",
        "Tell me about a time when you disagreed with your team and how you handled it.",
    ],
    "conflict": [
        "Tell me about a time you had a conflict with a teammate or stakeholder.",
        "Describe a time you had to deliver bad news to someone.",
    ],
    "leadership": [
        "Tell me about a time you had to take the lead on a project.",
        "Describe a situation where you motivated others to achieve a goal.",
    ],
    "failure": [
        "Tell me about a time you failed at something and what you learned.",
        "Describe a time a project did not go as planned.",
    ],
    "strengths_weaknesses": [
        "What do you consider your greatest strength, and can you give me an example?",
        "Tell me about a weakness you are actively working on.",
    ],
}


def call_llm_for_question(
    jd_text: str,
    mode: str,
    avoid: Optional[Set[str]] = None,
) -> str:
    """用 JD + mode 生一題題目，避免出現在 avoid 裡。"""
    avoid = avoid or set()

    # 1. 決定 Provider: 如果是 Behavioral 就用 Gemini，其他用 OpenAI
    provider = "gemini" if mode == "behavioral" else "openai"

    # ----- style：決定這題「大概是什麼類型」 -----
    if mode == "behavioral":
        style = "behavioral STAR-format question focusing on teamwork, conflict, impact, or ownership"
    elif mode == "project":
        style = "deep-dive question about one of the candidate's past projects that is relevant to this job"
    elif mode == "technical":
        style = (
            "technical interview question that directly reflects the core skills, tools, or concepts required for this job, "
            "phrased as a realistic spoken question an interviewer would ask"
        )
    elif mode == "case":
        style = (
            "data or ML product case question that asks the candidate to reason through a realistic scenario step by step, "
            "focusing on goals, assumptions, metrics, and trade-offs"
        )
    else:  # auto / mixed
        style = "interview question that mixes behavioral and resume-based aspects relevant to this job"

    # ----- auto 模式才在 Python 端選一個 angle -----
    angle = ""
    if mode == "auto":
        angle_choices = [
            "focus on the measurable impact and metrics of a past project",
            "focus on collaboration with teammates or cross-functional stakeholders",
            "focus on how the candidate dealt with ambiguity or unclear requirements",
            "focus on a tricky debugging or failure case and what they learned",
            "focus on how they balanced trade-offs such as speed vs quality or model complexity vs reliability",
        ]
        angle = random.choice(angle_choices)

    # ----- 避免問過的題目 -----
    avoid_block = ""
    if avoid:
        joined = "\n".join(f"- {q}" for q in list(avoid)[:10])
        avoid_block = (
            "We have already asked the following questions in previous turns.\n"
            "Do NOT repeat these questions. Ask something clearly different.\n"
            "Previously asked questions:\n"
            f"{joined}\n\n"
        )

    # ----- 組 prompt -----
    system_prompt = "You are an interview coach helping an interviewer generate realistic questions."
    
    user_prompt = (
        "The context block below may include an interviewer persona description "
        "(starting with 'Interviewer persona:') followed by the actual job description "
        "and any extra notes. Use ALL of this context.\n\n"
        f"Please write ONE {style}.\n\n"
        f"{avoid_block}"
        "Context (persona + job description + notes):\n"
        f"{jd_text}\n\n"
    )

    # technical：讓 LLM 自己根據 JD 選 angle
    if mode == "technical":
        user_prompt += (
            "First, infer from the context which technical areas are most important for this role.\n"
            "Then choose ONE primary angle (coding, algos, SQL, ML models, experiment design, system design, LLMs, MLOps, etc.) "
            "that best matches this job, and write a single question focusing on that angle.\n\n"
            "Write the question exactly as an interviewer would say it aloud. "
            "Do not mention the angle explicitly in the question.\n\n"
        )

    # case：根據 JD 出一題 case reasoning
    elif mode == "case":
        user_prompt += (
            "Create ONE realistic case-style interview question aligned with this role and context.\n"
            "Frame a short scenario (1–3 sentences) and then ask the candidate to talk through how they would:\n"
            "- clarify the business or product goal and define success metrics\n"
            "- make reasonable assumptions about the data, users, and constraints\n"
            "- outline a step-by-step approach\n"
            "- discuss trade-offs and validation\n\n"
            "The question should be answerable without specific company-internal knowledge.\n"
            "Write the question exactly as an interviewer would say it aloud.\n\n"
        )

    if angle:
        user_prompt += f"The question should specifically {angle}.\n\n"

    user_prompt += "Return only the question text."

    # ----- 呼叫 LLM (使用 ask_llm) -----
    q = ask_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        provider=provider,
        temperature=0.7
    )
    
    # 清理結果
    q = q.split("\n")[0].strip()

    # Retry 機制：如果重複了，再試一次 (稍微提高 temperature)
    if q in avoid:
        q = ask_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            provider=provider,
            temperature=0.9
        )
        q = q.split("\n")[0].strip()

    return q


def get_behavioral_question(
    profile_id: str,
    subtype: str,
    asked: Optional[Set[str]] = None,
) -> str:
    """
    [修改版] 現在這個函式會呼叫 Gemini 針對 subtype 生成題目，
    而不再只是從字典裡抽籤。
    """
    asked = asked or set()
    
    # 1. 如果 subtype 在字典裡，我們把它拿來當作 "Avoid list" (避免出一樣的)
    # 或者是當作範例讓 LLM 參考風格
    bank_examples = BEHAVIORAL_BANK.get(subtype, [])
    
    # 2. 準備 Prompt
    # 我們讓 Gemini 根據 subtype (如 teamwork, failure) 產生新題目
    system_prompt = "You are an expert interview coach."
    
    # 組合已經問過的題目 + 字典裡的題目，叫 Gemini 避開
    avoid_list = list(asked) + bank_examples
    avoid_text = "\n".join([f"- {q}" for q in avoid_list[:10]]) # 取前10個避免 prompt 太長
    
    user_prompt = f"""
    Generate ONE behavioral interview question specifically about '{subtype}'.
    
    The question should be in the STAR format style (Situation, Task, Action, Result).
    
    Constraint:
    - Do NOT ask exactly the same questions as below:
    {avoid_text}
    
    - Make it sound natural and professional.
    - Return ONLY the question text.
    """

    # 3. 呼叫 Gemini (因為是 Behavioral，我們強制用 Gemini)
    # 注意：這裡沒有 JD context，所以是純 Behavioral
    new_question = ask_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        provider="gemini", # 強制用 Gemini 發揮創意
        temperature=0.8    # 稍微調高讓題目更多變
    )
    
    # 簡單清理
    return new_question.strip()


def get_technical_question(
    jd_text: str,
    asked: Optional[Set[str]] = None,
) -> str:
    """Helper: 用 JD 產生 technical 題目"""
    asked = asked or set()
    return call_llm_for_question(jd_text=jd_text, mode="technical", avoid=asked)


def call_llm_for_project_question(
    jd_text: str,
    entry_title: str,
    bullets: List[Dict[str, str]],
    previous_qas: Optional[List[Dict[str, str]]] = None,
) -> str:
    """對某一個具體 project 做 deep dive"""
    prev_block = ""
    if previous_qas:
        lines = []
        for qa in previous_qas[-3:]:
            q = qa.get("question", "")
            a = qa.get("answer", "")
            lines.append(f"Q: {q}\nA: {a}\n")
        if lines:
            prev_block = (
                "Here is the previous conversation about this project:\n"
                + "\n".join(lines)
                + "\n\n"
                "Ask a follow-up question that goes deeper into a different aspect.\n\n"
            )

    bullet_text = "\n".join(f"- {b.get('text', '')}" for b in bullets)

    prompt = (
        "You are an interviewer doing a deep dive on ONE specific project from the candidate's resume.\n\n"
        "The context block below may include an interviewer persona description.\n"
        "Your question should stay realistic for that role.\n\n"
        f"Job description / context:\n{jd_text}\n\n"
        f"The project/experience is:\n{entry_title}\n\n"
        f"Relevant bullets from the resume:\n{bullet_text}\n\n"
        f"{prev_block}"
        "Write ONE short follow-up question you would ask about this project. "
        "Return only the question text."
    )

    # 專案深挖通常需要邏輯較強，維持使用 OpenAI
    return ask_llm(
        messages=[{"role": "user", "content": prompt}],
        provider="openai",
        temperature=0.7
    )


def call_llm_for_followup_question(
    jd_text: str,
    mode: str,
    base_question: str,
    bullets: List[Dict[str, str]],
    qa_history: Optional[List[Dict[str, str]]] = None,
    avoid: Optional[Set[str]] = None,
) -> str:
    """根據目前的題目 + QA 歷史 + JD，生一題 follow-up"""
    qa_history = qa_history or []
    avoid = avoid or set()

    # 決定 Follow-up 用誰：Behavioral follow-up 繼續用 Gemini
    provider = "gemini" if mode == "behavioral" else "openai"

    prev_block = ""
    if qa_history:
        lines = []
        for qa in qa_history[-3:]:
            q = qa.get("question", "")
            a = qa.get("answer", "")
            lines.append(f"Q: {q}\nA: {a}\n")
        prev_block = (
            "Here is the previous conversation about this topic:\n"
            + "\n".join(lines)
            + "\n\n"
        )

    bullet_text = "\n".join(f"- {b.get('text', '')}" for b in bullets)
    
    style_hint = "a deeper follow-up question."
    if mode == "behavioral":
        style_hint = "a deeper behavioral follow-up question (emotions, conflict, reflection)."
    elif mode == "technical":
        style_hint = "a deeper technical follow-up probing reasoning or understanding."

    avoid_block = ""
    if avoid:
        joined = "\n".join(f"- {q}" for q in list(avoid)[:10])
        avoid_block = (
            "Do NOT repeat or rephrase any of the following questions:\n"
            f"{joined}\n\n"
        )

    prompt = (
        "You are an interviewer asking follow-up questions in a realistic live interview.\n\n"
        "The context block below may include an interviewer persona description.\n"
        f"Job description / context:\n{jd_text}\n\n"
        f"The main question is:\n{base_question}\n\n"
        f"Relevant resume bullets:\n{bullet_text}\n\n"
        f"{prev_block}"
        f"{avoid_block}"
        f"Now ask ONE short {style_hint}\n"
        "Return only the question text, as you would say it to the candidate."
    )

    q = ask_llm(
        messages=[{"role": "user", "content": prompt}],
        provider=provider,
        temperature=0.7
    )
    return q.split("\n")[0].strip()


def generate_followup_question(
    jd_text: str,
    mode: str,
    base_question: str,
    bullets: List[Dict[str, str]],
    qa_history: Optional[List[Dict[str, str]]],
    avoid: Set[str],
) -> Optional[str]:
    """Retry Wrapper"""
    qa_history = qa_history or []

    def norm(s: str) -> str:
        return s.rstrip("?.! ").strip().lower()

    avoid_norm = {norm(q) for q in avoid if q}

    # 這裡內部會呼叫 call_llm_for_followup_question，它已經會自動選 provider 了
    for _ in range(3):
        q = call_llm_for_followup_question(
            jd_text=jd_text,
            mode=mode,
            base_question=base_question,
            bullets=bullets,
            qa_history=qa_history,
            avoid=avoid,
        )
        q_norm = norm(q)
        if q and q_norm not in avoid_norm:
            return q

    return None