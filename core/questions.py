# core/questions.py
from typing import Optional, List, Dict, Set
import random

from .llm_client import client


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
    """用 JD + mode 生一題題目，避免出現在 avoid 裡。
    ⚠️ 注意：
    - jd_text 可能不只是純 JD，還可能已經被前面的程式碼包成：
      "Interviewer persona: ...\n\nJob description and context: ... "
      也就是「persona + JD + 其他備註」的混合 context。
    - 這邊的 prompt 會明講要同時考慮 persona 和 JD。
    """
    avoid = avoid or set()

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

    # ----- auto 模式才在 Python 端選一個 angle，其餘交給 LLM -----
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
            "Do NOT repeat these questions or ask about the exact same scenario again. "
            "Ask something that probes a clearly different aspect of the candidate's skills "
            "(for example: a different project, a different type of challenge, or a different technical focus).\n"
            "Previously asked questions:\n"
            f"{joined}\n\n"
        )

    # ----- 組 prompt -----
    prompt = (
        "You are an interview coach helping an interviewer generate realistic questions.\n\n"
        "The context block below may include an interviewer persona description "
        "(starting with 'Interviewer persona:') followed by the actual job description "
        "and any extra notes. Use ALL of this context so that:\n"
        "- the question matches the role and required skills, and\n"
        "- the wording and tone feel consistent with the interviewer persona.\n\n"
        f"Please write ONE {style}.\n\n"
        f"{avoid_block}"
        "Context (persona + job description + notes):\n"
        f"{jd_text}\n\n"
    )

    # technical：讓 LLM 自己根據 JD 選 angle（方案 B）
    if mode == "technical":
        prompt += (
            "First, infer from the context which technical areas are most important for this role.\n"
            "Then choose ONE primary angle from the list below that best matches this job, "
            "and write a single question focusing on that angle:\n"
            "- coding or debugging in the languages or frameworks mentioned in the job description\n"
            "- algorithms and data structures relevant to the scale and constraints of this role\n"
            "- SQL and data manipulation for the kinds of data sources mentioned (e.g., warehouses, logs, event data)\n"
            "- training and evaluating machine learning models for the problems described\n"
            "- experiment design and A/B testing to measure product or model impact\n"
            "- end-to-end ML system or data pipeline design, including data ingestion and serving\n"
            "- LLM application design (prompting, tool use, or agent-like behavior) if the job mentions LLMs or GPT\n"
            "- Retrieval-Augmented Generation (RAG), embeddings, or vector databases if they appear in the job\n"
            "- deployment, serving, and MLOps (latency, throughput, monitoring, scaling) if the job involves production systems\n"
            "- safety, privacy, and guardrails (e.g., PII, hallucinations, abuse) if those concepts are mentioned\n\n"
            "If the job description explicitly mentions LLMs, RAG, embeddings, vector databases, MLOps, or production systems, "
            "prefer one of those angles.\n\n"
            "Write the question exactly as an interviewer would say it aloud. "
            "Do not mention the angle explicitly in the question.\n\n"
        )

    # case：根據 JD 出一題 case reasoning
    elif mode == "case":
        prompt += (
            "Create ONE realistic case-style interview question aligned with this role and context.\n"
            "Frame a short scenario (1–3 sentences) and then ask the candidate to talk through how they would:\n"
            "- clarify the business or product goal and define success metrics\n"
            "- make reasonable assumptions about the data, users, and constraints\n"
            "- outline a step-by-step approach to solve or investigate the problem\n"
            "- discuss trade-offs, risks, and how they would validate their solution\n\n"
            "The question should be answerable without specific company-internal knowledge, "
            "based only on common sense and what a strong candidate could infer.\n"
            "Avoid brainteaser puzzles or pure market-sizing questions unless the job description explicitly suggests consulting-style cases.\n\n"
            "Write the question exactly as an interviewer would say it aloud.\n\n"
        )

    # auto 模式：還是用你原本的 random angle（避免太 generic）
    if angle:
        prompt += f"The question should specifically {angle}.\n\n"

    prompt += "Return only the question text."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    q = text.split("\n")[0].strip()

    # 保護一下：如果 LLM 偶然回到 avoid 清單，再重試一次
    if q in avoid:
        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )
        text2 = resp2.choices[0].message.content.strip()
        q2 = text2.split("\n")[0].strip()
        if q2 and q2 not in avoid:
            q = q2

    return q


def get_behavioral_question(
    profile_id: str,
    subtype: str,
    asked: Optional[Set[str]] = None,
) -> str:
    """
    從內建 BEHAVIORAL_BANK 抽一題（跟 JD 無關，純 generic behavioral）
    """
    bank = BEHAVIORAL_BANK.get(subtype)
    if not bank:
        all_q = []
        for lst in BEHAVIORAL_BANK.values():
            all_q.extend(lst)
        bank = all_q

    asked = asked or set()
    remaining = [q for q in bank if q not in asked]

    if remaining:
        return random.choice(remaining)
    else:
        return random.choice(bank)


def get_technical_question(
    jd_text: str,
    asked: Optional[Set[str]] = None,
) -> str:
    """
    給 mock / practice 用的 helper：
    用 JD 產生一題 technical 題目，會避開 asked 裡出現過的題目文字。
    jd_text 一樣可以是「persona + JD」的混合 context。
    """
    asked = asked or set()
    return call_llm_for_question(jd_text=jd_text, mode="technical", avoid=asked)


def call_llm_for_project_question(
    jd_text: str,
    entry_title: str,
    bullets: List[Dict[str, str]],
    previous_qas: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    對某一個具體 project 做 deep dive，用 JD 當 context。
    - jd_text 也可以包含 interviewer persona（同 mock_interview 那邊 prepend 的字串）。
    - previous_qas 讓 LLM 知道之前聊過什麼，避免重複。
    """
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
                "Ask a follow-up question that goes deeper into a different aspect "
                "(decisions, trade-offs, metrics, collaboration, or reflection).\n\n"
            )

    bullet_text = "\n".join(f"- {b.get('text', '')}" for b in bullets)

    prompt = (
        "You are an interviewer doing a deep dive on ONE specific project from the candidate's resume.\n\n"
        "The context block below may include an interviewer persona description "
        "(starting with 'Interviewer persona:') followed by the job description and any extra notes.\n"
        "Your question should stay realistic for that role, and your tone should match the persona.\n\n"
        f"Job description / context:\n{jd_text}\n\n"
        f"The project/experience is:\n{entry_title}\n\n"
        f"Relevant bullets from the resume:\n{bullet_text}\n\n"
        f"{prev_block}"
        "Write ONE short follow-up question you would ask about this project. "
        "Return only the question text."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    return text.split("\n")[0].strip()


def call_llm_for_followup_question(
    jd_text: str,
    mode: str,
    base_question: str,
    bullets: List[Dict[str, str]],
    qa_history: Optional[List[Dict[str, str]]] = None,
    avoid: Optional[Set[str]] = None,
) -> str:
    """
    根據目前的題目 + QA 歷史 + JD，生一題 follow-up。
    mode 會影響 follow-up 的角度（behavioral / project / technical / auto / custom）。
    jd_text 同樣可以是「persona + JD」混合 context。
    """
    qa_history = qa_history or []
    avoid = avoid or set()

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

    style_hint = {
        "behavioral": "a deeper behavioral follow-up question (for example: emotions, conflict, reflection, or what they would do differently).",
        "project": "a deeper technical follow-up about decisions, trade-offs, metrics, or collaboration.",
        "technical": "a deeper technical follow-up that probes the candidate's reasoning, trade-offs, or understanding of the underlying concepts, tools, or system design.",
        "auto": "a deeper follow-up about impact, decisions, or what they would do differently.",
        "custom": "a deeper follow-up from a different perspective on the same situation.",
    }.get(mode, "a deeper follow-up question.")

    avoid_block = ""
    if avoid:
        joined = "\n".join(f"- {q}" for q in list(avoid)[:10])
        avoid_block = (
            "Do NOT repeat or rephrase any of the following questions:\n"
            f"{joined}\n\n"
        )

    prompt = (
        "You are an interviewer asking follow-up questions in a realistic live interview.\n\n"
        "The context block below may include an interviewer persona description "
        "(starting with 'Interviewer persona:') followed by the job description and any extra notes.\n"
        "Your follow-up should stay aligned with that role and persona.\n\n"
        f"Job description / context:\n{jd_text}\n\n"
        f"The main question is:\n{base_question}\n\n"
        f"Relevant resume bullets:\n{bullet_text}\n\n"
        f"{prev_block}"
        f"{avoid_block}"
        f"Now ask ONE short {style_hint}\n"
        "Return only the question text, as you would say it to the candidate."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    return text.split("\n")[0].strip()


def generate_followup_question(
    jd_text: str,
    mode: str,
    base_question: str,
    bullets: List[Dict[str, str]],
    qa_history: Optional[List[Dict[str, str]]],
    avoid: Set[str],
) -> Optional[str]:
    """
    包一圈 retry + 去重邏輯，避免 LLM 回答跟前面重複。
    jd_text 仍然可以是「persona + JD」混合 context。
    """
    qa_history = qa_history or []

    def norm(s: str) -> str:
        return s.rstrip("?.! ").strip().lower()

    avoid_norm = {norm(q) for q in avoid if q}

    for temp in [0.7, 0.9, 1.0]:
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
