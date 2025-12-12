# core/llm_client.py
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# 1. 初始化 OpenAI Client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    # 如果連 OpenAI都沒有，那真的沒救了，直接噴錯
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

openai_client = OpenAI(api_key=openai_api_key)

# 2. 初始化 Gemini Client (允許失敗)
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    print("Warning: GEMINI_API_KEY not found in .env. Will use OpenAI fallback.")


def ask_llm(
    messages: list[dict],
    provider: str = "openai",
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """
    統一的 LLM 呼叫接口。
    🔥 安全機制：如果 Gemini 呼叫失敗 (例如 Key Leaked / 403 / 網路錯誤)，
    會自動切換回 OpenAI，保證程式不崩潰。
    """

    # 定義一個內部函式來呼叫 OpenAI，方便等一下重複使用
    def call_openai():
        # 如果是 fallback 過來的，我們通常強制用便宜好用的 gpt-4o-mini
        target_model = "gpt-4o-mini" 
        try:
            response = openai_client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling OpenAI (Fallback failed): {str(e)}"

    # --- 主邏輯開始 ---
    try:
        # 如果指定要用 Gemini，且我們有 Key
        if provider == "gemini":
            if not gemini_api_key:
                print("⚠️ No Gemini Key found. Falling back to OpenAI.")
                return call_openai()
            
            try:
                # 嘗試呼叫 Gemini
                target_model = model or "gemini-2.5-flash-lite"
                gemini_model = genai.GenerativeModel(target_model)
                
                # Gemini 1.5 接受純文字 Prompt，我們把對話接起來
                full_prompt = "\n\n".join([m["content"] for m in messages])
                
                response = gemini_model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                return response.text.strip()
            
            except Exception as gemini_error:
                # 捕捉所有 Gemini 的錯誤 (包含 403 Key Leaked)
                print(f"🔴 Gemini Error: {gemini_error}")
                print("🔄 Automatically switching to OpenAI...")
                return call_openai()

        else:
            # 原本就指定用 OpenAI
            return call_openai()

    except Exception as e:
        return f"System Error: {str(e)}"

# --- 向下相容設定 (Backward Compatibility) ---
# 這一行是為了讓其他還沒改寫的檔案 (如 core/answers.py) 
# 可以繼續用 `from .llm_client import client` 而不報錯
client = openai_client
