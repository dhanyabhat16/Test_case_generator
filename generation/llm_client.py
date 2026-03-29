"""
llm_client.py
-------------
LLM client with Groq as the primary provider and Gemini as automatic fallback.
Reads API keys from environment variables (set in .env file).

Usage:
    from generation.llm_client import generate
    response = generate(prompt="Write tests for...")
"""

from __future__ import annotations

import os
import time
from typing import Optional

# Load .env file automatically if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # If dotenv not installed, keys must be set in the shell environment

# ── Model config ───────────────────────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"  # Fast, high quality, generous free tier
GEMINI_MODEL = "gemini-1.5-flash"  # Fast Gemini model for fallback

MAX_TOKENS   = 4096   # Max tokens to generate
TEMPERATURE  = 0.2    # Low temperature = more deterministic test code
MAX_RETRIES  = 3      # Number of retry attempts before switching provider
RETRY_DELAY  = 2      # Seconds to wait between retries


def generate(
    prompt: str,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Send a prompt to the LLM and return the generated text.
    Tries Groq first; falls back to Gemini automatically if Groq fails.

    Args:
        prompt      : The full prompt string (built by prompt_engine.py).
        max_tokens  : Max tokens to generate (default 4096).
        temperature : Sampling temperature (default 0.2 for code generation).

    Returns:
        The generated text string from the LLM.

    Raises:
        RuntimeError: If both Groq and Gemini fail after all retries.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    # Try Groq first
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        result = _try_groq(prompt, max_tokens, temperature, groq_key)
        if result is not None:
            return result
        print("[LLMClient] Groq failed or hit rate limit. Switching to Gemini...")
    else:
        print("[LLMClient] GROQ_API_KEY not set. Trying Gemini...")

    # Fallback to Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        result = _try_gemini(prompt, max_tokens, temperature, gemini_key)
        if result is not None:
            return result
        print("[LLMClient] Gemini also failed.")
    else:
        print("[LLMClient] GEMINI_API_KEY not set.")

    raise RuntimeError(
        "Both Groq and Gemini failed. "
        "Check your API keys in .env and your internet connection."
    )


# ── Groq client ────────────────────────────────────────────────────────────────

def _try_groq(
    prompt: str,
    max_tokens: int,
    temperature: float,
    api_key: str,
) -> Optional[str]:
    """
    Attempt generation via Groq API with retry logic.
    Returns the response text, or None if all attempts fail.
    """
    try:
        from groq import Groq
    except ImportError as exc:
        print(f"[LLMClient] groq package not installed: {exc}")
        return None

    client = Groq(api_key=api_key)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[LLMClient] Groq attempt {attempt}/{MAX_RETRIES} "
                  f"(model: {GROQ_MODEL})...")

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert software testing engineer. "
                            "Generate clean, correct, and complete test code. "
                            "Return only code — no markdown fences, no explanations "
                            "outside of code comments."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            text = response.choices[0].message.content or ""
            text = _clean_llm_output(text)

            if text:
                print(f"[LLMClient] ✅ Groq responded ({len(text)} chars)")
                return text

        except Exception as exc:
            err_str = str(exc).lower()

            # Rate limit — back off and retry
            if "rate_limit" in err_str or "429" in err_str:
                print(f"[LLMClient] Groq rate limit hit. "
                      f"Waiting {RETRY_DELAY * attempt}s...")
                time.sleep(RETRY_DELAY * attempt)
                continue

            # Context too long — can't retry with same prompt
            if "context" in err_str or "token" in err_str:
                print(f"[LLMClient] Groq context length error: {exc}")
                return None

            # Other errors — retry
            print(f"[LLMClient] Groq error on attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return None  # All retries exhausted


# ── Gemini client ──────────────────────────────────────────────────────────────

def _try_gemini(
    prompt: str,
    max_tokens: int,
    temperature: float,
    api_key: str,
) -> Optional[str]:
    """
    Attempt generation via Google Gemini API with retry logic.
    Returns the response text, or None if all attempts fail.
    """
    try:
        import google.generativeai as genai
    except ImportError as exc:
        print(f"[LLMClient] google-generativeai package not installed: {exc}")
        return None

    genai.configure(api_key=api_key)

    system_instruction = (
        "You are an expert software testing engineer. "
        "Generate clean, correct, and complete test code. "
        "Return only code — no markdown fences, no explanations "
        "outside of code comments."
    )

    generation_config = {
        "max_output_tokens": max_tokens,
        "temperature"      : temperature,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[LLMClient] Gemini attempt {attempt}/{MAX_RETRIES} "
                  f"(model: {GEMINI_MODEL})...")

            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config=generation_config,
                system_instruction=system_instruction,
            )

            response = model.generate_content(prompt)
            text     = response.text or ""
            text     = _clean_llm_output(text)

            if text:
                print(f"[LLMClient] ✅ Gemini responded ({len(text)} chars)")
                return text

        except Exception as exc:
            err_str = str(exc).lower()

            if "quota" in err_str or "429" in err_str or "rate" in err_str:
                print(f"[LLMClient] Gemini rate limit hit. "
                      f"Waiting {RETRY_DELAY * attempt}s...")
                time.sleep(RETRY_DELAY * attempt)
                continue

            print(f"[LLMClient] Gemini error on attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return None  # All retries exhausted


# ── Output cleaner ─────────────────────────────────────────────────────────────

def _clean_llm_output(text: str) -> str:
    """
    Strip markdown code fences that some models add even when told not to.
    e.g. ```python ... ``` → just the code inside.
    """
    text = text.strip()

    # Remove leading ```python or ```gherkin or ``` fence
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (the opening fence)
        lines = lines[1:]
        # Remove last line if it's a closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing LLM client...")
    test_prompt = (
        "Write a single pytest function that tests if 1 + 1 == 2. "
        "Return only the Python code."
    )
    try:
        result = generate(test_prompt, max_tokens=200)
        print("\n=== LLM Response ===")
        print(result)
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("Make sure GROQ_API_KEY or GEMINI_API_KEY is set in your .env file.")
