"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    redacted = response

    # PII / secret patterns. Thứ tự quan trọng: pattern dài / đặc thù chạy trước,
    # và mỗi pattern chạy trên bản ĐÃ redact → không đếm trùng 1 chuỗi 2 lần.
    PII_PATTERNS = {
        # API key kiểu "sk-vinbank-secret-2024", "sk-demo-key-abc123"
        "api_key": r"\bsk-[A-Za-z0-9_-]{6,}",
        # Email
        "email": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b",
        # "password=admin123", "password: x", "password is admin123" → che phần giá trị
        # (giá trị dừng trước dấu phẩy / chấm phẩy / chấm cuối câu).
        "password": r"\b(?:password|passwd|pwd|mat khau|mật khẩu)\s*(?:is|là|[:=])\s*\S+?(?=[.,;]?(?:\s|$))",
        # Secret nội bộ của lab dù không có chữ "password" đứng trước
        "internal_secret": r"\badmin123\b|\bdb\.vinbank\.internal(?::\d+)?",
        # CCCD 12 số / CMND 9 số (đứng riêng, không nằm trong số dài hơn)
        "national_id": r"(?<!\d)(?:\d{12}|\d{9})(?!\d)",
        # SĐT VN: 0 + 9–10 số (di động 10 số, bàn 11 số). Không bắt hotline 1900...
        "phone": r"(?<!\d)(?:\+84|0)\d{9,10}(?!\d)",
    }

    for name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted, re.IGNORECASE)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# Lớp bổ sung cho output (ngoài yêu cầu tối thiểu CP2)
# ============================================================

def _compact(text: str) -> str:
    """Bỏ mọi ký tự không phải chữ/số + lowercase: 'a d-m.i n 1 2 3' → 'admin123'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def detect_secret_leak(response: str) -> list[str]:
    """Lớp 2a — canary: secret DEMO thật của hệ thống có lọt ra không?

    Regex ở content_filter bắt theo *dạng* (sk-..., password=...). Lớp này bắt
    theo *giá trị*, kể cả khi model viết tách / chèn ký tự / viết ngược
    để né regex. Có lộ → chặn CẢ câu trả lời (không chỉ redact).
    """
    compact = _compact(response)
    leaked = []
    for secret in DEMO_SECRETS:
        needle = _compact(secret)
        if len(needle) >= 6 and (needle in compact or needle[::-1] in compact):
            leaked.append(secret)
    return leaked


# Dấu hiệu model đang đọc lại system prompt / chỉ dẫn nội bộ của nó.
PROMPT_LEAK_PATTERNS = [
    r"do not disclose",
    r"security\s*\(must not violate\)",
    r"internal note",
    r"\bmy (system )?(prompt|instructions) (is|are|say|says)\b",
    r"\bi (was|am) (instructed|told|programmed) to\b",
    r"you are a helpful customer service assistant for vinbank",
]


def detect_prompt_leak(response: str) -> bool:
    """Lớp 2b — câu trả lời có trích lại system prompt không?"""
    lower = response.lower()
    return any(re.search(p, lower) for p in PROMPT_LEAK_PATTERNS)


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    SAFE_REFUSAL = (
        "I'm sorry, I can't share that information. "
        "Please contact VinBank support for further help."
    )

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.last_block_layer: str | None = None

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 0. Canary secret (kể cả bị che/tách chữ) hoặc lộ system prompt
        #    → chặn TOÀN BỘ câu trả lời, không cố redact từng phần.
        if detect_secret_leak(response_text) or detect_prompt_leak(response_text):
            self.blocked_count += 1
            self.last_block_layer = (
                "secret_leak" if detect_secret_leak(response_text) else "prompt_leak"
            )
            llm_response.content = self._make_content(self.SAFE_REFUSAL)
            return llm_response
        self.last_block_layer = None

        # 1. Regex PII / secret → thay nội dung bằng bản đã [REDACTED].
        filtered = content_filter(response_text)
        if not filtered["safe"]:
            self.redacted_count += 1
            response_text = filtered["redacted"]
            llm_response.content = self._make_content(response_text)

        # 2. (Optional) LLM-as-Judge — chỉ chạy khi đã khởi tạo judge.
        if self.use_llm_judge:
            judge = await llm_safety_check(response_text)
            if not judge["safe"]:
                self.blocked_count += 1
                self.last_block_layer = "llm_judge"
                llm_response.content = self._make_content(self.SAFE_REFUSAL)

        # 3. Trả response (đã sửa nếu cần).
        return llm_response

    @staticmethod
    def _make_content(text: str) -> types.Content:
        return types.Content(role="model", parts=[types.Part.from_text(text=text)])


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
