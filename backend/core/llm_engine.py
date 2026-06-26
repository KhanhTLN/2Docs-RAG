from __future__ import annotations
import sys, os, json, re, logging, requests
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)

_instance = None


def get_llm():
    global _instance
    if _instance is None:
        _instance = LLMEngine()
    return _instance


class LLMEngine:

    _VALID_MUC_DO = ("cao", "trung binh", "thap")

    # ── System prompt tối ưu cho 7b ──────────────────────────────────
    # Ngắn hơn nhưng rõ ràng hơn → 7b tuân theo tốt hơn 14b prompt dài
    _SYS_COMPARE = (
        "So sánh hợp đồng VN. Trả JSON array thay đổi. "
        'Mỗi phần tử: {"change_type":"SUA|THEM|XOA","mo_ta":"...",'
        '"vi_tri":"...","trich_dan_a":"...","trich_dan_b":"...",'
        '"muc_do":"cao|trung binh|thap"}. Không markdown. [] nếu không đổi.'
    )
    
    _SYS_SUMMARY = (
        "Bạn là trợ lý tóm tắt văn bản pháp lý tiếng Việt.\n"
        "Tóm tắt ngắn gọn, khách quan, nêu rõ điều khoản cụ thể."
    )

    def __init__(self):
        import config as cfg
        self._url    = cfg.LLM_BASE_URL
        self._model  = cfg.LLM_MODEL_NAME
        self._temp   = cfg.LLM_TEMPERATURE          # 0.05 từ config
        self._maxt   = cfg.LLM_MAX_TOKENS
        self._tout   = cfg.LLM_TIMEOUT
        self._num_ctx     = getattr(cfg, "LLM_NUM_CTX",        4096)
        self._repeat_pen  = getattr(cfg, "LLM_REPEAT_PENALTY", 1.1)
        self._top_p       = getattr(cfg, "LLM_TOP_P",          0.9)
        self._ready = False
        self._check()

    def _check(self):
        try:
            r = requests.get(f"{self._url}/api/tags", timeout=5)
            if r.status_code == 200:
                names = [m["name"] for m in r.json().get("models", [])]
                if any(self._model in n for n in names):
                    self._ready = True
                    logger.info(
                        f"Ollama ready | {self._model} | "
                        f"temp={self._temp} num_ctx={self._num_ctx}"
                    )
                else:
                    logger.warning(f"Model chưa pull: ollama pull {self._model}")
        except requests.ConnectionError:
            logger.warning("Ollama chưa chạy. Chạy: ollama serve")

    # ── Public API ────────────────────────────────────────────────────

    def compare_chunks(self, text_a: str, text_b: str, context: str = "") -> List[dict]:
        if text_a.strip() == text_b.strip():
            return [self._no_change(context)]

        loc = f"{context}\n" if context else ""
        max_chars = 600 if max(len(text_a), len(text_b)) > 800 else 800
        a_text = text_a[:max_chars].strip()
        b_text = text_b[:max_chars].strip()

        prompt = (
            f"{loc}A:\n{a_text}\n\nB:\n{b_text}\n\n"
            "Liệt kê thay đổi. Trả JSON array."
        )

        raw   = self._chat(prompt, self._SYS_COMPARE)
        items = self._parse_list(raw)
        items = [self._fix_item(d) for d in items]

        # FIX: fallback chỉ khi LLM trả [] mà 2 đoạn thực sự khác nhau
        if not items:
            logger.warning(f"LLM trả [] tại [{context}] — dùng fallback SUA")
            items = [{
                "change_type": "SUA",
                "mo_ta": f"Nội dung thay đổi tại: {context}" if context else "Nội dung thay đổi",
                "vi_tri": context,
                # FIX: giới hạn 120 ký tự, không cắt giữa từ
                "trich_dan_a": _safe_truncate(text_a, 120),
                "trich_dan_b": _safe_truncate(text_b, 120),
                "muc_do": "trung binh",
            }]

        logger.debug(f"LLM: {len(items)} thay đổi tại [{context}]")
        return items

    def summarize(self, changes: List[dict], name_a: str, name_b: str) -> str:
        if not changes:
            return "Không phát hiện thay đổi đáng kể."
        meaningful = [c for c in changes if c.get("change_type") not in ("KHONG_DOI", "KHONG DOI NOI DUNG")]
        if not meaningful:
            return "Hai tài liệu có nội dung tương đương."
        prompt = (
            f"Tài liệu gốc: {name_a}\n"
            f"Tài liệu mới: {name_b}\n\n"
            f"Danh sách thay đổi:\n"
            f"{json.dumps(meaningful[:12], ensure_ascii=False, indent=2)}\n\n"
            "Viết tóm tắt 3-5 câu về các thay đổi quan trọng nhất. "
            "Nêu rõ điều khoản cụ thể và mức độ ảnh hưởng."
        )
        return self._chat(prompt, self._SYS_SUMMARY)

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Private helpers ───────────────────────────────────────────────

    def _fix_item(self, d: dict) -> dict:
        """Chuẩn hoá muc_do và change_type về dạng lowercase không dấu."""
        raw_muc_do = re.sub(r"\s+", " ", str(d.get("muc_do", "")).strip().lower())
        _muc_map = {
            "cao": "cao", "high": "cao", "3": "cao",
            "trung binh": "trung binh", "trung bình": "trung binh",
            "medium": "trung binh", "tb": "trung binh", "2": "trung binh",
            "thap": "thap", "thấp": "thap", "low": "thap", "1": "thap",
        }
        d["muc_do"] = _muc_map.get(raw_muc_do, "trung binh")

        # Chuẩn hoá change_type
        raw_ct = str(d.get("change_type", "SUA")).strip().upper()
        _ct_map = {
            "SUA": "SUA", "MODIFIED": "SUA",
            "THEM": "THEM", "ADDED": "THEM", "THÊM": "THEM",
            "XOA": "XOA", "DELETED": "XOA", "XÓA": "XOA",
            "KHONG_DOI": "KHONG DOI NOI DUNG",
            "KHONG DOI NOI DUNG": "KHONG DOI NOI DUNG",
            "DOI_VI_TRI" : "DOI VI TRI", "DOI VI TRI": "REORDER",
            "UNCHANGED": "KHONG DOI NOI DUNG",
        }
        d["change_type"] = _ct_map.get(raw_ct, "SUA")
        return d

    def _no_change(self, context: str) -> dict:
        return {
            "change_type": "KHONG DOI NOI DUNG",
            "mo_ta": "Nội dung 2 đoạn giống nhau.",
            "vi_tri": context,
            "trich_dan_a": "",
            "trich_dan_b": "",
            "muc_do": "thap",
        }

    def _chat(self, prompt: str, system: Optional[str] = None) -> str:
        import time
        total_chars = len(prompt) + (len(system) if system else 0)
        est_tokens = total_chars // 3
        logger.info(
            f"[PROFILE] llm_request | prompt_chars={len(prompt)} "
            f"total_chars={total_chars} est_tokens≈{est_tokens}"
        )

        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        t0 = time.time()
        try:
            r = requests.post(
                f"{self._url}/api/chat",
                json={
                    "model": self._model,
                    "messages": msgs,
                    "stream": False,
                    "options": {
                        "temperature":    self._temp,
                        "num_predict":    self._maxt,
                        "num_ctx":        self._num_ctx,
                        "repeat_penalty": self._repeat_pen,
                        "top_p":          self._top_p,
                    },
                },
                timeout=self._tout,
            )
            r.raise_for_status()
            elapsed = time.time() - t0
            content = r.json()["message"]["content"].strip()
            logger.info(
                f"[PROFILE] llm_response | time={elapsed:.1f}s | "
                f"response_chars={len(content)}"
            )
            return content
        except requests.Timeout:
            raise TimeoutError(f"LLM timeout sau {self._tout}s.")
        except requests.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")

    @staticmethod
    def _coerce_change_item(x) -> dict | None:
        if isinstance(x, dict) and "change_type" in x:
            return x
        if isinstance(x, str):
            s = x.strip()
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "change_type" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
            m = re.search(r"\{[^{}]+\}", s)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if isinstance(obj, dict) and "change_type" in obj:
                        return obj
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _items_from_list(data: list) -> List[dict]:
        result: List[dict] = []
        for x in data:
            item = LLMEngine._coerce_change_item(x)
            if item:
                result.append(item)
        return result

    @staticmethod
    def _parse_list(raw: str) -> List[dict]:
        """
        Parse JSON array từ output của Qwen2.5-7b.
        Hỗ trợ array chứa string JSON và text thừa quanh JSON.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()

        start = cleaned.find("[")
        end   = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start: end + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, list):
                    return LLMEngine._items_from_list(data)
            except json.JSONDecodeError:
                pass

        try:
            fixed = re.sub(r",\s*([\]}])", r"\1", cleaned)
            fixed = fixed.replace("'", '"')
            start2 = fixed.find("[")
            end2   = fixed.rfind("]")
            if start2 != -1 and end2 > start2:
                data = json.loads(fixed[start2: end2 + 1])
                if isinstance(data, list):
                    return LLMEngine._items_from_list(data)
        except json.JSONDecodeError:
            pass

        result = []
        for obj_str in re.findall(r"\{[^{}]+\}", cleaned, re.DOTALL):
            item = LLMEngine._coerce_change_item(obj_str)
            if item:
                result.append(item)
        if not result and raw.strip() not in ("[]", ""):
            logger.warning(f"[LLM_PARSE_FAIL] Không parse được JSON | raw={raw[:500]}")
        return result


def _safe_truncate(text: str, max_chars: int) -> str:
    """Cắt text không quá max_chars, ưu tiên cắt tại khoảng trắng."""
    if len(text) <= max_chars:
        return text.strip()
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.strip() + "…"
