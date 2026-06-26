from __future__ import annotations
import sys, os, logging, re, unicodedata
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Chunk, MatchedPair, ChangeItem, ChangeType, Citation, DocSource

logger = logging.getLogger(__name__)


# ── Unicode helpers ───────────────────────────────────────────────────

def _unicode_normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text.lower())
    return text.strip()


def _citation_match(needle: str, haystack: str) -> bool:
    n = _unicode_normalize(needle[:60])
    h = _unicode_normalize(haystack)
    return n in h


class Comparator:

    def __init__(self):
        import config as cfg
        self.min_citation_len = cfg.CITATION_MIN_LEN
        # Ngưỡng severity từ config (có thể tune qua .env)
        self._low_sim    = getattr(cfg, "SEVERITY_LOW_SIM_FLOOR",     0.93)
        self._low_ratio  = getattr(cfg, "SEVERITY_LOW_RATIO_FLOOR",   0.88)
        self._med_sim    = getattr(cfg, "SEVERITY_MEDIUM_SIM_FLOOR",  0.82)
        self._med_ratio  = getattr(cfg, "SEVERITY_MEDIUM_RATIO_FLOOR",0.70)
        self._hi_sim     = getattr(cfg, "SEVERITY_HIGH_SIM_FLOOR",    0.70)
        self._hi_ratio   = getattr(cfg, "SEVERITY_HIGH_RATIO_FLOOR",  0.42)
        from core.llm_engine import get_llm
        self.llm = get_llm()
        self._corpus_a: set[str] = set()
        self._corpus_b: set[str] = set()

    # ── Public ────────────────────────────────────────────────────────

    def compare_all(self, pairs: List[MatchedPair]) -> List[ChangeItem]:
        import time
        t0 = time.time()
        self._corpus_a = {
            _unicode_normalize(p.chunk_a.text)
            for p in pairs if p.chunk_a
        }
        self._corpus_b = {
            _unicode_normalize(p.chunk_b.text)
            for p in pairs if p.chunk_b
        }
        for p in pairs:
            if p.chunk_a:
                self._corpus_a.add(self._list_item_body(p.chunk_a.text))
            if p.chunk_b:
                self._corpus_b.add(self._list_item_body(p.chunk_b.text))

        results: List[ChangeItem] = []
        for pair in pairs:
            results.extend(self._compare_one(pair))
        elapsed = time.time() - t0
        logger.info(
            f"[PROFILE] compare_all={elapsed:.1f}s | "
            f"pairs={len(pairs)} | changes={len(results)}"
        )
        return results

    # ── Core comparison ───────────────────────────────────────────────

    def _compare_one(self, pair: MatchedPair) -> List[ChangeItem]:
        if pair.chunk_a and not pair.chunk_b:
            if self._content_in_other_corpus(pair.chunk_a.text, self._corpus_b):
                return []
            return [self._make_deleted(pair.chunk_a)]
        if pair.chunk_b and not pair.chunk_a:
            if self._content_in_other_corpus(pair.chunk_b.text, self._corpus_a):
                logger.info("[DECISION] skip_false_add | content exists in A")
                return []
            return [self._make_added(pair.chunk_b)]
        if not pair.chunk_a or not pair.chunk_b:
            return []

        items: List[ChangeItem] = []
        pending_reorder = self._detect_reorder(pair.chunk_a, pair.chunk_b)

        def _finish(result: List[ChangeItem]) -> List[ChangeItem]:
            substantive = (
                ChangeType.MODIFIED, ChangeType.ADDED, ChangeType.DELETED,
            )
            if any(c.change_type in substantive for c in result):
                return [c for c in result if c.change_type != ChangeType.REORDERED]
            if result:
                return result
            return [pending_reorder] if pending_reorder else []

        if self._normalized_equal(pair.chunk_a.text, pair.chunk_b.text):
            logger.info("[DECISION] exact_match")
            return _finish(items) or [ChangeItem(
                change_type=ChangeType.UNCHANGED,
                mo_ta="Nội dung 2 đoạn tương đương sau khi chuẩn hóa.",
                vi_tri=self._resolve_vi_tri("", pair.chunk_a, pair.chunk_b),
                citation_a=None, citation_b=None, muc_do="thap",
            )]

        ca, cb = pair.chunk_a, pair.chunk_b
        vi_tri = self._resolve_vi_tri("", ca, cb)
        sim = pair.sim_score
        ratio = SequenceMatcher(
            None,
            self._normalize(ca.text)[:2000],
            self._normalize(cb.text)[:2000],
        ).ratio()

        rule_raw = self._detect_rule_changes(ca.text, cb.text)
        list_raw = self._detect_list_reorder_changes(ca.text, cb.text, ca, cb)
        line_raw = self._detect_line_diff_changes(ca.text, cb.text)
        diff_raw = self._detect_token_diff_changes(ca.text, cb.text)
        picked  = self._merge_raw_changes(rule_raw, diff_raw, line_raw)
        if not picked and self._critical_legal_signal_changed(ca.text, cb.text):
            picked = self._merge_raw_changes(
                self._detect_isolated_number_diff(ca.text, cb.text), [], [],
            )

        if list_raw and not picked and ratio >= 0.95 and len(list_raw) >= 2:
            logger.info(
                f"[DECISION] list_reorder | sim={sim:.3f} ratio={ratio:.3f} "
                f"n={len(list_raw)} vi_tri=[{vi_tri}]"
            )
            items.extend(self._raw_list_to_items(list_raw, ca, cb, sim))
            return _finish(items)

        # Bước 2 — LOW DIFFERENCE: không gọi LLM
        if sim >= self._low_sim and ratio >= self._low_ratio:
            logger.info(
                f"[DECISION] low_diff | sim={sim:.3f} ratio={ratio:.3f} vi_tri=[{vi_tri}]"
            )
            if picked:
                items.extend(self._raw_list_to_items(picked, ca, cb, sim))
                return _finish(items)
            if self._critical_legal_signal_changed(ca.text, cb.text):
                items.append(self._fallback_modified(ca, cb, vi_tri, sim))
                return _finish(items)
            items.append(self._generic_modified(
                ca, cb, vi_tri, sim, ratio,
                "Khác biệt nhỏ, nội dung gần tương đương.",
                "thap",
            ))
            return _finish(items)

        # Bước 3 — MEDIUM: rule/diff trước; không LLM nếu sim>0.95 & ratio>0.90
        if sim >= self._med_sim and ratio >= self._med_ratio:
            if picked:
                logger.info(
                    f"[DECISION] medium_rule | sim={sim:.3f} ratio={ratio:.3f} "
                    f"n={len(picked)} vi_tri=[{vi_tri}]"
                )
                items.extend(self._raw_list_to_items(picked, ca, cb, sim))
                return _finish(items)
            if sim > 0.95 and ratio > 0.90:
                logger.info(
                    f"[DECISION] fallback | sim={sim:.3f} ratio={ratio:.3f} vi_tri=[{vi_tri}]"
                )
                items.append(self._fallback_modified(ca, cb, vi_tri, sim))
                return _finish(items)

        # Không LLM cho cặp similarity cực cao dù có tín hiệu pháp lý
        if sim > 0.95 and ratio > 0.90:
            if picked:
                logger.info(
                    f"[DECISION] high_sim_rule | sim={sim:.3f} ratio={ratio:.3f} vi_tri=[{vi_tri}]"
                )
                items.extend(self._raw_list_to_items(picked, ca, cb, sim))
                return _finish(items)
            logger.info(
                f"[DECISION] fallback | sim={sim:.3f} ratio={ratio:.3f} vi_tri=[{vi_tri}]"
            )
            items.append(self._fallback_modified(ca, cb, vi_tri, sim))
            return _finish(items)

        # Bước 4 — LARGE DIFFERENCE: chỉ gọi LLM khi sim/ratio dưới ngưỡng medium
        if picked and not self._should_call_llm(sim, ratio):
            logger.info(
                f"[DECISION] rule_engine | sim={sim:.3f} ratio={ratio:.3f} "
                f"n={len(picked)} vi_tri=[{vi_tri}]"
            )
            items.extend(self._raw_list_to_items(picked, ca, cb, sim))
            return _finish(items)

        if not self._should_call_llm(sim, ratio):
            logger.info(
                f"[DECISION] fallback | sim={sim:.3f} ratio={ratio:.3f} vi_tri=[{vi_tri}]"
            )
            items.append(self._fallback_modified(ca, cb, vi_tri, sim))
            return _finish(items)

        meta_context = self._build_meta_context(ca, cb)
        excerpt_a, excerpt_b = self._extract_diff_excerpts(ca.text, cb.text)
        logger.info(
            f"[DECISION] llm | sim={sim:.3f} ratio={ratio:.3f} "
            f"excerpt={len(excerpt_a)}/{len(excerpt_b)} vi_tri=[{vi_tri}]"
        )
        try:
            raw_items = self.llm.compare_chunks(
                excerpt_a, excerpt_b, context=meta_context
            )
        except (TimeoutError, RuntimeError) as e:
            logger.warning(f"LLM lỗi tại [{meta_context}]: {e}")
            items.append(self._fallback_modified(ca, cb, vi_tri, sim))
            return _finish(items)

        for d in raw_items:
            item = self._build_item(d, ca, cb, sim)
            if item:
                items.append(item)

        if not items:
            items.append(self._fallback_modified(ca, cb, vi_tri, sim))

        return _finish(items)

    # ── Reorder detection ─────────────────────────────────────────────

    def _detect_reorder(self, ca: Chunk, cb: Chunk) -> ChangeItem | None:
        if self._normalized_equal(ca.text, cb.text):
            return None

        ratio = SequenceMatcher(
            None,
            self._normalize(ca.text)[:2000],
            self._normalize(cb.text)[:2000],
        ).ratio()
        if ratio < 0.95:
            return None

        dieu_a = (ca.metadata.dieu or "").strip()
        dieu_b = (cb.metadata.dieu or "").strip()

        def norm(s: str) -> str:
            return re.sub(r"[\.\s]+$", "", _unicode_normalize(s))

        heading_changed = bool(dieu_a and dieu_b and norm(dieu_a) != norm(dieu_b))
        index_shifted   = abs((ca.metadata.chunk_index or 0) - (cb.metadata.chunk_index or 0)) >= 3
        if not heading_changed and not index_shifted:
            return None

        if heading_changed:
            dieu_a_clean = self._extract_dieu_number(dieu_a)
            dieu_b_clean = self._extract_dieu_number(dieu_b)
            mo_ta = f"Điều khoản đổi số thứ tự: '{dieu_a_clean}' → '{dieu_b_clean}'."
            vi_tri = f"{dieu_a_clean} (A) → {dieu_b_clean} (B)"
        else:
            vi_tri_a = self._resolve_vi_tri("", ca, ca)
            vi_tri_b = self._resolve_vi_tri("", cb, cb)
            mo_ta = "Nội dung tương ứng được di chuyển sang vị trí khác."
            vi_tri = f"{vi_tri_a} → {vi_tri_b}"

        return ChangeItem(
            change_type=ChangeType.REORDERED, mo_ta=mo_ta, vi_tri=vi_tri,
            citation_a=Citation(source=DocSource.A, text=ca.text[:200].strip(),
                                heading_path=ca.metadata.heading_path,
                                chunk_index=ca.metadata.chunk_index),
            citation_b=Citation(source=DocSource.B, text=cb.text[:200].strip(),
                                heading_path=cb.metadata.heading_path,
                                chunk_index=cb.metadata.chunk_index),
            muc_do="trung binh",
        )

    # ── Added / Deleted ───────────────────────────────────────────────

    def _make_deleted(self, chunk: Chunk) -> ChangeItem:
        vi_tri = self._resolve_vi_tri("", chunk, chunk)
        return ChangeItem(
            change_type=ChangeType.DELETED,
            mo_ta=f"Điều khoản bị xóa: {vi_tri}",
            vi_tri=vi_tri,
            citation_a=Citation(source=DocSource.A, text=chunk.text[:300].strip(),
                                heading_path=chunk.metadata.heading_path,
                                chunk_index=chunk.metadata.chunk_index),
            citation_b=None, muc_do="cao",
        )

    def _make_added(self, chunk: Chunk) -> ChangeItem:
        vi_tri = self._resolve_vi_tri("", chunk, chunk)
        return ChangeItem(
            change_type=ChangeType.ADDED,
            mo_ta=f"Điều khoản thêm mới: {vi_tri}",
            vi_tri=vi_tri,
            citation_a=None,
            citation_b=Citation(source=DocSource.B, text=chunk.text[:300].strip(),
                                heading_path=chunk.metadata.heading_path,
                                chunk_index=chunk.metadata.chunk_index),
            muc_do="cao",
        )

    # ── Build ChangeItem từ LLM output ───────────────────────────────

    def _build_item(self, d: dict, ca: Chunk, cb: Chunk, sim_score: float) -> ChangeItem | None:
        raw_type  = d.get("change_type", "SUA")
        _type_map = {
            "THEM":               ChangeType.ADDED,
            "XOA":                ChangeType.DELETED,
            "SUA":                ChangeType.MODIFIED,
            "KHONG DOI NOI DUNG": ChangeType.UNCHANGED,
            "KHONG_DOI":          ChangeType.UNCHANGED,
            "DOI VI TRI":         ChangeType.REORDERED,
        }
        change_type = _type_map.get(str(raw_type).strip().upper(), ChangeType.MODIFIED)

        mo_ta = d.get("mo_ta", "").strip()
        if not mo_ta:
            return None

        citation_a = self._extract_citation(
            d.get("trich_dan_a", ""), ca, DocSource.A
        )
        citation_b = self._extract_citation(
            d.get("trich_dan_b", ""), cb, DocSource.B
        )

        llm_muc_do       = d.get("muc_do", "trung binh")
        heuristic_muc_do, ly_giai = self._infer_severity(ca.text, cb.text, sim_score)
        has_critical     = self._critical_legal_signal_changed(ca.text, cb.text)
        muc_do           = self._pick_severity(
            change_type, llm_muc_do, heuristic_muc_do, has_critical
        )

        vi_tri = self._resolve_vi_tri(d.get("vi_tri", ""), ca, cb)

        return ChangeItem(
            change_type=change_type, mo_ta=mo_ta, vi_tri=vi_tri,
            citation_a=citation_a, citation_b=citation_b,
            muc_do=muc_do, ly_giai=ly_giai,
        )

    # ── Citation extraction ───────────────────────────────────────────

    def _extract_citation(
        self, trich_dan: str, chunk: Chunk, source: DocSource
    ) -> Citation | None:
        
        trich = trich_dan.strip()
        if not trich or len(trich) < self.min_citation_len:
            return None

        # Kiểm tra 60 ký tự đầu của trích dẫn có trong chunk không
        if _citation_match(trich, chunk.text):
            return Citation(
                source=source, text=trich,
                heading_path=chunk.metadata.heading_path,
                chunk_index=chunk.metadata.chunk_index,
            )

        # Fallback: tìm từ khóa dài nhất từ trich_dan trong chunk
        key = _unicode_normalize(trich[:40])
        if len(key) >= 10 and key in _unicode_normalize(chunk.text):
            return Citation(
                source=source, text=trich,
                heading_path=chunk.metadata.heading_path,
                chunk_index=chunk.metadata.chunk_index,
            )

        logger.debug(
            f"Citation {source.value} không khớp chunk [{chunk.metadata.heading_path}]: "
            f"'{trich[:50]}'"
        )
        return None

    # ── vi_tri resolution ─────────────────────────────────────────────

    _GENERIC_HEADINGS = ("phan dau tai lieu", "doan ")

    @staticmethod
    def _is_generic_heading(hp: str) -> bool:
        low = (hp or "").strip().lower()
        return not low or any(low.startswith(g) for g in Comparator._GENERIC_HEADINGS)

    @staticmethod
    def _infer_vi_tri_from_text(text: str) -> str:
        """Suy vị trí từ nội dung chunk khi metadata không có cấu trúc."""
        if not text:
            return ""
        m = re.search(
            r"(?:Điều|ĐIỀU|Dieu|DIEU)\s+(\d+)\s*[\.\:]?\s*([^\n]{0,80})?",
            text[:800],
            re.IGNORECASE,
        )
        if m:
            num = m.group(1)
            title = (m.group(2) or "").strip().rstrip(".:")
            if title:
                return f"Điều {num}: {title}"
            return f"Điều {num}"
        m = re.search(r"(?:^|\n)\s*(\d+(?:\.\d+)+)\.?\s+\S", text[:800], re.MULTILINE)
        if m:
            kn = m.group(1)
            return f"Điều {kn.split('.')[0]} > Khoản {kn}"
        return ""

    @staticmethod
    def _extract_dieu_number(dieu_raw: str) -> str:
        """Trích 'Điều X' từ chuỗi dieu metadata, bỏ phần tiêu đề phía sau.
        Ví dụ: 'Điều 3. Thời hạn thuê' → 'Điều 3'
        """
        if not dieu_raw:
            return ""
        m = re.match(r"((?:Điều|Dieu|Phụ lục|Phu luc)\s+[\d\w]+)", dieu_raw, re.IGNORECASE)
        return m.group(1).strip() if m else dieu_raw.strip()

    def _resolve_vi_tri(
        self, llm_vi_tri: str, ca: Chunk, cb: Chunk, include_diem: bool = True,
    ) -> str:
        """
        Build vi_tri từ metadata chunk.
        Format chuẩn: "Điều X > Khoản Y > Điểm Z"
        """
        if llm_vi_tri and llm_vi_tri.strip():
            return llm_vi_tri.strip()

        dieu_raw = khoan_raw = diem_raw = ""
        hp_a = hp_b = ""
        for chunk in (ca, cb):
            if not chunk or not chunk.metadata:
                continue
            m = chunk.metadata
            dieu_raw = dieu_raw or (m.dieu or "")
            khoan_raw = khoan_raw or (m.khoan or "")
            diem_raw = diem_raw or (m.diem or "")
            if chunk is ca:
                hp_a = m.heading_path or ""
            else:
                hp_b = m.heading_path or ""

        parts = []
        dieu_clean = self._extract_dieu_number(dieu_raw)
        if dieu_clean:
            parts.append(dieu_clean)
        if khoan_raw:
            parts.append(f"Khoản {khoan_raw.rstrip('.').rstrip(')')}")
        if include_diem and diem_raw:
            parts.append(f"Điểm {diem_raw}")

        hp = hp_a or hp_b
        if parts and not self._is_generic_heading(hp):
            return " > ".join(parts)

        combined = "\n".join(filter(None, [
            ca.text if ca else "",
            cb.text if cb else "",
        ]))
        inferred = self._infer_vi_tri_from_text(combined)
        if inferred:
            return inferred
        if parts:
            return " > ".join(parts)
        if hp and not self._is_generic_heading(hp):
            return hp
        return hp or ""

    # ── Severity logic ────────────────────────────────────────────────

    def _infer_severity(self, text_a: str, text_b: str, sim_score: float) -> tuple[str, str]:
        a_norm = self._normalize(text_a)
        b_norm = self._normalize(text_b)
        ratio  = SequenceMatcher(None, a_norm[:2000], b_norm[:2000]).ratio()

        has_critical = self._critical_legal_signal_changed(text_a, text_b)

        # Critical signal đổi → luôn là cao
        if has_critical:
            return "cao", "Thay đổi số liệu, chủ thể, phủ định hoặc điều kiện ràng buộc."

        # FIX: dùng ngưỡng từ config thay vì hardcode
        if sim_score >= self._low_sim and ratio >= self._low_ratio:
            return "thap", "Chủ yếu thay đổi cách diễn đạt, nội dung gần tương đương."

        if sim_score >= self._med_sim and ratio >= self._med_ratio:
            # Vùng xám: nâng lên "cao" nếu nội dung chứa từ khoá pháp lý nhạy cảm
            if self._contains_sensitive_legal_terms(text_a, text_b):
                return "cao", "Nội dung liên quan đến điều khoản tài chính/thời hạn/trách nhiệm — cần xem xét kỹ."
            return "trung binh", "Nội dung có sửa đổi một phần nhưng vẫn giữ nhiều thành phần chung."

        if sim_score < self._hi_sim or ratio < self._hi_ratio:
            return "cao", "Nội dung thay đổi lớn, có dấu hiệu viết lại hoặc lệch nghĩa."

        return "trung binh", "Độ tương đồng trung bình, cần xem xét kỹ."

    @staticmethod
    def _pick_severity(
        change_type: ChangeType,
        llm_muc_do: str,
        heuristic_muc_do: str,
        has_critical: bool,
    ) -> str:
        if change_type in (ChangeType.ADDED, ChangeType.DELETED):
            return "cao"
        if change_type == ChangeType.REORDERED:
            return "trung binh"

        order = {"thap": 1, "trung binh": 2, "cao": 3}
        llm_norm = str(llm_muc_do).strip().lower()
        if llm_norm not in order:
            llm_norm = "trung binh"

        llm_score = order[llm_norm]
        heu_score = order.get(heuristic_muc_do, 2)

        # Nếu LLM và heuristic đều đánh giá thấp → tin thap
        if llm_score == 1 and heu_score == 1 and not has_critical:
            return "thap"

        # Nếu có critical signal → ít nhất trung bình
        result_score = max(llm_score, heu_score)
        if has_critical and result_score < 2:
            result_score = 2

        return {1: "thap", 2: "trung binh", 3: "cao"}[result_score]

    # ── Critical signal detection ─────────────────────────────────────

    @staticmethod
    def _critical_legal_signal_changed(text_a: str, text_b: str) -> bool:
        a_low = _unicode_normalize(text_a)
        b_low = _unicode_normalize(text_b)

        patterns = {
            "money":    r"\b\d[\d\.,]*\s*(?:vnd|vnđ|đồng|usd|%)\b",
            "date":     r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "duration": (
                r"(?:\.{2,}|[\u2026]+|_{2,})?\s*\d+\s*(?:\.{2,}|[\u2026]+|_{2,})?"
                r"\s*(?:ngay|thang|nam|gio|ngày|tháng|năm|giờ)\b"
                r"|(?:\.{2,}|[\u2026]+|_{2,})\s*\d+\s*(?:\.{2,}|[\u2026]+|_{2,})"
            ),
            "actor":    r"\b(?:bên a|bên b|ben a|ben b|bên mua|bên bán|ben mua|ben ban)\b",
            "negation": r"\b(?:không|khong|chưa|chua|cấm|cam|không được|khong duoc)\b",
        }
        for pat in patterns.values():
            if Counter(re.findall(pat, a_low)) != Counter(re.findall(pat, b_low)):
                return True

        # Số liệu tổng quát (FIX: dùng Counter)
        if Counter(re.findall(r"\d+(?:[\.,]\d+)?", a_low)) != \
           Counter(re.findall(r"\d+(?:[\.,]\d+)?", b_low)):
            return True

        # Từ khoá pháp lý quan trọng
        keywords = (
            "bồi thường", "boi thuong",
            "phạt vi phạm", "phat vi pham",
            "chấm dứt", "cham dut",
            "gia hạn", "thanh toán", "thanh toan",
            "nghĩa vụ", "nghia vu",
            "bảo mật", "bao mat",
            "sở hữu trí tuệ", "so huu tri tue",
            "đơn phương", "don phuong",
            "bất khả kháng", "bat kha khang",
        )
        set_a = {k for k in keywords if k in a_low}
        set_b = {k for k in keywords if k in b_low}
        return set_a != set_b

    # ── LLM gate & rule/diff helpers ──────────────────────────────────

    def _should_call_llm(self, sim: float, ratio: float) -> bool:
        """Chỉ gọi LLM ở vùng LARGE: sim/ratio dưới ngưỡng medium."""
        if sim > 0.95 and ratio > 0.90:
            return False
        if sim >= self._low_sim and ratio >= self._low_ratio:
            return False
        return sim < self._med_sim or ratio < self._med_ratio

    def _raw_list_to_items(
        self, raw: List[dict], ca: Chunk, cb: Chunk, sim: float,
    ) -> List[ChangeItem]:
        items: List[ChangeItem] = []
        for d in raw:
            item = self._build_item(d, ca, cb, sim)
            if item:
                items.append(item)
        return items

    def _generic_modified(
        self, ca: Chunk, cb: Chunk, vi_tri: str,
        sim: float, ratio: float, mo_ta: str, muc_do: str,
    ) -> ChangeItem:
        _, ly_giai = self._infer_severity(ca.text, cb.text, sim)
        return ChangeItem(
            change_type=ChangeType.MODIFIED,
            mo_ta=mo_ta,
            vi_tri=vi_tri,
            citation_a=Citation(
                source=DocSource.A, text=ca.text[:220].strip(),
                heading_path=ca.metadata.heading_path,
                chunk_index=ca.metadata.chunk_index,
            ),
            citation_b=Citation(
                source=DocSource.B, text=cb.text[:220].strip(),
                heading_path=cb.metadata.heading_path,
                chunk_index=cb.metadata.chunk_index,
            ),
            muc_do=muc_do,
            ly_giai=ly_giai or f"sim={sim:.3f}, ratio={ratio:.3f}",
        )

    @staticmethod
    def _snippet(text: str, needle: str, width: int = 120) -> str:
        if not text:
            return ""
        if not needle:
            return text[:width].strip()
        low, key = _unicode_normalize(text), _unicode_normalize(needle)
        idx = low.find(key)
        if idx < 0:
            return text[:width].strip()
        start, end = max(0, idx - 25), min(len(text), idx + len(needle) + 50)
        out = text[start:end].strip()
        if start > 0:
            out = "…" + out
        if end < len(text):
            out = out + "…"
        return out

    def _diff_counter_pattern(
        self, pattern: str, label: str, text_a: str, text_b: str,
    ) -> List[dict]:
        a_low, b_low = _unicode_normalize(text_a), _unicode_normalize(text_b)
        found_a = re.findall(pattern, a_low)
        found_b = re.findall(pattern, b_low)
        if Counter(found_a) == Counter(found_b):
            return []

        only_a = list((Counter(found_a) - Counter(found_b)).elements())
        only_b = list((Counter(found_b) - Counter(found_a)).elements())
        changes: List[dict] = []
        pairs = min(len(only_a), len(only_b))

        for i in range(pairs):
            old_v, new_v = only_a[i], only_b[i]
            changes.append({
                "change_type": "SUA",
                "mo_ta": f"Thay đổi {label}: '{old_v}' → '{new_v}'",
                "trich_dan_a": self._snippet(text_a, old_v),
                "trich_dan_b": self._snippet(text_b, new_v),
                "muc_do": "cao",
            })
        for v in only_a[pairs:]:
            changes.append({
                "change_type": "XOA",
                "mo_ta": f"Xóa {label}: '{v}'",
                "trich_dan_a": self._snippet(text_a, v),
                "trich_dan_b": "",
                "muc_do": "cao",
            })
        for v in only_b[pairs:]:
            changes.append({
                "change_type": "THEM",
                "mo_ta": f"Thêm {label}: '{v}'",
                "trich_dan_a": "",
                "trich_dan_b": self._snippet(text_b, v),
                "muc_do": "cao",
            })
        return changes

    _RE_FILL_NUM = (
        r"(?:\.{2,}|[\u2026]+|_{2,})\s*\d+\s*(?:\.{2,}|[\u2026]+|_{2,})"
    )
    _RE_DURATION = (
        r"(?:\.{2,}|[\u2026]+|_{2,})?\s*\d+\s*(?:\.{2,}|[\u2026]+|_{2,})?"
        r"\s*(?:ngày|ngay|tháng|thang|năm|nam|giờ|gio)\b"
    )

    def _detect_isolated_number_diff(self, text_a: str, text_b: str) -> List[dict]:
        """Bắt 10→100 trong ô điền (..10..) khi detector chính không khớp format."""
        changes: List[dict] = []
        changes.extend(self._diff_counter_pattern(
            self._RE_FILL_NUM, "số điền", text_a, text_b,
        ))
        changes.extend(self._diff_counter_pattern(
            self._RE_DURATION, "thời hạn", text_a, text_b,
        ))
        if changes:
            return changes[:4]

        a_low, b_low = _unicode_normalize(text_a), _unicode_normalize(text_b)
        a_nums = re.findall(r"(?<![.\d])\d+(?![.\d])", a_low)
        b_nums = re.findall(r"(?<![.\d])\d+(?![.\d])", b_low)
        if Counter(a_nums) == Counter(b_nums):
            return []

        only_a = list((Counter(a_nums) - Counter(b_nums)).elements())
        only_b = list((Counter(b_nums) - Counter(a_nums)).elements())
        for old_v, new_v in zip(only_a, only_b):
            if old_v == new_v:
                continue
            changes.append({
                "change_type": "SUA",
                "mo_ta": f"Thay đổi số liệu: '{old_v}' → '{new_v}'",
                "trich_dan_a": self._snippet(text_a, old_v),
                "trich_dan_b": self._snippet(text_b, new_v),
                "muc_do": "cao",
            })
        return changes[:4]

    def _detect_rule_changes(self, text_a: str, text_b: str) -> List[dict]:
        """Rule-based: số, thời hạn, ngày, chủ thể, phủ định, tiền tệ."""
        changes: List[dict] = []
        changes.extend(self._diff_counter_pattern(
            r"\d[\d\.,]*\s*(?:%|vnd|vnđ|đồng|usd)\b", "tỉ lệ/tiền tệ", text_a, text_b,
        ))
        changes.extend(self._diff_counter_pattern(
            r"\d[\d\.,]*\s*(?:triệu|trieu|tỷ|ty|tỷ đồng|ty dong)\b", "số tiền", text_a, text_b,
        ))
        changes.extend(self._diff_counter_pattern(
            self._RE_DURATION, "thời hạn", text_a, text_b,
        ))
        changes.extend(self._diff_counter_pattern(
            self._RE_FILL_NUM, "số điền", text_a, text_b,
        ))
        changes.extend(self._diff_counter_pattern(
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "ngày tháng", text_a, text_b,
        ))
        changes.extend(self._diff_counter_pattern(
            r"\b(?:bên\s*[ab]|ben\s*[ab]|bên\s*thuê|ben\s*thue|bên\s*cho\s*thuê|"
            r"bên\s*mua|bên\s*bán|ben\s*mua|ben\s*ban)\b",
            "chủ thể", text_a, text_b,
        ))
        changes.extend(self._diff_counter_pattern(
            r"\b(?:không\s+được|khong\s+duoc|không\s+có\s+quyền|khong\s+co\s+quyen|"
            r"cấm|cam|chưa\s+được|chua\s*duoc)\b",
            "phủ định/quyền", text_a, text_b,
        ))
        if not changes:
            changes.extend(self._diff_counter_pattern(
                r"\d+(?:[\.,]\d+)?", "số liệu", text_a, text_b,
            ))
        return changes[:8]

    _LIST_ITEM_RE = re.compile(
        r"^([a-z\u0111])\)\s*(.+)$",
        re.MULTILINE | re.IGNORECASE,
    )

    @staticmethod
    def _parse_list_items(text: str) -> dict[str, str]:
        """Map điểm (a, b, c, đ) → nội dung đã chuẩn hóa."""
        items: dict[str, str] = {}
        for m in Comparator._LIST_ITEM_RE.finditer(text):
            letter = m.group(1).lower()
            body = _unicode_normalize(m.group(2).rstrip(";."))
            if body:
                items[letter] = body
        return items

    @staticmethod
    def _list_item_body(text: str) -> str:
        """Bỏ nhãn điểm (a), b), đ)...) — chỉ lấy nội dung."""
        m = Comparator._LIST_ITEM_RE.match(text.strip())
        if m:
            return _unicode_normalize(m.group(2).rstrip(";."))
        return _unicode_normalize(text)

    def _content_in_other_corpus(self, text: str, corpus: set[str]) -> bool:
        """Nội dung chunk đã có ở phía đối diện → không báo THEM/XOA giả."""
        if not text or not corpus:
            return False
        norm = _unicode_normalize(text)
        if norm in corpus:
            return True
        body = self._list_item_body(text)
        return len(body) >= 8 and body in corpus

    def _detect_list_reorder_changes(
        self, text_a: str, text_b: str, ca: Chunk, cb: Chunk,
    ) -> List[dict] | None:
        """
        Phát hiện đổi thứ tự các điểm a/b/c/d/đ khi nội dung giữ nguyên.
        Trả None nếu không phải danh sách điểm; [] nếu danh sách giống hệt; list dict nếu reorder.
        """
        items_a = self._parse_list_items(text_a)
        items_b = self._parse_list_items(text_b)
        if len(items_a) < 2 or len(items_b) < 2:
            return None

        bodies_a = Counter(items_a.values())
        bodies_b = Counter(items_b.values())
        if bodies_a != bodies_b:
            return None

        if items_a == items_b:
            return []

        body_ratio = SequenceMatcher(
            None, _unicode_normalize(text_a), _unicode_normalize(text_b),
        ).ratio()
        if body_ratio < 0.95:
            return None

        # Cùng tập nội dung, khác gán nhãn điểm → reorder
        changes: List[dict] = []
        body_to_letter_a = {v: k for k, v in items_a.items()}
        body_to_letter_b = {v: k for k, v in items_b.items()}

        vi_tri_khoan = self._resolve_vi_tri("", ca, cb, include_diem=False)

        for body in sorted(bodies_a.keys()):
            la = body_to_letter_a.get(body, "")
            lb = body_to_letter_b.get(body, "")
            if la and lb and la != lb:
                snippet = body[:70] + ("…" if len(body) > 70 else "")
                changes.append({
                    "change_type": "DOI VI TRI",
                    "mo_ta": (
                        f"Đổi thứ tự điểm: nội dung '{snippet}' "
                        f"từ điểm {la}) sang điểm {lb})"
                    ),
                    "vi_tri": vi_tri_khoan,
                    "trich_dan_a": self._find_list_line(text_a, la),
                    "trich_dan_b": self._find_list_line(text_b, lb),
                    "muc_do": "trung binh",
                })

        if len(changes) >= 2:
            return changes[:6]

        return None

    @staticmethod
    def _find_list_line(text: str, letter: str) -> str:
        for m in Comparator._LIST_ITEM_RE.finditer(text):
            if m.group(1).lower() == letter.lower():
                return m.group(0).strip()[:120]
        return ""

    _RE_BULLET = re.compile(r"^[-–•]\s*")

    @staticmethod
    def _merge_raw_changes(*groups: List[dict]) -> List[dict]:
        merged: List[dict] = []
        seen: set[str] = set()
        phrase_keys: set[str] = set()

        for group in groups:
            for d in group:
                mo = d.get("mo_ta", "")
                if d.get("change_type") == "SUA" and mo.startswith("Thay đổi cụm từ"):
                    ta = _unicode_normalize(d.get("trich_dan_a", ""))[:80]
                    tb = _unicode_normalize(d.get("trich_dan_b", ""))[:80]
                    if ta or tb:
                        phrase_keys.add(f"{ta}|{tb}")

        for group in groups:
            for d in group:
                mo = d.get("mo_ta", "")
                if mo.startswith("Sửa dòng:") and phrase_keys:
                    ta = _unicode_normalize(d.get("trich_dan_a", ""))[:80]
                    tb = _unicode_normalize(d.get("trich_dan_b", ""))[:80]
                    if any(
                        (ta and ta in k) or (tb and tb in k) for k in phrase_keys
                    ):
                        continue
                key = mo[:100]
                if key and key not in seen:
                    seen.add(key)
                    merged.append(d)
        return merged[:10]

    @staticmethod
    def _split_content_lines(text: str) -> List[str]:
        lines: List[str] = []
        for raw in text.splitlines():
            s = raw.strip()
            if len(s) < 8:
                continue
            s = Comparator._RE_BULLET.sub("", s).strip()
            if len(s) >= 8:
                lines.append(s)
        return lines

    @staticmethod
    def _line_exists_in(line: str, other_text: str) -> bool:
        body = _unicode_normalize(line)
        return len(body) >= 12 and body in _unicode_normalize(other_text)

    def _line_muc_do(self, line_a: str, line_b: str, line_ratio: float) -> str:
        a, b = _unicode_normalize(line_a), _unicode_normalize(line_b)
        dur = r"\d+\s*(?:ngày|ngay|tháng|thang|năm|nam)"
        if re.search(dur, b) and not re.search(dur, a):
            return "cao"
        if re.search(r"không|khong|cấm|cam|không được|khong duoc", a) and \
           re.search(r"không|khong|cấm|cam|không được|khong duoc", b):
            return "trung binh"
        if line_ratio >= self._low_ratio:
            return "thap"
        return "trung binh" if line_ratio >= 0.45 else "cao"

    def _emit_line_replace(
        self, olds: List[str], news: List[str],
        text_a: str, text_b: str, changes: List[dict],
    ) -> None:
        if not olds or not news:
            return
        pairs: List[tuple[str, str]] = []
        if len(olds) == len(news):
            pairs = list(zip(olds, news))
        else:
            used: set[int] = set()
            for la in olds:
                best_j, best_r = -1, 0.0
                for j, lb in enumerate(news):
                    if j in used:
                        continue
                    r = SequenceMatcher(
                        None, _unicode_normalize(la), _unicode_normalize(lb),
                    ).ratio()
                    if r > best_r:
                        best_r, best_j = r, j
                if best_j >= 0 and best_r >= 0.35:
                    used.add(best_j)
                    pairs.append((la, news[best_j]))
        for la, lb in pairs:
            if _unicode_normalize(la) == _unicode_normalize(lb):
                continue
            lr = SequenceMatcher(None, _unicode_normalize(la), _unicode_normalize(lb)).ratio()
            if lr >= 0.98:
                continue
            if lr >= 0.85:
                continue
            changes.append({
                "change_type": "SUA",
                "mo_ta": f"Sửa dòng: '{la[:65]}' → '{lb[:65]}'",
                "trich_dan_a": la[:120].strip(),
                "trich_dan_b": lb[:120].strip(),
                "muc_do": self._line_muc_do(la, lb, lr),
            })

    def _detect_line_diff_changes(self, text_a: str, text_b: str) -> List[dict]:
        """So sánh từng dòng/gạch đầu dòng — bắt sửa điều khoản dạng bullet."""
        lines_a = self._split_content_lines(text_a)
        lines_b = self._split_content_lines(text_b)
        if max(len(lines_a), len(lines_b)) < 2:
            return []

        na = [_unicode_normalize(l) for l in lines_a]
        nb = [_unicode_normalize(l) for l in lines_b]
        changes: List[dict] = []

        for tag, i1, i2, j1, j2 in SequenceMatcher(None, na, nb).get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                self._emit_line_replace(
                    lines_a[i1:i2], lines_b[j1:j2], text_a, text_b, changes,
                )
            elif tag == "delete":
                for la in lines_a[i1:i2]:
                    if self._line_exists_in(la, text_b):
                        continue
                    changes.append({
                        "change_type": "XOA",
                        "mo_ta": f"Xóa dòng: '{la[:65]}'",
                        "trich_dan_a": la[:120].strip(),
                        "trich_dan_b": "",
                        "muc_do": "trung binh",
                    })
            elif tag == "insert":
                for lb in lines_b[j1:j2]:
                    if self._line_exists_in(lb, text_a):
                        continue
                    muc = "cao" if re.search(
                        r"\d+\s*(?:ngày|ngay|tháng|thang|năm|nam)", _unicode_normalize(lb),
                    ) else "trung binh"
                    changes.append({
                        "change_type": "THEM",
                        "mo_ta": f"Thêm dòng: '{lb[:65]}'",
                        "trich_dan_a": "",
                        "trich_dan_b": lb[:120].strip(),
                        "muc_do": muc,
                    })
        return changes[:8]

    @staticmethod
    def _is_alignment_fragment(fragment: str, other_text: str) -> bool:
        """Bỏ THEM/XOA giả do lệch token khi xóa/sửa đoạn dài (vd. 'tài.')."""
        core = re.sub(r"^[\W_]+|[\W_]+$", "", _unicode_normalize(fragment))
        if not core or len(core) >= 12:
            return False
        return core in _unicode_normalize(other_text)

    def _detect_token_diff_changes(self, text_a: str, text_b: str) -> List[dict]:
        """SequenceMatcher trên token — phát hiện thêm/xóa/sửa cụm từ."""
        words_a, words_b = text_a.split(), text_b.split()
        if not words_a or not words_b:
            return []

        na = [self._normalize(w) for w in words_a]
        nb = [self._normalize(w) for w in words_b]
        changes: List[dict] = []

        for tag, i1, i2, j1, j2 in SequenceMatcher(None, na, nb).get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                old = " ".join(words_a[i1:i2])
                new = " ".join(words_b[j1:j2])
                if not old or not new or old == new:
                    continue
                pair_ratio = SequenceMatcher(
                    None, self._normalize(old), self._normalize(new),
                ).ratio()
                old_digits = re.findall(r"\d+", old)
                new_digits = re.findall(r"\d+", new)
                digits_changed = old_digits != new_digits
                if pair_ratio >= 0.95 and not digits_changed:
                    continue
                muc = "trung binh"
                if digits_changed or re.search(
                    r"không|khong|tối thiểu|toi thieu|cấm|cam|không được|khong duoc",
                    self._normalize(old + " " + new),
                ):
                    muc = "cao"
                changes.append({
                    "change_type": "SUA",
                    "mo_ta": f"Thay đổi cụm từ: '{old[:70]}' → '{new[:70]}'",
                    "trich_dan_a": self._snippet(text_a, old),
                    "trich_dan_b": self._snippet(text_b, new),
                    "muc_do": muc,
                })
            elif tag == "delete":
                old = " ".join(words_a[i1:i2])
                if len(old) >= 4:
                    if self._list_item_body(old) in _unicode_normalize(text_b):
                        continue
                    changes.append({
                        "change_type": "XOA",
                        "mo_ta": f"Xóa cụm từ: '{old[:70]}'",
                        "trich_dan_a": self._snippet(text_a, old),
                        "trich_dan_b": "",
                        "muc_do": "trung binh",
                    })
            elif tag == "insert":
                new = " ".join(words_b[j1:j2])
                if len(new) >= 4:
                    if self._list_item_body(new) in _unicode_normalize(text_a):
                        continue
                    if self._is_alignment_fragment(new, text_a):
                        continue
                    changes.append({
                        "change_type": "THEM",
                        "mo_ta": f"Thêm cụm từ: '{new[:70]}'",
                        "trich_dan_a": "",
                        "trich_dan_b": self._snippet(text_b, new),
                        "muc_do": "trung binh",
                    })
        return changes[:5]

    @staticmethod
    def _extract_diff_excerpts(text_a: str, text_b: str, context_words: int = 45) -> Tuple[str, str]:
        """Chỉ gửi đoạn quanh diff vào LLM — không gửi cả chunk dài."""
        wa, wb = text_a.split(), text_b.split()
        if not wa or not wb:
            return text_a[:500], text_b[:500]

        na = [re.sub(r"\s+", " ", w.lower()).strip() for w in wa]
        nb = [re.sub(r"\s+", " ", w.lower()).strip() for w in wb]
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, na, nb).get_opcodes():
            if tag == "equal":
                continue
            sa = max(0, i1 - context_words)
            ea = min(len(wa), i2 + context_words)
            sb = max(0, j1 - context_words)
            eb = min(len(wb), j2 + context_words)
            return " ".join(wa[sa:ea]), " ".join(wb[sb:eb])

        if len(text_a) <= 600 and len(text_b) <= 600:
            return text_a, text_b
        return text_a[:500], text_b[:500]

    # ── Utilities ─────────────────────────────────────────────────────

    def _fallback_modified(
        self, ca: Chunk, cb: Chunk, vi_tri: str, sim_score: float
    ) -> ChangeItem:
        muc_do, ly_giai = self._infer_severity(ca.text, cb.text, sim_score)
        return ChangeItem(
            change_type=ChangeType.MODIFIED,
            mo_ta=f"Nội dung thay đổi tại: {vi_tri}" if vi_tri else "Nội dung thay đổi",
            vi_tri=vi_tri,
            citation_a=Citation(source=DocSource.A, text=ca.text[:220].strip(),
                                heading_path=ca.metadata.heading_path,
                                chunk_index=ca.metadata.chunk_index),
            citation_b=Citation(source=DocSource.B, text=cb.text[:220].strip(),
                                heading_path=cb.metadata.heading_path,
                                chunk_index=cb.metadata.chunk_index),
            muc_do=muc_do, ly_giai=ly_giai,
        )

    @staticmethod
    def _build_meta_context(ca: Chunk, cb: Chunk) -> str:
        """Build chuỗi context giàu metadata cho LLM prompt.
        Ví dụ: 'Điều 5 > Khoản 2 > Điểm a (A) | Điều 5 > Khoản 2 > Điểm a (B)'
        """
        def _fmt(c: Chunk, label: str) -> str:
            m = c.metadata
            parts = []
            if m.dieu:
                parts.append(m.dieu.strip())
            if m.khoan:
                parts.append(f"Khoản {m.khoan.rstrip('.').rstrip(')')}")
            if m.diem:
                parts.append(f"Điểm {m.diem}")
            loc = " > ".join(parts) if parts else m.heading_path or ""
            return f"{loc} ({label})" if loc else label

        ctx_a = _fmt(ca, "A")
        ctx_b = _fmt(cb, "B")
        return f"{ctx_a} | {ctx_b}"

    @staticmethod
    def _contains_sensitive_legal_terms(text_a: str, text_b: str) -> bool:
        """Kiểm tra xem cả 2 đoạn text có chứa từ khoá pháp lý nhạy cảm không.
        Dùng để nâng severity trong vùng xám (medium sim_score).
        """
        combined = _unicode_normalize(text_a + " " + text_b)
        sensitive_terms = (
            "tiền", "tien",
            "thời hạn", "thoi han",
            "lãi suất", "lai suat",
            "bồi thường", "boi thuong",
            "phạt", "phat",
            "chấm dứt", "cham dut",
            "thanh toán", "thanh toan",
            "bảo lãnh", "bao lanh",
            "thế chấp", "the chap",
            "trách nhiệm", "trach nhiem",
            "nghĩa vụ", "nghia vu",
            "bảo hiểm", "bao hiem",
            "đơn phương", "don phuong",
        )
        return any(term in combined for term in sensitive_terms)

    @staticmethod
    def _normalized_equal(text_a: str, text_b: str) -> bool:
        def norm(text: str) -> str:
            text = _unicode_normalize(text)
            return re.sub(r"[^\w\s%/.-]", "", text).strip()
        return norm(text_a) == norm(text_b)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()
