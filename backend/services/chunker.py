from __future__ import annotations
import re, sys, os, logging
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import Chunk, ChunkMeta, DocSource
from services.reader import ParsedDoc

logger = logging.getLogger(__name__)

RE_DIEU    = re.compile(r"^(?:Dieu|DIEU)\s+\d+", re.IGNORECASE)
RE_DIEU_VN = re.compile(r"^(?:\u0110i\u1ec1u|\u0110I\u1ec0U)\s+\d+", re.IGNORECASE)
RE_KHOAN   = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+\S|^\d+[\.\)]\s+\S")
RE_DIEM    = re.compile(r"^([a-z\u0111\u0110]\)\s+.{0,100})$")
RE_PHU_LUC = re.compile(r"^(Ph\u1ee5\s*l\u1ee5c\s+[\dA-Z]+.{0,80})$", re.IGNORECASE)
RE_DIEU_INLINE = re.compile(
    r"(?:Điều|ĐIỀU|Dieu|DIEU)\s+(\d+)\s*[\.\:]?\s*([^\n]{0,80})?",
    re.IGNORECASE,
)
RE_KHOAN_INLINE = re.compile(r"(?:^|\n)\s*(\d+(?:\.\d+)+)\.?\s+\S", re.MULTILINE)


def _is_dieu(s: str) -> bool:
    return bool(RE_DIEU.match(s) or RE_DIEU_VN.match(s))


def _match_khoan(s: str) -> Optional[str]:
    """Trả về mã khoản: '4.3', '6.2.2', '13.1'..."""
    m = re.match(r"^(\d+(?:\.\d+)+)\.?\s+\S", s)
    if m:
        return m.group(1)
    m = re.match(r"^(\d+)[\.\)]\s*\S", s)
    if m:
        return m.group(1)
    return None


def _dieu_from_khoan(khoan: str) -> str:
    return f"Điều {khoan.split('.')[0]}"


def _infer_structure_from_lines(lines: List[str]) -> tuple[Optional[str], Optional[str], str]:
    """Suy Điều/Khoản từ vài dòng đầu chunk khi metadata trống."""
    text = "\n".join(lines[:8])
    m = RE_DIEU_INLINE.search(text)
    if m:
        num, title = m.group(1), (m.group(2) or "").strip().rstrip(".:")
        dieu = f"Điều {num}" + (f": {title}" if title else "")
        return dieu, None, dieu
    m = RE_KHOAN_INLINE.search(text)
    if m:
        kn = m.group(1)
        dieu = _dieu_from_khoan(kn)
        return dieu, kn, f"{dieu} > Khoản {kn}"
    return None, None, ""


def _heading_path(dieu: Optional[str], khoan: Optional[str], diem: Optional[str]) -> str:
    parts = [p for p in [dieu, khoan, diem] if p]
    return " > ".join(parts)


class Chunker:

    def __init__(self):
        import config as cfg
        self.max_size = cfg.CHUNK_MAX
        self.overlap  = cfg.CHUNK_OVL
        self.min_size = cfg.CHUNK_MIN

    def chunk(
        self, doc: ParsedDoc, doc_id: str,
        source: DocSource, session_id: str,
    ) -> List[Chunk]:
        lines    = [p.text for p in doc.paragraphs]
        page_map = {i: p.page for i, p in enumerate(doc.paragraphs)}

        chunks = self._structural(lines, page_map, doc_id, source, session_id)
        if len(chunks) < 3:
            logger.info(f"Khong nhan cau truc ro ({doc_id}) -> sliding window")
            chunks = self._sliding("\n".join(lines), doc_id, source, session_id)

        chunks = [c for c in chunks if len(c.text.strip()) >= self.min_size]
        logger.info(f"{len(chunks)} chunks | {doc_id} source={source.value}")
        return chunks

    def _structural(
        self, lines: List[str], page_map: dict,
        doc_id: str, source: DocSource, session_id: str,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        idx = 0
        cur_dieu:  Optional[str] = None
        cur_khoan: Optional[str] = None
        cur_diem:  Optional[str] = None
        cur_lines: List[str]     = []
        cur_page:  int           = 0

        def flush():
            nonlocal idx, cur_khoan, cur_diem
            if not cur_lines:
                return
            body = "\n".join(cur_lines).strip()
            if len(body) < self.min_size:
                return

            dieu, khoan, diem = cur_dieu, cur_khoan, cur_diem
            hp = _heading_path(dieu, khoan, diem)
            if not hp:
                idieu, ikhoan, ihp = _infer_structure_from_lines(cur_lines)
                dieu = dieu or idieu
                khoan = khoan or ikhoan
                hp = ihp or _heading_path(dieu, khoan, diem) or "Phan dau tai lieu"
            chunks.append(Chunk(
                text=body,
                metadata=ChunkMeta(
                    doc_id=doc_id, source=source, session_id=session_id,
                    dieu=dieu, khoan=khoan, diem=diem,
                    page=cur_page, chunk_index=idx, heading_path=hp,
                ),
            ))
            idx += 1

        for li, line in enumerate(lines):
            s    = line.strip()
            page = page_map.get(li, 0)
            if not s:
                continue

            if RE_PHU_LUC.match(s) or _is_dieu(s):
                flush()
                cur_dieu  = s; cur_khoan = None; cur_diem = None
                cur_lines = [s]; cur_page = page

            elif (kn := _match_khoan(s)):
                if len(cur_lines) > 1 or (len(cur_lines) == 1 and cur_lines[0] != cur_dieu):
                    flush()
                if not cur_dieu:
                    cur_dieu = _dieu_from_khoan(kn)
                cur_khoan = kn
                cur_diem = None
                cur_lines = ([cur_dieu] if cur_dieu else []) + [s]
                cur_page  = page

            elif RE_DIEM.match(s) and cur_khoan:
                cur_diem = s[0]
                cur_lines.append(s)

            else:
                cur_lines.append(s)
                if not cur_page and page:
                    cur_page = page

            # TỐI ƯU 2: Force flush khi chunk quá lớn (Smart Text Overlap)
            if len("\n".join(cur_lines)) > self.max_size:
                # Tạm rút dòng hiện tại 's' ra để chuẩn bị flush chunk trước đó
                if cur_lines[-1] == s:
                    cur_lines.pop()
                
                # Lưu lại nội dung cũ để tính toán Overlap
                prev_lines = cur_lines.copy()
                flush()
                
                # Khởi tạo chunk mới bằng Tiêu đề Điều/Khoản
                header = []
                if cur_dieu:  header.append(cur_dieu)
                if cur_khoan: header.append(cur_khoan)
                
                # Tính toán Tail Overlap dựa trên số lượng ký tự thực tế (self.overlap)
                tail = []
                current_ovl_len = 0
                for old_line in reversed(prev_lines):
                    if old_line in header: continue # Không lấy lặp tiêu đề
                    if current_ovl_len + len(old_line) > self.overlap and tail:
                        break # Dừng lại nếu đã đủ số lượng ký tự overlap quy định
                    tail.insert(0, old_line)
                    current_ovl_len += len(old_line)
                
                # Chunk mới = Tiêu đề + Khúc đuôi chunk trước + Dòng hiện tại
                cur_lines = header + tail + [s]

        flush()
        return chunks

    def _sliding(
        self, text: str, doc_id: str,
        source: DocSource, session_id: str,
    ) -> List[Chunk]:
        chunks = []
        step   = self.max_size - self.overlap
        for i, start in enumerate(range(0, len(text), step)):
            snippet = text[start: start + self.max_size].strip()
            if len(snippet) < self.min_size:
                continue
            idieu, ikhoan, hp = _infer_structure_from_lines(snippet.splitlines())
            chunks.append(Chunk(
                text=snippet,
                metadata=ChunkMeta(
                    doc_id=doc_id, source=source, session_id=session_id,
                    dieu=idieu, khoan=ikhoan,
                    chunk_index=i,
                    heading_path=hp or f"Doan {i + 1}",
                ),
            ))
        return chunks
