"""Benchmark 5 truy vấn trên corpus dịch vụ/quy định đại học.

Chạy:  python bench.py                 # chiến lược mặc định (heading)
       python bench.py --strategy fixed
       python bench.py > ket_qua_benchmark.txt

Mỗi thành viên chỉ đổi ĐÚNG MỘT dòng — `STRATEGY` bên dưới — sang chiến lược
của mình. Mọi thứ khác giữ nguyên để việc so sánh trong nhóm là công bằng.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src import (
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    LocalEmbedder,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)

# === DÒNG DUY NHẤT MỖI THÀNH VIÊN ĐỔI ==================================
STRATEGY = "recursive"  # heading | heading+overlap | fixed | recursive | sentence
# =======================================================================

CORPUS_DIR = Path("data/khao-thi-phuc-khao")  # corpus khảo thí & phúc khảo (Quy chế 610/QĐ-ĐHCN)
CHUNK_SIZE = 500


class HeadingChunker:
    """Chiến lược tùy chỉnh: chia theo tiêu đề mục của văn bản quy định.

    Lý do thiết kế: quy định học vụ đã được người soạn chia sẵn theo mục
    ("## 5. Quy định mượn/trả tài liệu"), nên mỗi mục vốn là một đơn vị ngữ
    nghĩa trọn vẹn. Cắt theo ranh giới đó giữ được trọn điều khoản thay vì
    cắt ngang một bảng hạn mức hay một câu điều kiện.

    Mục nào dài hơn chunk_size thì hạ xuống RecursiveChunker, và tiêu đề được
    GẮN LẠI vào từng mảnh con — nếu không, từ mảnh thứ hai trở đi cả người đọc
    lẫn embedding đều mất ngữ cảnh "mục này đang nói về cái gì".

    Tham số `overlap` là kết quả của phân tích lỗi: chia theo mục thì mỗi dữ
    kiện chỉ nằm trong ĐÚNG MỘT chunk, tức chỉ có một cơ hội lọt top-k. Cho mỗi
    chunk kéo theo `overlap` ký tự cuối của chunk liền trước thì một con số nằm
    sát ranh giới mục có hai cơ hội được truy xuất.
    """

    HEADING = re.compile(r"^#{1,6}\s+.*$", re.M)

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = 0) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._fallback = RecursiveChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        starts = [match.start() for match in self.HEADING.finditer(text)]
        if not starts:
            return self._add_overlap(self._fallback.chunk(text.strip()))
        if starts[0] > 0:
            starts.insert(0, 0)
        bounds = starts + [len(text)]

        chunks: list[str] = []
        for begin, end in zip(bounds, bounds[1:]):
            section = text[begin:end].strip()
            if not section:
                continue
            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue

            heading = section.splitlines()[0].strip()
            body = section[len(heading) :].strip()
            for piece in self._fallback.chunk(body):
                chunks.append(f"{heading}\n{piece}")
        return self._add_overlap(chunks)

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        if self.overlap <= 0 or len(chunks) < 2:
            return chunks
        overlapped = [chunks[0]]
        for previous, current in zip(chunks, chunks[1:]):
            overlapped.append(f"{previous[-self.overlap :].strip()}\n{current}")
        return overlapped


CHUNKERS = {
    "heading": HeadingChunker(chunk_size=CHUNK_SIZE),
    "heading+overlap": HeadingChunker(chunk_size=CHUNK_SIZE, overlap=150),
    "fixed": FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=120),
    "recursive": RecursiveChunker(chunk_size=CHUNK_SIZE),
    "sentence": SentenceChunker(max_sentences_per_chunk=4),
}

# 5 benchmark query của nhóm. `must_contain` là chuỗi đặc trưng PHẢI xuất hiện
# trong ngữ cảnh truy xuất được — dùng để chấm ở mức nội dung, không chỉ mức
# doc_id (chấm theo doc_id thổi phồng kết quả, xem docs/SCORING.md).
QUERIES = [
    {
        "id": "Q1",
        "question": "Tôi muốn phúc khảo bài thi tự luận thì phải làm gì và trong thời hạn bao lâu?",
        "gold_doc": "phuc-khao-nguoi-hoc",
        "gold_answer": (
            "Làm đơn phúc khảo điểm thi (Mẫu 12), chuyển đơn cùng phiếu đóng tiền phúc khảo đến "
            "giáo vụ Khoa/Viện của đơn vị chủ quản học phần trong vòng 14 ngày làm việc kể từ ngày "
            "điểm thi được công bố. Nộp muộn hơn phải được Trưởng đơn vị chủ quản học phần đồng ý."
        ),
        "must_contain": "trong vòng 14 ngày làm việc, kể từ ngày điểm thi được công bố",
        "metadata_filter": {"audience": "student"},
        # Điều 26 đã được tách thành hai tài liệu theo audience: phần người học
        # (nộp đơn, 14 ngày làm việc) và phần giảng viên (chấm phúc khảo, 05 ngày
        # làm việc). Cùng từ vựng, khác đáp án — không lọc thì dễ trả nhầm.
        "ab_test": True,
    },
    {
        "id": "Q2",
        "question": "Thời lượng tối đa của một bài thi tự luận là bao nhiêu phút?",
        "gold_doc": "hinh-thuc-thoi-luong-thi",
        "gold_answer": "Tối thiểu 50 phút và tối đa 120 phút, tùy số tín chỉ của học phần và số câu hỏi trong đề.",
        "must_contain": "tối đa là 120 phút",
        "metadata_filter": None,
    },
    {
        "id": "Q3",
        "question": "Đến phòng thi muộn bao lâu thì không được dự thi?",
        "gold_doc": "nguoi-hoc-du-thi",
        "gold_answer": (
            "Người học đến muộn quá 15 phút sau khi đã phát đề thi sẽ không được dự thi; "
            "phải có mặt tại phòng thi trước giờ thi ít nhất 15 phút."
        ),
        "must_contain": "đến muộn quá 15 phút sau khi đã phát đề thi sẽ không được dự thi",
        "metadata_filter": None,
    },
    {
        "id": "Q4",
        "question": "Hai giảng viên chấm tiểu luận lệch nhau từ 2 điểm trở lên thì xử lý thế nào?",
        "gold_doc": "cham-thi",
        "gold_answer": (
            "Hai GV thảo luận để thống nhất kết quả; nếu không thống nhất được thì báo CNBM "
            "xem xét quyết định. Lệch dưới 02 điểm thì lấy trung bình cộng."
        ),
        "must_contain": "báo CNBM xem xét quyết định",
        "metadata_filter": None,
    },
    {
        "id": "Q5",
        "question": "Những lỗi vi phạm nào khiến người học bị đình chỉ thi?",
        "gold_doc": "xu-ly-vi-pham-nguoi-hoc",
        "gold_answer": (
            "Mang tài liệu hoặc phương tiện bị cấm vào phòng thi; đưa đề thi ra ngoài khu vực thi "
            "hoặc nhận bài giải từ bên ngoài; đã bị cảnh cáo mà vẫn vi phạm; viết, vẽ nội dung "
            "không liên quan; gây rối, đe dọa, xúc phạm CBCT hoặc người học khác; không chấp hành "
            "yêu cầu của CBCT về kỷ luật phòng thi. Hậu quả: điểm 0 cho học phần."
        ),
        # Câu hỏi dạng liệt kê: chấm theo độ phủ các nhóm lỗi, không chỉ một chuỗi.
        "must_contain": [
            "mang theo tài liệu hoặc phương tiện bị cấm vào phòng thi",
            "đưa đề thi ra ngoài khu vực thi",
            "có hành vi gây rối, lời nói hoặc cử chỉ đe dọa, xúc phạm CBCT",
            "không chấp hành các yêu cầu của CBCT liên quan đến kỷ luật phòng thi",
        ],
        "metadata_filter": None,
    },
]


def normalize(text: str) -> str:
    """Bỏ nhiễu định dạng để so khớp chuỗi đặc trưng cho ổn định."""
    return re.sub(r"\s+", " ", text.replace("*", "").replace("’", "'")).lower()


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    _, front_matter, body = raw.split("---", 2)
    metadata = {
        key: value.strip().strip('"')
        for key, value in re.findall(r"^(\w+):\s*(.+)$", front_matter, re.M)
    }
    return metadata, body.strip()


def load_documents(chunker) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        for position, chunk in enumerate(chunker.chunk(body)):
            documents.append(
                Document(
                    id=f"{path.stem}#{position}",
                    content=chunk,
                    # Trải metadata frontmatter vào MỌI chunk, nếu không
                    # search_with_filter() không có gì để lọc. doc_id trỏ về
                    # file gốc, còn Document.id mới là "file#0".
                    metadata={**metadata, "doc_id": path.stem, "chunk_index": position},
                )
            )
    return documents


def build_embedder():
    load_dotenv(override=False)
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder()
        except Exception as error:  # noqa: BLE001 - lab fallback: phải chạy được cả khi offline
            print(f"[canh bao] LocalEmbedder that bai ({error}); quay ve MockEmbedder")
    return _mock_embed


def demo_llm(prompt: str) -> str:
    """LLM giả lập: tóm tắt khối ngữ cảnh để kiểm tra grounding bằng mắt."""
    context = prompt.split("=== NGỮ CẢNH ===")[1].split("=== CÂU HỎI ===")[0].strip()
    first_line = context.splitlines()[0] if context else "(rong)"
    return f"[DEMO LLM] Tra loi dua tren {context.count('[')} doan ngu canh, bat dau tu {first_line}"


def print_results(results: list[dict], gold_doc: str) -> None:
    if not results:
        print("      (khong co ket qua)")
        return
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        marker = "<== GOLD" if metadata["doc_id"] == gold_doc else ""
        preview = re.sub(r"\s+", " ", result["content"])[:86]
        print(
            f"      {rank}. score={result['score']:+.4f}  {result['id']:32}"
            f" audience={metadata['audience']:8} {marker}"
        )
        print(f"         {preview}...")


def score_query(query: dict, results: list[dict]) -> tuple[int, str]:
    """Chấm hai mức: doc_id trong top-3 VÀ ngữ cảnh chứa được câu trả lời.

    `must_contain` là một chuỗi (câu hỏi tra một dữ kiện) hoặc một danh sách
    chuỗi (câu hỏi dạng liệt kê). Với dạng liệt kê, chấm theo **độ phủ**: một
    câu hỏi "kể tên các đối tượng được miễn học phí" mà top-3 chỉ lấy được một
    nửa danh sách thì không thể coi là trả lời đủ, nhưng cũng không phải là
    trượt hoàn toàn — nên nó nhận điểm giữa.
    """
    needles = query["must_contain"]
    if isinstance(needles, str):
        needles = [needles]

    ranks = [i for i, r in enumerate(results, start=1) if r["metadata"]["doc_id"] == query["gold_doc"]]
    context = normalize(" ".join(r["content"] for r in results))
    found = [needle for needle in needles if normalize(needle) in context]
    coverage = len(found) / len(needles)

    if not ranks:
        return 0, "gold doc vang mat trong top-3"
    if coverage < 0.5:
        return 0, f"gold doc o top-{ranks[0]} nhung ngu canh phu {len(found)}/{len(needles)} y cua dap an"
    if coverage < 1.0:
        return 1, f"gold o top-{ranks[0]}, ngu canh chi phu {len(found)}/{len(needles)} y cua dap an"
    if ranks[0] == 1:
        return 2, "gold o top-1 va ngu canh chua du dap an"
    return 1, f"gold o top-{ranks[0]}, ngu canh chua du dap an"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=STRATEGY, choices=sorted(CHUNKERS))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    chunker = CHUNKERS[args.strategy]
    embedder = build_embedder()
    documents = load_documents(chunker)

    lengths = [len(doc.content) for doc in documents]
    print("=" * 78)
    print(f"BENCHMARK - chien luoc: {args.strategy}  (chunk_size={CHUNK_SIZE})")
    print(f"Embedding backend     : {getattr(embedder, '_backend_name', type(embedder).__name__)}")
    print(f"Corpus                : {len(list(CORPUS_DIR.glob('*.md')))} tai lieu -> {len(documents)} chunk")
    print(f"Do dai chunk          : min={min(lengths)} avg={sum(lengths) // len(lengths)} max={max(lengths)}")
    print("=" * 78)

    store = EmbeddingStore(collection_name="bench_dich_vu_dai_hoc", embedding_fn=embedder)
    store.add_documents(documents)
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

    total = 0
    for query in QUERIES:
        print(f"\n[{query['id']}] {query['question']}")
        print(f"   Gold answer : {query['gold_answer']}")
        print(f"   Gold doc    : {query['gold_doc']}   filter={query['metadata_filter']}")

        results = store.search_with_filter(
            query["question"], top_k=args.top_k, metadata_filter=query["metadata_filter"]
        )
        print("   Top-3:")
        print_results(results, query["gold_doc"])

        points, reason = score_query(query, results)
        total += points
        print(f"   => diem: {points}/2  ({reason})")

        if query.get("ab_test"):
            print("   --- A/B: cung cau hoi, KHONG dung metadata_filter ---")
            no_filter = store.search_with_filter(query["question"], top_k=args.top_k, metadata_filter=None)
            print_results(no_filter, query["gold_doc"])
            ab_points, ab_reason = score_query(query, no_filter)
            print(f"   => diem khong filter: {ab_points}/2  ({ab_reason})")

    print("\n" + "=" * 78)
    print(f"TONG DIEM RETRIEVAL: {total}/{2 * len(QUERIES)}  (chien luoc: {args.strategy})")
    print("=" * 78)

    print("\n=== KnowledgeBaseAgent - kiem tra grounding tren Q1 ===")
    print(agent.answer(QUERIES[0]["question"], top_k=args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
