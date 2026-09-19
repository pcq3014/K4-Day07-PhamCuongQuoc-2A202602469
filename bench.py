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
STRATEGY = "heading"  # heading | heading+overlap | fixed | recursive | sentence
# =======================================================================

CORPUS_DIR = Path("data/dich-vu-dai-hoc")  # corpus dịch vụ/quy định các trường ĐH tại Hà Nội
CHUNK_SIZE = 700


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
        "question": "Tôi được mượn tối đa bao nhiêu giáo trình và bao nhiêu tài liệu tham khảo cùng một lúc?",
        "gold_doc": "utc-thu-vien-sinh-vien",
        "gold_answer": (
            "Bạn đọc có thẻ đa năng (người học) được mượn không quá 10 giáo trình và "
            "02 tài liệu tham khảo tại một thời điểm."
        ),
        "must_contain": "không được quá 10 giáo trình, 02 tài liệu tham khảo",
        "metadata_filter": {"audience": "student"},
        # Câu hỏi cố ý KHÔNG nêu người hỏi là ai. Cùng một trang quy định của
        # ĐH GTVT đã tách thành hai tài liệu theo audience, cùng từ vựng nhưng
        # khác đáp án (cán bộ/giảng viên: 07 giáo trình, 03 tài liệu tham khảo).
        "ab_test": True,  # chạy hai lần: có filter và không filter
    },
    {
        "id": "Q2",
        "question": "Sinh viên Học viện Ngoại giao được mượn tối đa bao nhiêu tài liệu in và trong bao nhiêu ngày?",
        "gold_doc": "dav-muon-tra-sinh-vien",
        "gold_answer": "Sinh viên các khóa được mượn tối đa 2 tài liệu in trong 2 ngày, gia hạn 2 lần, mỗi lần 1 ngày.",
        "must_contain": "2 tài liệu in trong 2 ngày",
        "metadata_filter": None,
    },
    {
        "id": "Q3",
        "question": "Khu nội trú của Trường Đại học Thương mại mở cửa và đóng cửa lúc mấy giờ?",
        "gold_doc": "tmu-noi-quy-khu-noi-tru",
        "gold_answer": "Mở cửa từ 5 giờ, đóng cửa từ 23 giờ.",
        "must_contain": "Đóng cửa: từ 23 giờ",
        "metadata_filter": None,
    },
    {
        "id": "Q4",
        "question": "Phí nội trú ký túc xá một học kỳ đối với sinh viên hệ chính quy là bao nhiêu tiền?",
        "gold_doc": "utc-huong-dan-ky-tuc-xa",
        "gold_answer": "600.000đ/học kỳ với sinh viên hệ chính quy và 400.000đ/học kỳ với sinh viên hệ cử tuyển.",
        "must_contain": "600.000đ/học kỳ đối với sinh viên hệ chính quy",
        "metadata_filter": None,
    },
    {
        "id": "Q5",
        "question": "Những đối tượng sinh viên nào được miễn 100% học phí?",
        "gold_doc": "utc-che-do-chinh-sach",
        "gold_answer": (
            "Người có công với cách mạng và thân nhân; sinh viên tàn tật, khuyết tật thuộc hộ nghèo "
            "hoặc cận nghèo; sinh viên dân tộc thiểu số thuộc hộ nghèo, cận nghèo; sinh viên dân tộc "
            "thiểu số rất ít người ở vùng khó khăn."
        ),
        # Câu hỏi dạng liệt kê: chấm theo độ phủ 4 nhóm đối tượng, không chỉ một chuỗi.
        "must_contain": [
            "Người có công với cách mạng và thân nhân của người có công với cách mạng",
            "Sinh viên bị tàn tật, khuyết tật thuộc diện hộ nghèo hoặc hộ cận nghèo",
            "Sinh viên là người dân tộc thiểu số thuộc hộ nghèo và hộ cận nghèo",
            "dân tộc thiểu số rất ít người",
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
