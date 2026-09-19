# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Cường Quốc
**MSSV:** 2A202602469
**Nhóm:** G43
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**

Hai vector embedding chỉ về gần cùng một hướng trong không gian nhiều chiều, tức hai đoạn text nói về **cùng một ý** — bất kể chúng dùng từ ngữ khác nhau hay độ dài khác nhau. Cosine chỉ đo góc, không đo độ lớn, nên "câu ngắn" và "đoạn dài" cùng chủ đề vẫn có thể đạt điểm cao.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên được mượn tối đa bao nhiêu cuốn sách?"
- Câu B: "Hạn mức mượn tài liệu của người học là bao nhiêu quyển?"
- Tại sao tương đồng: hai câu **không chung một từ khóa nào** ở vị trí quan trọng ("sinh viên" ↔ "người học", "cuốn sách" ↔ "quyển", "tối đa" ↔ "hạn mức") nhưng hỏi đúng một chuyện. Đo thực tế bằng `paraphrase-multilingual-MiniLM-L12-v2`: **0.9134**. Đây là bằng chứng embedding nắm nghĩa chứ không so khớp chuỗi.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên bị phạt 30.000 đồng khi làm mất tickê tủ gửi đồ."
- Câu B: "Phí nội trú một học kỳ là 600.000 đồng."
- Tại sao khác: cùng là quy định dành cho sinh viên và cùng nói về tiền, nhưng thuộc hai dịch vụ hoàn toàn khác (thư viện / ký túc xá). Đo thực tế: **0.3109** — thấp nhất trong 5 cặp tôi thử, dù vẫn cao hơn tôi dự đoán vì cả hai câu cùng khuôn "sinh viên + số tiền".

**Tại sao cosine được ưu tiên hơn khoảng cách Euclid cho text embeddings?**

Khoảng cách Euclid bị chi phối bởi **độ dài vector**, mà độ dài lại tương quan với độ dài văn bản. Dùng Euclid thì một đoạn 700 ký tự và một câu hỏi 12 từ sẽ "xa" nhau dù cùng nội dung. Cosine chuẩn hoá độ lớn đi, chỉ giữ lại hướng — tức giữ lại ý nghĩa. Thêm nữa, với vector đã chuẩn hoá (`||v|| = 1`) thì cosine bằng đúng tích vô hướng, nên tính nhanh hơn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10.000 ký tự, `chunk_size=500`, `overlap=50`. Bao nhiêu chunks?**

- Bước nhảy (step) mỗi lần = `chunk_size − overlap` = 500 − 50 = **450**
- Công thức: `ceil((10000 − 50) / 450)` = `ceil(9950 / 450)` = `ceil(22.11)` = **23 chunks**
- Kiểm lại bằng chính code trong repo thay vì tin công thức suông:

```bash
python -c "from src.chunking import FixedSizeChunker; print(len(FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)))"
# 23
```

Công thức và code khớp nhau: **23**.

**Nếu overlap tăng lên 100 thì sao?**

Step giảm còn 400, nên `ceil((10000 − 100) / 400)` = `ceil(24.75)` = **25 chunks** (đã kiểm lại bằng `FixedSizeChunker`, ra đúng 25). Tăng overlap → mỗi chunk nhích ít hơn → cần nhiều chunk hơn để phủ hết tài liệu, kéo theo nhiều embedding hơn và chi phí lưu trữ cao hơn.

Lý do vẫn muốn overlap lớn: overlap chống **cắt ngang câu trả lời**. Một mốc thời hạn hoặc một con số nằm đúng ranh giới cắt sẽ bị chia đôi, khiến không chunk nào chứa trọn đáp án. Với overlap, mỗi thông tin xuất hiện trong **nhiều hơn một chunk**, tức có nhiều hơn một cơ hội lọt top-k. Phần 5 cho thấy đây không phải lý thuyết suông — nhưng cũng không miễn phí: khi tôi thêm overlap vào chiến lược của mình, Q2 được cứu còn Q1 lại tụt khỏi top-1, vì phần văn bản kéo theo làm loãng chunk vốn đang khớp nhất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:

Tách câu bằng regex **lookbehind**: `re.compile(r"(?<=[.!?])\s+")`. Điểm mấu chốt là nếu split thẳng bằng `[.!?]\s+` thì dấu câu bị nuốt mất và mọi chunk thành câu cụt; lookbehind cho phép cắt ở vị trí *sau* dấu câu mà vẫn giữ nó lại. `\s+` phủ luôn cả `". "` lẫn `".\n"`. Sau khi tách thì strip từng câu, bỏ câu rỗng, rồi gom theo bước nhảy `max_sentences_per_chunk` bằng slicing.

Edge case đã xử lý: text rỗng hoặc toàn khoảng trắng trả `[]`, và `max(1, ...)` trong `__init__` chặn tham số 0 hoặc âm.

**Edge case tôi biết là mình chưa xử lý được:** chữ viết tắt và số thập phân. `"TS. Nguyễn Văn A"`, `"v.v."`, `"1.5 tín chỉ"` đều bị cắt sai thành hai câu. Với corpus quy định học vụ tiếng Việt thì đây là rủi ro thật vì văn bản đầy chức danh và số hiệu. Sửa đúng cần một danh sách viết tắt hoặc thư viện tách câu chuyên dụng, nằm ngoài phạm vi lab.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:

Thuật toán chạy **hai chiều**, và đây là chỗ tôi phải viết lại lần hai vì lần đầu chỉ làm một chiều:

- *Đệ quy xuống sâu:* mảnh nào vẫn dài hơn `chunk_size` thì gọi lại `_split(piece, rest)` với danh sách separator còn lại — cắt bằng ranh giới "to" (`\n\n`) trước để giữ ngữ nghĩa, chỉ hạ xuống ranh giới nhỏ hơn khi bắt buộc.
- *Gom lên:* dùng một biến `buffer`, nối các mảnh nhỏ liền kề lại cho tới sát `chunk_size` rồi mới đẩy vào kết quả. Thiếu bước này, `LONG_TEXT` trong test (1.000 ký tự toàn từ "word ") sẽ sinh ra ~200 chunk vụn 4 ký tự thay vì ~10 chunk gần 100 ký tự.

**Ba base case:**
1. `current_text` rỗng → `[]`
2. `len(current_text) <= chunk_size` → `[current_text]` (mảnh đã vừa, giữ nguyên)
3. Hết separator **hoặc** gặp separator rỗng `""` → cắt cứng theo ký tự. Nhánh này là thứ làm `test_empty_separators_falls_back_gracefully` pass: test truyền thẳng `separators=[]` nên nếu không có nhánh 3 thì hàm sẽ đâm vào `remaining_separators[0]` và nổ `IndexError`.

Thêm một nhánh phụ: nếu `text.split(separator)` chỉ ra đúng 1 mảnh (separator không xuất hiện) thì bỏ qua separator đó và đệ quy ngay với `rest`, tránh lặp vô hạn.

**`compute_similarity`** — hướng tiếp cận:

Tái sử dụng `_dot` có sẵn để tính cả tích vô hướng lẫn hai chuẩn (`||a|| = sqrt(dot(a, a))`). Chốt chặn chia cho 0: kiểm `norm_a == 0.0 or norm_b == 0.0` **trước khi chia**, trả `0.0` — về mặt toán học vector 0 không có hướng nên không có góc để đo, trả 0 là quy ước hợp lý và đúng với `test_zero_vector_returns_0`.

**`ChunkingStrategyComparator.compare`** — hướng tiếp cận:

Dựng một dict `{tên: instance chunker}` rồi lặp, nhờ vậy không lặp code ba lần. Ba key phải đúng chính tả `fixed_size` / `by_sentences` / `recursive` vì test đọc theo tên chính xác. `avg_length` có chốt chặn chia 0 bằng biểu thức điều kiện `(total / len(chunks)) if chunks else 0.0` cho trường hợp text rỗng.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:

Tôi viết **hai helper trước, bốn method công khai sau** — làm ngược lại thì cùng một logic similarity sẽ bị chép ra bốn chỗ.

- `_make_record` chuẩn hoá một `Document` thành record. Hai chi tiết đáng nghĩ: (a) `dict(doc.metadata or {})` — **copy** metadata chứ không dùng thẳng object của người gọi, để store không sửa trộm dữ liệu bên ngoài; (b) `metadata.setdefault("doc_id", doc.id)` — bảo đảm record luôn có `doc_id`, vì `delete_document` phụ thuộc hoàn toàn vào khoá này. Ở `bench.py` tôi tạo nhiều `Document` từ một file với id kiểu `"file#0"`, `"file#1"` nên `doc_id` được set trỏ về **file gốc**, còn `Document.id` mới là id của chunk.
- `_search_records` chạy similarity trên **một tập record bất kỳ** truyền vào, không đụng tới `self._store`. Đây là quyết định thiết kế quan trọng nhất trong file: `search()` và `search_with_filter()` chỉ khác nhau ở *tập ứng viên*, nên cho cả hai đi qua cùng một đường code thì không thể lệch kết quả, và `test_no_filter_returns_all_candidates` pass hiển nhiên thay vì pass may rủi.

`add_documents` giữ đúng hợp đồng **một `Document` vào, một record ra** — store không tự chunk, việc chunking do tầng ngoài (`bench.py`) làm. `search` chỉ là `self._search_records(query, self._store, top_k)`.

Hai điểm khác với bản gợi ý trong đề:

1. **Tôi bỏ hẳn nhánh ChromaDB.** Code khởi tạo sẵn có một cái bẫy: `self._use_chroma = True` được gán *trước* khi client thực sự được tạo. Không test nào cần Chroma và `requirements.txt` không cài nó, nên nếu máy chấm bài tình cờ có `chromadb` thì mọi method sẽ rẽ vào nhánh chưa cài đặt và cả 14 test sập. Dùng in-memory thuần là lựa chọn an toàn và tôi ghi rõ lý do trong comment.
2. **Tôi xếp hạng bằng `compute_similarity` thay vì `_dot`.** Docstring cho phép dùng dot product vì `MockEmbedder` đã chuẩn hoá vector (`||v|| = 1`), khi đó dot == cosine. Nhưng embedding thật không phải lúc nào cũng chuẩn hoá, mà benchmark của tôi chạy trên backend thật — nên tôi dùng cosine đầy đủ để kết quả đúng với mọi backend. Chi phí thêm không đáng kể ở quy mô lab.

Kết quả sắp xếp dùng khoá `(-score, index_chèn)` nên khi điểm bằng nhau thứ tự vẫn ổn định qua các lần chạy, và record trả về **bỏ trường `embedding`** đi — vector 384 chiều in ra terminal làm bẩn output không đọc nổi.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:

**Lọc TRƯỚC rồi mới search.** Nếu lấy top-k rồi mới bỏ cái không khớp thì có thể còn lại 0 kết quả dù store vẫn đầy tài liệu hợp lệ — k slot đã bị chiếm hết bởi tài liệu sai đối tượng. Benchmark của tôi chứng minh đúng rủi ro này: ở Q1 khi **không** lọc, hai slot đầu top-3 bị tài liệu `audience=faculty` và `audience=all` chiếm mất (xem mục 5). Lọc sau sẽ để lại đúng 1 kết quả thay vì 3.

`metadata_filter` rỗng hoặc `None` thì dùng thẳng `self._store` làm tập ứng viên, nên đường đi của `search()` và `search_with_filter(filter=None)` hội tụ về cùng một chỗ. Điều kiện khớp là `all(record["metadata"].get(k) == v for k, v in metadata_filter.items())` — dùng `.get()` nên tài liệu thiếu khoá đó sẽ bị loại chứ không nổ `KeyError`.

`delete_document` dựng lại list chỉ gồm record có `metadata['doc_id'] != doc_id`, so sánh độ dài trước/sau để biết có xoá được gì không, rồi trả `removed > 0`. Cách này xoá **mọi chunk** của một tài liệu trong một lượt, đúng yêu cầu khi một file đã bị chia thành hàng chục chunk.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:

Ba nhịp: truy xuất top-k → dựng prompt có ngữ cảnh → gọi `llm_fn`.

Phần tôi đầu tư nhiều nhất là **cách dựng ngữ cảnh**. Mỗi chunk được đánh số `[1] [2] [3]` kèm nguồn lấy từ metadata theo thứ tự ưu tiên `source_url` → `source` → `doc_id` → `id`, và prompt yêu cầu model trích dẫn đúng số đó. Nhờ vậy câu trả lời **truy vết được** về đúng chunk và đúng URL gốc — đây là tiêu chí *Source Traceability* trong `docs/EVALUATION.md`, và với corpus quy định học vụ thì nó không phải tính năng phụ: người đọc phải kiểm được câu trả lời dựa trên văn bản nào.

Hai ràng buộc chống bịa nằm ngay trong prompt: "không dùng kiến thức ngoài ngữ cảnh, không suy đoán quy định", và một câu thoát cố định khi ngữ cảnh không đủ thông tin.

Trường hợp store rỗng được chặn **trước** khi gọi LLM: trả thẳng `EMPTY_CONTEXT_ANSWER` thay vì crash hoặc đốt một lượt gọi model vô ích.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```
$ pytest tests/ -v
platform win32 -- Python 3.14.0, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\VinUni\Day 7\K4-Day07-PhamCuongQuoc-2A202602469
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.13s ==============================
```

**Số lượng bài test vượt qua (pass):** **42 / 42**

Không còn dòng `raise NotImplementedError` nào trong `src/`.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Dự đoán được ghi **trước khi chạy** `compute_similarity()`. Backend: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều).

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|-----|-------|-------|---------|--------------|-------|
| P1 | "Sinh viên được mượn tối đa bao nhiêu cuốn sách?" | "Hạn mức mượn tài liệu của người học là bao nhiêu quyển?" | cao — 0.90 | **0.9134** | ✅ lệch +0.013 |
| P2 | "Khu nội trú đóng cửa lúc 23:00." | "Giờ đóng cửa của ký túc xá là 11 giờ đêm." | cao — 0.85 | **0.6534** | ❌ lệch −0.197 |
| P3 | "Điều kiện xét học bổng khuyến khích **học tập**." | "Điều kiện xét học bổng khuyến khích **nghiên cứu khoa học**." | cao — 0.88 | **0.7822** | ⚠️ lệch −0.098 |
| P4 | "Thời hạn nộp phiếu hủy học phần." | "Thời hạn trả sách cho thư viện." | trung bình — 0.50 | **0.4976** | ✅ lệch −0.002 |
| P5 | "Sinh viên bị phạt 30.000 đồng khi làm mất tickê tủ gửi đồ." | "Phí nội trú một học kỳ là 600.000 đồng." | thấp — 0.15 | **0.3109** | ⚠️ lệch +0.161 |

**Xếp hạng dự đoán:** P1 > P3 > P2 > P4 > P5
**Xếp hạng thực tế:** P1 > P3 > P2 > P4 > P5 — **khớp hoàn toàn về thứ tự**, nhưng lệch khá nhiều ở giá trị tuyệt đối của P2 và P5.

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

Bất ngờ nhất là **P2 chỉ đạt 0.65 dù là paraphrase thật**. Tôi đã đánh giá cao khả năng model quy đổi `"23:00"` ↔ `"11 giờ đêm"`. Model xử lý rất tốt từ đồng nghĩa (P1: 0.91, "sinh viên" ↔ "người học", "cuốn sách" ↔ "quyển") nhưng **yếu hẳn với biến thể định dạng số**, vì hai chuỗi đó tách thành những token gần như không liên quan. Đây là cảnh báo trực tiếp cho corpus quy định học vụ — vốn dày đặc mốc giờ, số ngày, số tiền: không nên trông cậy hoàn toàn vào embedding cho câu hỏi tra số liệu. Và Q2 ở mục 5 hỏng đúng vì cơ chế này.

Bất ngờ thứ hai là **P3 vẫn đạt 0.78**. Hai câu chỉ khác nhau một cụm từ nhưng mô tả **hai loại học bổng khác nhau với hai bộ điều kiện khác nhau** — về mặt tra cứu quy định thì trả nhầm là trả sai. Embedding lại thấy chúng gần như đồng nghĩa vì phần lớn token trùng nhau và "học tập" với "nghiên cứu khoa học" cùng nằm trong vùng ngữ nghĩa học thuật.

Gộp hai điều trên: **cosine đo độ giống chủ đề, không đo độ giống câu trả lời.** Đó cũng là lý do metadata filter ở mục 5 có giá trị thật — nó phân biệt được thứ mà embedding không phân biệt nổi.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

**Chiến lược của tôi:** `HeadingChunker` — chia theo tiêu đề/mục của văn bản quy định (`chunk_size=700`), mục dài quá ngưỡng thì hạ xuống `RecursiveChunker` và **gắn lại tiêu đề vào từng mảnh con**. Code đầy đủ trong `bench.py`.

**Cấu hình chạy:** 9 tài liệu (dịch vụ/quy định của các trường đại học tại Hà Nội) → **79 chunk** (min 12 / trung bình 400 / max 751 ký tự), embedder `paraphrase-multilingual-MiniLM-L12-v2` (384 chiều), `top_k=3`. Output đầy đủ: `ket_qua_benchmark.txt`.

| # | Câu hỏi | Top-1 chunk truy xuất được | Score | Có liên quan? | Câu trả lời của Agent |
|---|---------|---------------------------|-------|---------------|----------------------|
| 1 | Tôi được mượn tối đa bao nhiêu giáo trình và bao nhiêu tài liệu tham khảo cùng một lúc? | `utc-thu-vien-sinh-vien#2` — "## 2. Hạn mức mượn tài liệu về nhà của người học" | +0.6542 | ✅ | Trích đúng: không quá 10 giáo trình, 02 tài liệu tham khảo |
| 2 | Sinh viên Học viện Ngoại giao được mượn tối đa bao nhiêu tài liệu in và trong bao nhiêu ngày? | `dav-muon-tra-giang-vien#0` — **tài liệu của giảng viên** | +0.7228 | ❌ Sai đối tượng ở top-1, chunk chứa đáp án ở hạng 13 | Grounding sai: không chunk nào trong top-3 chứa bảng hạn mức của sinh viên |
| 3 | Khu nội trú của Trường ĐH Thương mại mở cửa và đóng cửa lúc mấy giờ? | `tmu-noi-quy-khu-noi-tru#1` — "## 1. Thời gian mở cửa, đóng cửa" | +0.6303 | ✅ | Trích đúng: mở cửa từ 5 giờ, đóng cửa từ 23 giờ |
| 4 | Phí nội trú ký túc xá một học kỳ đối với sinh viên hệ chính quy là bao nhiêu? | `utc-huong-dan-ky-tuc-xa#5` — "## 3. Mức phí đóng, nộp khi sinh viên vào KTX" | +0.7609 | ✅ | Trích đúng: 600.000đ/học kỳ chính quy, 400.000đ/học kỳ cử tuyển |
| 5 | Những đối tượng sinh viên nào được miễn 100% học phí? | `utc-che-do-chinh-sach#7` — "### 1.1. Đối tượng được miễn 100% học phí" | +0.7721 | ⚠️ Đúng mục nhưng **chỉ phủ 2/4** nhóm đối tượng | Trả lời thiếu: nêu được nhóm dân tộc thiểu số, thiếu nhóm người có công và nhóm khuyết tật |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **4 / 5**

**Điểm retrieval theo `docs/SCORING.md`: 7 / 10** (Q1, Q3, Q4 mỗi câu 2 điểm; Q5 được 1 điểm vì trả lời thiếu; Q2 được 0 điểm).

### Chấm hai mức, và cách chấm câu hỏi dạng liệt kê

`bench.py` của tôi chấm **hai mức** chứ không chỉ một:

1. *Mức doc_id:* `doc_id` của tài liệu gold có nằm trong top-3 không → **5/5 câu đạt**.
2. *Mức nội dung:* ngữ cảnh truy xuất được có thật sự chứa câu trả lời không → **3/5 câu đạt đủ, 1 câu đạt một phần, 1 câu trượt**.

Chênh lệch 5/5 ↔ 3.5/5 chính là lý do không được chấm theo `doc_id`. Nếu chỉ kiểm `doc_id` thì tôi đã báo cáo 10/10 và không hề biết mình có hai lỗi thật.

Với **câu hỏi dạng liệt kê** (Q5: "những đối tượng nào được miễn 100% học phí"), kiểm một chuỗi duy nhất là quá thô — đáp án gồm 4 nhóm đối tượng, lấy được 1 nhóm không thể coi là trả lời đúng, mà lấy được 3 nhóm cũng không thể coi là trượt. Nên `must_contain` của tôi nhận **danh sách chuỗi** và chấm theo **độ phủ**: phủ đủ 4/4 và gold ở top-1 thì 2 điểm; phủ từ 50% trở lên thì 1 điểm; dưới 50% thì 0 điểm.

### Phân tích lỗi 1 — Q2: chunk là bảng số thua chunk là văn xuôi (Bài tập 3.5)

**Hiện tượng:** top-1 là `dav-muon-tra-giang-vien#0` (+0.7228) — tài liệu dành cho **giảng viên**, trong khi câu hỏi nói rõ "Sinh viên". Tài liệu gold `dav-muon-tra-sinh-vien` có mặt ở top-2 nhưng là chunk `#0`, tức phần **tiêu đề và ghi chú phạm vi**, không chứa số liệu. Chunk thật sự chứa đáp án là `#1` ("## 1. Hạn mức tài liệu in") xếp **hạng 13** với score +0.5086.

**Tại sao:**

1. **Cosine đo độ giống chủ đề, không đo mật độ thông tin trả lời được.** Chunk `#1` phần lớn là **bảng markdown** — embedding của một bảng số loãng hơn hẳn embedding của văn xuôi. Hai chunk `#0` của hai tài liệu lại là văn xuôi đầy từ trùng với câu hỏi ("mượn trả tài liệu", "sinh viên", "giảng viên", "Học viện Ngoại giao"), nên thắng dễ dàng.
2. **Từ "Sinh viên" trong câu hỏi không đủ để phân biệt**, vì tài liệu của giảng viên cũng chứa từ "sinh viên" trong câu ghi chú trỏ sang tài liệu kia. Đây chính là hiện tượng đã đo được ở cặp P3 mục 4: một cụm từ khác biệt không kéo được cosine xuống.
3. **Chunker theo heading không có overlap:** mỗi dữ kiện chỉ nằm trong đúng một chunk, tức chỉ có một cơ hội lọt top-k.

**Đề xuất cải thiện — đã kiểm chứng bằng số liệu, không phải phỏng đoán:**

| Cách sửa | Q2 | Tổng điểm |
|---|---|---|
| Giữ nguyên, nâng `top_k` 3 → 5 | vẫn trượt | 7/10 |
| Giữ nguyên, nâng `top_k` → 10 | vẫn trượt | 7/10 |
| Giữ nguyên, nâng `top_k` → 15 | ✅ cứu được | 8/10 |
| Thêm `overlap=150` vào `HeadingChunker` | ✅ cứu được (1/2 điểm) | 7/10 |

Kết luận thẳng thắn: **nâng `top_k` không phải cách sửa tốt ở đây** — phải lên tới 15 mới bắt được chunk đúng, tức nhồi 15 đoạn vào ngữ cảnh LLM để lấy 1 đoạn hữu ích. Cách sửa đúng gốc là thêm overlap, và tôi đã cài thật (`HeadingChunker(overlap=150)`, xem `bench.py`).

Nhưng overlap **không miễn phí**: nó cứu Q2 và nâng độ phủ Q5 từ 2/4 lên 3/4, đồng thời lại **đánh mất top-1 ở Q1** (tụt xuống top-2), vì phần văn bản kéo theo làm loãng chunk vốn đang khớp nhất. Tổng điểm vẫn là 7/10 — chỉ đổi chỗ lỗi chứ không xoá được lỗi. Đây là phát hiện tôi thấy giá trị nhất: **overlap là đánh đổi precision ↔ recall, không phải cải tiến một chiều.**

Cách sửa triệt để hơn mà tôi chưa kịp làm: thêm một câu văn xuôi tóm tắt ngay dưới mỗi bảng hạn mức ("Sinh viên các khóa được mượn tối đa 2 tài liệu in trong 2 ngày") để chunk chứa bảng có phần văn xuôi cạnh tranh được. Tôi đã làm việc này cho tài liệu DAV nhưng câu tóm tắt lại rơi vào chunk `#1`, còn chunk `#0` mới là chunk thắng — tức vị trí đặt câu tóm tắt cũng quan trọng.

### Phân tích lỗi 2 — Q5: danh sách dài bị cắt qua nhiều chunk

**Hiện tượng:** cả 3 slot top-3 đều thuộc đúng tài liệu gold, top-1 và top-2 đều thuộc đúng mục `### 1.1. Đối tượng được miễn 100% học phí`. Vậy mà ngữ cảnh chỉ phủ **2/4** nhóm đối tượng.

**Tại sao:** mục 1.1 dài hơn `chunk_size` nên bị `RecursiveChunker` cắt thành 4 mảnh (`#4`–`#7`). Tiêu đề được gắn lại vào từng mảnh (đúng thiết kế), nhưng mảnh `#5` — chứa nhóm "người có công với cách mạng" và nhóm "sinh viên tàn tật, khuyết tật" — bị chi phối bởi đoạn liệt kê rất dài về "con của người hoạt động cách mạng trước ngày 01/01/1945…", vốn cách xa câu hỏi về mặt ngữ nghĩa. Kết quả: `#5` xếp **hạng 47** với score +0.4066, trong khi `#6` và `#7` cùng mục lại xếp hạng 1 và 2.

**Đề xuất:** với câu hỏi dạng liệt kê, top-k theo chunk là sai đơn vị. Nên **gộp lại theo mục**: nếu một chunk thuộc mục X lọt top-k thì nạp trọn mục X vào ngữ cảnh thay vì chỉ mảnh đó. Chi phí là ngữ cảnh dài hơn, nhưng với văn bản quy định thì một điều khoản trọn vẹn đáng giá hơn ba mảnh rời.

### Lọc bằng metadata có giúp ích không — bằng chứng A/B

Q1 được chạy **hai lần trên cùng một store**, chỉ khác tham số `metadata_filter`. Corpus đã tách quy định thư viện ĐH Giao thông vận tải thành hai tài liệu cùng nguồn, cùng từ vựng, khác đối tượng và **khác đáp án** (người học: 10 giáo trình / 02 tài liệu tham khảo; cán bộ, giảng viên: 07 giáo trình / 03 tài liệu tham khảo). Câu hỏi cố ý không nêu người hỏi là ai.

| Lần chạy | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| **Có** `metadata_filter={"audience": "student"}` | `utc-thu-vien-sinh-vien#2` (+0.6542) — student | `dav-muon-tra-sinh-vien#0` (+0.5905) — student | `utc-thu-vien-sinh-vien#3` (+0.5883) — student |
| **Không** filter | `utc-thu-vien-sinh-vien#2` (+0.6542) — student | `utc-thu-vien-can-bo#2` (+0.6260) — **staff** | `dav-muon-tra-giang-vien#0` (+0.5982) — **faculty** |

Không lọc thì **2/3 slot bị tài liệu sai đối tượng chiếm**, và tài liệu hạn mức của cán bộ (+0.6260) bám sát tài liệu đúng (+0.6542) — chênh nhau chỉ 0.028. Ở khoảng cách đó, chỉ cần câu hỏi diễn đạt khác đi một chút là thứ hạng đảo, và agent sẽ trả lời "07 giáo trình, 03 tài liệu tham khảo" cho một sinh viên — sai, nhưng nghe rất thuyết phục vì đó cũng là quy định thật của cùng một thư viện.

Lọc trước đẩy toàn bộ ba slot về đúng `audience=student`. Ở lần chạy này cả hai cách đều đạt 2/2 điểm vì chunk đúng vốn đã ở top-1, nhưng **biên an toàn hoàn toàn khác nhau**: có filter thì không tài liệu sai đối tượng nào còn cơ hội; không filter thì nó đứng ngay sau lưng.

Đây cũng là lý do tôi lọc **trước** khi xếp hạng trong `search_with_filter`. Nếu lọc sau, ở lần chạy không filter tôi sẽ chỉ còn lại 1 kết quả sau khi loại hai tài liệu sai đối tượng, dù store vẫn còn nhiều chunk hợp lệ.

### So sánh với các chiến lược khác (cùng corpus, cùng 5 query, cùng embedder)

| Chiến lược | Số chunk | Độ dài TB | Điểm | Mạnh ở | Yếu ở |
|---|---|---|---|---|---|
| **`heading` (của tôi)** | 79 | 400 | **7/10** | Q1 top-1; là chiến lược **duy nhất** ăn điểm ở Q5 nhờ giữ được tiêu đề tiểu mục | Q2 (chunk bảng số) |
| `heading+overlap` | 79 | 523 | 7/10 | Cứu Q2, Q5 phủ 3/4 | Mất top-1 ở Q1 |
| `fixed` (overlap 120) | 55 | 662 | 7/10 | Q2 top-1 nhờ overlap | Q1 chỉ top-3; Q5 phủ 0/4 |
| `sentence` | 70 | 440 | 7/10 | Ổn định ở Q2–Q4 | Q5 phủ 0/4 |
| `recursive` | 69 | 446 | 5/10 | — | Trượt cả Q2 lẫn Q5 |

Bốn chiến lược hoà nhau ở 7/10 nhưng **hỏng ở những câu khác nhau** — đây là điều tôi không đoán trước. Ưu thế riêng của chiến lược theo heading nằm ở Q5: nó là chiến lược duy nhất giữ được tiêu đề `### 1.1. Đối tượng được miễn 100% học phí` dính vào nội dung, nên còn ăn được 1 điểm; ba chiến lược còn lại cắt tiêu đề rời khỏi danh sách và phủ 0/4.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

> *[ĐIỀN SAU BUỔI DEMO]*

---


## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 7 / 10 |
| **Tổng phần cá nhân** | **57 / 60** |

*Trừ 3 điểm ở mục Kết quả truy xuất: Q2 không lấy được chunk chứa đáp án trong top-3 (0/2), Q5 chỉ phủ được 2/4 ý của đáp án (1/2). Tôi giữ nguyên con số thật thay vì nới lỏng cách chấm hoặc đổi sang cấu hình có lợi hơn — cả hai lỗi đều đã được truy đến nguyên nhân và kiểm chứng cách sửa bằng số liệu ở trên, và theo `docs/SCORING.md` thì phần giải thích được đánh giá cao hơn phần điểm số.*
