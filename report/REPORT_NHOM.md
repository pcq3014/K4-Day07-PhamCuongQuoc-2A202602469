# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** G43
**Thành viên:** Nguyễn Trường Bảo (2A202602540), Phạm Cường Quốc (2A202602469), Đỗ Đức Đại (2A202602725), Đỗ Ngọc Phi (2A202602531)
**Ngày:** 19/09/2026

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

**Cấu hình chung khi đo:** embedder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local), `top_k = 3`, `chunk_size = 500`, cùng 10 tài liệu và cùng 5 câu hỏi. Mỗi người chỉ đổi dòng `CHUNKER` trong `bench.py`. Agent trả lời theo kiểu trích xuất (in chunk top-1) vì nhóm không dùng LLM trả phí. Toàn bộ output nằm trong `report/benchmark/`.

### Phân công vai trò

Nhóm 4 người theo lab: ba vai R1–R3 cộng thêm vai **Report & Demo**. **Trưởng nhóm: Đỗ Ngọc Phi** (kiêm R1 · Data). Vai là trách nhiệm điều phối cộng thêm; cả 4 người đều tự code `src/`, tự chạy benchmark với chiến lược riêng và không ai trùng chiến lược.

| Thành viên | MSSV | Vai | Chiến lược chunking | Việc điều phối chính |
|---|---|---|---|---|
| Đỗ Ngọc Phi | 2A202602531 | **Trưởng nhóm** · R1 · Data | Sentence (3 câu/chunk) | Điều phối nhóm. Chốt chủ đề và nguồn (IUH), kiểm `robots.txt`, crawl và làm sạch trang HTML, chép và tách 9 Điều từ PDF theo `audience`, giữ `sources.csv`, chạy script kiểm tra CP2. Gom kết quả 4 thành viên, chấm lại theo 3 mức, dẫn phần phân tích lỗi (tìm ra lỗi dòng ghi nguồn của Heading v1 và câu thăm dò filter) |
| Phạm Cường Quốc | 2A202602469 | R2 · Benchmark | Recursive (500) | Viết 5 câu hỏi kèm gold answer, kiểm từng ý chính có trong tài liệu gold và không trùng tài liệu khác, thiết kế câu hỏi cần filter `audience=student` |
| Đỗ Đức Đại | 2A202602725 | R3 · Strategy | HeadingChunker (custom, v1 → v2) | Bảo đảm không trùng chiến lược, chạy baseline `ChunkingStrategyComparator`, viết chunker theo Điều |
| Nguyễn Trường Bảo | 2A202602540 | Report & Demo | FixedSize (500, overlap 100) | Viết báo cáo nhóm từ kết quả đã tổng hợp, chuẩn bị và dẫn phần demo |

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** Khảo thí và phúc khảo, cụ thể là *Quy chế quản lý công tác thi và đánh giá kết quả học tập* của Trường Đại học Công nghiệp TP. Hồ Chí Minh (IUH), ban hành kèm Quyết định số 610/QĐ-ĐHCN ngày 21/02/2025.

**Tại sao nhóm chọn chủ đề này?**
> Quy chế thi quy định **cùng một chủ đề cho hai đối tượng với đáp án khác nhau**. Ví dụ với phúc khảo bài tự luận: người học nộp đơn trong 14 ngày làm việc, còn giảng viên phải chấm xong trong 05 ngày làm việc. Nhờ vậy `metadata_filter={"audience": "student"}` có việc thật để làm. Văn bản được chia sẵn theo Chương/Điều/khoản, rất hợp để thử chunk theo heading, và có nhiều con số, mốc thời gian nên gold answer kiểm chứng được. Ngoài ra, nhiều nhóm khác đã chọn thư viện, nên chủ đề này giúp nhóm có góc nhìn riêng khi thảo luận chéo.

### Danh sách tài liệu (Data Inventory)

9 tài liệu đầu được chép tay từ bản PDF scan của quy chế và tách theo Điều và theo đối tượng. Tài liệu thứ 10 được crawl từ trang HTML của Phòng Khảo thí và ĐBCL. Số ký tự tính trên phần thân, đã bỏ frontmatter.

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | Trách nhiệm của người học khi dự thi (`nguoi-hoc-du-thi`) | [PDF Quy chế 610](https://tqa.iuh.edu.vn/wp-content/uploads/2025/12/Quyet-dinh-so-610-QD-DHCN-ve-viec-ban-hanh-Quy-che-quan-ly-cong-tac-thi-va-danh-gia-ket-qua-hoc-tap..pdf) — Điều 13 | 2026-09-19 / 610/QĐ-ĐHCN ngày 21/02/2025 | 3 217 | audience=student, category=to-chuc-thi |
| 2 | Phúc khảo — quy định dành cho người học (`phuc-khao-nguoi-hoc`) | PDF Quy chế 610 — Điều 26 (khoản 1, 2c, 3, 5) | 2026-09-19 / 610/QĐ-ĐHCN | 1 544 | audience=student, category=phuc-khao |
| 3 | Phúc khảo — quy định dành cho giảng viên và đơn vị (`phuc-khao-giang-vien`) | PDF Quy chế 610 — Điều 26 (khoản 2a, 2b, 4), Điều 27 | 2026-09-19 / 610/QĐ-ĐHCN | 2 442 | audience=faculty, category=phuc-khao |
| 4 | Trách nhiệm của cán bộ coi thi (`can-bo-coi-thi`) | PDF Quy chế 610 — Điều 10 (khoản 3), Điều 11 | 2026-09-19 / 610/QĐ-ĐHCN | 6 373 | audience=faculty, category=to-chuc-thi |
| 5 | Chấm thi và nhập điểm (`cham-thi`) | PDF Quy chế 610 — Điều 21, 23, 24, 25 | 2026-09-19 / 610/QĐ-ĐHCN | 5 316 | audience=faculty, category=cham-thi |
| 6 | Nội dung và biên soạn đề thi (`bien-soan-de-thi`) | PDF Quy chế 610 — Điều 7, 8 | 2026-09-19 / 610/QĐ-ĐHCN | 6 038 | audience=faculty, category=ra-de |
| 7 | Hình thức và thời lượng thi (`hinh-thuc-thoi-luong-thi`) | PDF Quy chế 610 — Điều 6 | 2026-09-19 / 610/QĐ-ĐHCN | 3 426 | audience=all, category=hinh-thuc-thi |
| 8 | Xử lý người học vi phạm quy chế thi (`xu-ly-vi-pham-nguoi-hoc`) | PDF Quy chế 610 — Điều 29 | 2026-09-19 / 610/QĐ-ĐHCN | 3 808 | audience=student, category=xu-ly-vi-pham |
| 9 | Xử lý cán bộ vi phạm quy chế tổ chức thi, chấm thi (`xu-ly-vi-pham-can-bo`) | PDF Quy chế 610 — Điều 30 | 2026-09-19 / 610/QĐ-ĐHCN | 2 752 | audience=faculty, category=xu-ly-vi-pham |
| 10 | Các loại hình thi trực tuyến (`loai-hinh-thi-truc-tuyen`) | [tqa.iuh.edu.vn/cong-tac-khao-thi/loai-hinh-thi-truc-tuyen/](https://tqa.iuh.edu.vn/cong-tac-khao-thi/loai-hinh-thi-truc-tuyen/) | 2026-09-19 / not-stated | 6 954 | audience=all, category=hinh-thuc-thi, source_type=html-crawled |

Phân bố `audience`: faculty 5, student 3, all 2. File `data/khao-thi-phuc-khao/sources.csv` khớp 1-1 với 10 file, và script kiểm tra CP2 của lab báo 10/10 OK.

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

**Ghi chú về quá trình thu thập (những điều nhóm gặp phải):**
- **`robots.txt`:** chỉ cấm `/wp-admin/`. Tuy nhiên tường lửa của trường trả 403 cho User-Agent mặc định `Python-urllib`, và `RobotFileParser` của Python hiểu 403 là "cấm tất cả". Nhóm đọc lại `robots.txt` bằng **đúng UA của crawler** (`Day7DataFoundationsCourse/1.0`) thì được 200 và các trang đều được phép. Nhóm không giả làm trình duyệt, vẫn giữ giãn cách ≥1 giây giữa các request.
- **SSL:** server không gửi chứng chỉ trung gian (RapidSSL), nên Python báo `CERTIFICATE_VERIFY_FAILED`. Nhóm tải chứng chỉ trung gian từ địa chỉ chính thức của DigiCert ghi trong chứng chỉ, ghép vào bộ CA, **vẫn xác thực SSL đầy đủ**.
- **PDF là bản scan** (28 trang, 0 ký tự trích được), nên nhóm chép tay các Điều cần dùng, giữ nguyên văn, và ghi `source_type: pdf-scan-transcribed`. Không đưa PDF thô vào `data/`.
- **Bản tóm tắt trên web lệch với văn bản gốc.** Trang [Quy định phúc khảo](https://tqa.iuh.edu.vn/cong-tac-khao-thi/quy-dinh-phuc-khao/) ghi là trích Điều 26 của chính quy chế này, nhưng ghi "14 ngày" thay vì "14 ngày **làm việc**", và ghi phúc khảo trắc nghiệm "trong vòng 03 ngày" thay vì "trả kết quả trong vòng **2 ngày làm việc**". Nhóm **dùng PDF làm nguồn chính thức** và loại trang tóm tắt, để corpus không chứa hai đáp án mâu thuẫn nhau.
- **Các trang bị loại:**
  - Trang "Tổng hợp kết quả phúc khảo": chỉ là danh sách link tới file kết quả, có thể chứa dữ liệu cá nhân của sinh viên.
  - 4 trang con của "Loại hình thi trực tuyến": trùng nội dung với trang tổng quan.
- **Làm sạch trang crawl:** bỏ menu, footer và khối "TIN NỔI BẬT" (8 169 → 6 954 ký tự). Crawler sinh frontmatter có ngoặc kép (`doc_id: "..."`), khiến script CP2 báo lệch, nên nhóm đã chuẩn hoá lại.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `audience` | enum | `student` / `faculty` / `all` | Trường lọc chính. Tách các quy định cùng từ vựng nhưng khác đối tượng (phúc khảo của người học và của giảng viên) |
| `category` | enum | `phuc-khao`, `cham-thi`, `to-chuc-thi`, `xu-ly-vi-pham`, `ra-de`, `hinh-thuc-thi` | Lọc thêm theo mảng nghiệp vụ, ví dụ chỉ tìm trong `xu-ly-vi-pham` |
| `articles` | string | `Điều 26 (khoản 1, 2c, 3, 5)` | Truy vết câu trả lời về đúng Điều và khoản trong văn bản gốc |
| `source_url` | URL | link PDF / trang HTML | Truy vết nguồn gốc (provenance) |
| `retrieved_at` | date | `2026-09-19` | Biết dữ liệu lấy khi nào để kiểm tra độ mới |
| `document_version` | string | `610/QĐ-ĐHCN ngày 21/02/2025` / `not-stated` | Phân biệt phiên bản quy chế; không bịa số hiệu khi nguồn không nêu |
| `source_type` | enum | `pdf-scan-transcribed` / `html-crawled` | Biết tài liệu nào chép tay (cần kiểm lại khi có nghi ngờ) |
| `doc_id` | string | `phuc-khao-nguoi-hoc` | Trỏ về file gốc; dùng cho `delete_document` và để chấm theo tài liệu |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

> Mỗi thành viên thử **một chiến lược khác nhau** trên cùng bộ tài liệu; nhóm tổng hợp và so sánh ở đây.

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare(body, chunk_size=500)` trên 3 tài liệu khác kiểu nhau: quy chế dài, quy chế ngắn, và bài viết văn xuôi. Frontmatter đã được bỏ trước khi chạy. Cột "trọn câu" là tỉ lệ chunk kết thúc tại ranh giới câu hoặc ý (`. ; : ) ! ?`), dùng làm thước đo khách quan cho việc giữ ngữ cảnh.

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| `can-bo-coi-thi` (6 373 ký tự) | FixedSizeChunker (`fixed_size`) | 13 | 490.3 | Kém: chỉ 15% chunk trọn câu, cắt ngang giữa khoản |
| | SentenceChunker (`by_sentences`) | 20 | 316.9 | Tốt: 100% trọn câu, nhưng có chunk dài tới 618 ký tự vì không giới hạn kích thước |
| | RecursiveChunker (`recursive`) | 18 | 352.4 | Tốt: 94% trọn câu, mỗi khoản thường nằm trọn trong một chunk |
| `phuc-khao-nguoi-hoc` (1 544 ký tự) | FixedSizeChunker (`fixed_size`) | 4 | 386.2 | Kém: 25% trọn câu, chunk cuối chỉ còn 45 ký tự |
| | SentenceChunker (`by_sentences`) | 6 | 255.8 | Trung bình: trọn câu, nhưng nhãn "2. Đối với môn thi tự luận" bị tách khỏi nội dung |
| | RecursiveChunker (`recursive`) | 4 | 384.5 | Trung bình: 50% trọn câu (các dòng tiêu đề khoản không có dấu câu) |
| `loai-hinh-thi-truc-tuyen` (6 954 ký tự) | FixedSizeChunker (`fixed_size`) | 14 | 496.8 | Kém: 7% trọn câu |
| | SentenceChunker (`by_sentences`) | 15 | 461.5 | Tốt về câu, nhưng có chunk 761 ký tự gộp sang mục khác |
| | RecursiveChunker (`recursive`) | 20 | 345.9 | Tốt: 85% trọn câu, cắt theo đoạn văn |

**Nhận xét:** Fixed-size cắt bất chấp cấu trúc, nên con số và điều kiện dễ bị chẻ đôi. Sentence giữ câu trọn vẹn nhưng không kiểm soát được độ dài, và cắt nhầm ở các số thứ tự "1.", "2." của khoản. Recursive là baseline tốt nhất vì mỗi khoản trong quy chế là một đoạn `\n\n`.

### Chiến lược của từng thành viên

**Thành viên 1 — Nguyễn Trường Bảo (Report & Demo)**
- **Loại chiến lược:** FixedSize, `FixedSizeChunker(chunk_size=500, overlap=100)`, tạo 107 chunk.
- **Mô tả & lý do chọn cho chủ đề này:** Làm mốc so sánh đơn giản, dễ tái lập. Overlap 100 ký tự (20%) để một con số hay thời hạn nằm ngay ranh giới vẫn xuất hiện trọn trong ít nhất một chunk.
- **Code snippet (nếu custom):** không (dùng chunker có sẵn).

**Thành viên 2 — Phạm Cường Quốc (R2 · Benchmark)**
- **Loại chiến lược:** Recursive, `RecursiveChunker(chunk_size=500)`, tạo 118 chunk.
- **Mô tả & lý do chọn:** Quy chế viết theo khoản và điểm, mỗi khoản là một đoạn. Recursive tách ở `\n\n` trước nên thường giữ trọn một khoản (một quy định) trong một chunk, và chỉ cắt nhỏ hơn khi khoản quá dài.
- **Code snippet (nếu custom):** không (dùng chunker có sẵn).

**Thành viên 3 — Đỗ Đức Đại (R3 · Strategy)**
- **Loại chiến lược:** custom, `HeadingChunker` (chunk theo Điều), tạo 124 chunk.
- **Mô tả & lý do chọn:**
  - Người soạn quy chế đã chia sẵn văn bản thành các Điều, mỗi Điều là một đơn vị ngữ nghĩa trọn vẹn. Chunker tách tại mỗi dòng `## Điều N` hoặc `Điều N`.
  - Điều nào dài quá 500 ký tự thì hạ xuống `RecursiveChunker`, và **gắn lại tiêu đề Điều vào từng mảnh con** để mảnh sau không mất ngữ cảnh "đây là Điều nói về gì".
  - **v1** gộp phần mở đầu (tiêu đề tài liệu và dòng ghi nguồn) vào chunk đầu tiên. Sau khi phân tích lỗi Q1 (xem mục 3), **v2** bỏ dòng ghi nguồn `> ...` khỏi nội dung chunk, vì thông tin nguồn đã nằm trong metadata.
- **Code snippet (nếu custom):**
```python
class HeadingChunker:
    HEADING = re.compile(r"^(#{2,6}\s+\S.*|Điều\s+\d+.*)$", re.MULTILINE)

    def __init__(self, chunk_size: int = 500, drop_notes: bool = True) -> None:
        self.chunk_size = chunk_size
        self.drop_notes = drop_notes

    def chunk(self, text: str) -> list[str]:
        starts = [m.start() for m in self.HEADING.finditer(text)]
        if not starts:
            return RecursiveChunker(chunk_size=self.chunk_size).chunk(text)

        preamble = text[: starts[0]].strip()
        if self.drop_notes:
            preamble = "\n".join(line for line in preamble.splitlines() if not line.startswith(">")).strip()
        chunks: list[str] = []
        for begin, end in zip(starts, starts[1:] + [len(text)]):
            section = text[begin:end].strip()
            if section:
                chunks.extend(self._split_section(section))
        if preamble:
            chunks[0] = f"{preamble}\n\n{chunks[0]}"
        return chunks

    def _split_section(self, section: str) -> list[str]:
        if len(section) <= self.chunk_size:
            return [section]
        heading, _, body = section.partition("\n")
        inner_size = max(self.chunk_size - len(heading) - 1, 100)
        return [f"{heading}\n{piece}" for piece in RecursiveChunker(chunk_size=inner_size).chunk(body)]
```

**Thành viên 4 — Đỗ Ngọc Phi (Trưởng nhóm · R1 · Data)**
- **Loại chiến lược:** Sentence, `SentenceChunker(max_sentences_per_chunk=3)`, tạo 116 chunk.
- **Mô tả & lý do chọn:** Là chiến lược có sẵn duy nhất còn trống, và dùng để kiểm một giả thuyết riêng: quy định thường được viết thành những câu dài hoàn chỉnh (điều kiện, hậu quả, danh sách ngăn bằng dấu `;`), nên giữ trọn câu có thể giữ trọn ý. Điểm yếu đã biết từ baseline: chunker coi "1.", "2.", "Điều 26." là hết câu, và không giới hạn độ dài (chunk dài nhất 927 ký tự).
- **Code snippet (nếu custom):** không (dùng chunker có sẵn).

### So Sánh Giữa Các Thành Viên

Điểm truy xuất chấm đúng theo `docs/SCORING.md`: 2đ khi top-3 chứa **đủ các ý chính** của gold answer **và** chunk top-1 là đoạn liên quan; 1đ khi chỉ có một phần ý, hoặc đoạn liên quan không ở top-1; 0đ khi không có ý nào (chi tiết ở mục 3).

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Trường Bảo | FixedSize (500, overlap 100) | **5** | Là chiến lược **duy nhất** đưa đoạn chứa "tối đa là 120 phút" lên top-1 ở Q2 (score 0.907): chunk dài và đều nên gom trọn cả khoản thời lượng | Cắt ngang câu: đoạn "làm đơn… trong vòng 14 ngày làm việc" (Q1) rơi khỏi top-3, chỉ còn mảnh "…14 ngày làm việc, phải có sự đồng ý…" nói về nộp muộn. Q4 lấy nhầm Điều |
| Phạm Cường Quốc | Recursive (500) | **6** | Giữ trọn khoản, Q1 đủ 3/3 ý với đoạn đúng ở top-1 | Không biết khoản thuộc Điều nào: Q4 lấy Điều 21 (chấm **tự luận**) và Điều 29e (đạo văn tiểu luận) thay vì Điều 24 (chấm **tiểu luận**) |
| Đỗ Đức Đại | HeadingChunker v1 → v2 | **5 → 7** | Tiêu đề Điều đi kèm mọi mảnh, nên là chiến lược duy nhất đưa được đoạn Điều 24 vào top-3 ở Q4 | v1: dòng ghi nguồn ~400 ký tự gộp vào chunk đầu làm loãng embedding, Q1 được 0 điểm. Các mảnh cùng một Điều có điểm sát nhau nên đoạn đúng hay đứng hạng 2–3 (Q2, Q4). Danh sách dài bị cắt qua 2 chunk (Q5) |
| Đỗ Ngọc Phi | Sentence (3 câu/chunk) | **6** | Là chiến lược **duy nhất gom đủ 3/3 ý ở Q5**: cả danh sách lỗi đình chỉ thi là một câu dài ngăn bằng `;`, nên không bị cắt đôi. Q1 cũng đủ 3/3 ý | Tách nhãn khoản khỏi nội dung: "2." rơi về chunk trước, nên chunk "Đối với môn thi tự luận… 07 ngày" chiếm top-1 ở Q1 thay vì chunk có "14 ngày làm việc". Q4 chỉ lấy được Điều 21 |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> **HeadingChunker v2 (7/10)**. Với văn bản quy chế, lỗi nguy hiểm nhất không phải lấy sai tài liệu mà là **lấy đúng tài liệu nhưng sai Điều**. `cham-thi` có cả Điều 21 (tự luận: không thống nhất thì mời người chấm thứ ba) lẫn Điều 24 (tiểu luận: lệch từ 2 điểm thì báo CNBM), từ vựng gần như giống nhau. Chỉ heading chunker giữ nhãn "Điều 24. Chấm thi tiểu luận, đồ án" trên chunk nên đưa được đoạn này vào top-3. Recursive và Sentence cùng đứng thứ hai (6/10). Recursive tôn trọng khoản nhưng mất ngữ cảnh Điều; Sentence giữ trọn câu dài (thắng ở Q5) nhưng tách số thứ tự khoản khỏi nội dung. Fixed-size (5/10) cắt ngang câu, làm quy trình và thời hạn nằm ở hai chunk khác nhau. Không chiến lược nào vượt quá 7/10: điểm yếu chung là đoạn đúng thường đứng hạng 2–3 thay vì top-1, và danh sách dài bị cắt. Heading chỉ dẫn đầu sau khi sửa lỗi v1, cho thấy chiến lược tốt vẫn cần phân tích lỗi mới phát huy được.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

> **Đúng 5 câu hỏi**, đa dạng, có thể kiểm chứng; **ít nhất 1 câu** cần lọc metadata mới trả lời tốt. Đây là bộ câu hỏi chung cho mọi thành viên chạy.

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Tôi muốn phúc khảo bài thi tự luận thì phải làm gì và trong thời hạn bao lâu? *(dạng quy trình và thời hạn; **cần filter `audience=student`**)* | Làm đơn phúc khảo điểm thi (Mẫu 12), chuyển đơn cùng phiếu đóng tiền phúc khảo đến giáo vụ Khoa/Viện của đơn vị chủ quản học phần **trong vòng 14 ngày làm việc** kể từ ngày điểm thi được công bố. Nộp muộn hơn thì phải được Trưởng đơn vị chủ quản học phần đồng ý. | `phuc-khao-nguoi-hoc`, Điều 26 khoản 1 |
| 2 | Thời lượng tối đa của một bài thi tự luận là bao nhiêu phút? *(tra số liệu)* | Tối thiểu 50 phút và **tối đa 120 phút**, tùy số tín chỉ của học phần và số câu hỏi trong đề. | `hinh-thuc-thoi-luong-thi`, Điều 6 khoản 4d |
| 3 | Đến phòng thi muộn bao lâu thì không được dự thi? *(điều kiện)* | Người học **đến muộn quá 15 phút sau khi đã phát đề thi** sẽ không được dự thi. Người học phải có mặt trước giờ thi ít nhất 15 phút. | `nguoi-hoc-du-thi`, Điều 13 khoản 2 |
| 4 | Hai giảng viên chấm tiểu luận lệch nhau từ 2 điểm trở lên thì xử lý thế nào? *(quy trình)* | Hai GV **thảo luận để thống nhất kết quả**. Nếu không thống nhất được thì **báo CNBM xem xét quyết định**. (Lệch dưới 2 điểm thì lấy trung bình cộng.) | `cham-thi`, Điều 24 khoản 3 |
| 5 | Những lỗi vi phạm nào khiến người học bị đình chỉ thi? *(liệt kê)* | Mang tài liệu hoặc phương tiện bị cấm vào phòng thi; đưa đề thi ra ngoài khu vực thi hoặc nhận bài giải từ bên ngoài; đã bị cảnh cáo trong giờ thi của học phần đó mà vẫn vi phạm; viết, vẽ nội dung không liên quan; gây rối, đe dọa, xúc phạm CBCT hoặc người học khác; không chấp hành yêu cầu của CBCT về kỷ luật phòng thi. Hậu quả: điểm 0 cho học phần. | `xu-ly-vi-pham-nguoi-hoc`, Điều 29 khoản 1c |

Mọi ý chính dùng để chấm (các chuỗi `must_contain` trong `bench.py`) đều đã được kiểm là **có trong tài liệu gold và không xuất hiện ở tài liệu nào khác**.

### Tổng hợp chất lượng truy xuất của nhóm

> Cách chấm (theo `docs/SCORING.md`): **2 điểm/câu** — top-3 chứa chunk liên quan + agent trả lời đúng (2), có liên quan nhưng thiếu/không ở top-1 (1), không có trong top-3 (0).

Nhóm chấm ở **ba mức** để thấy rõ cách chấm ảnh hưởng tới kết luận thế nào:

| Cách chấm | Fixed (Bảo) | Recursive (Quốc) | Heading v1 (Đại) | Heading v2 (Đại) | Sentence (Phi) |
|---|---|---|---|---|---|
| Theo `doc_id` (file gold có trong top-3) | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Theo 1 chuỗi đặc trưng, file gold ở top-1 (lần chấm đầu) | 6/10 | 8/10 | 8/10 | 10/10 | 8/10 |
| **Đủ ý chính + đoạn liên quan ở top-1 (rubric chính thức)** | **5/10** | **6/10** | **5/10** | **7/10** | **6/10** |

Càng chấm kỹ, điểm càng giảm và thứ hạng càng thay đổi. Chấm theo `doc_id` thì 5 cấu hình ngang nhau, nhưng rubric chính thức mới cho thấy khác biệt thật.

Điểm từng câu (rubric chính thức; output đầy đủ trong `report/benchmark/*.txt`):

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Phúc khảo tự luận: làm gì, bao lâu | Recursive, Heading v2 (2đ) | Recursive và Heading v2: 3/3 ý, đoạn đúng ở top-1. Sentence: 3/3 ý nhưng top-1 là đoạn "thông báo kết quả trong 07 ngày" (1đ). Fixed chỉ có mảnh nộp muộn (0/3 ý), Heading v1 cũng 0/3 | **Bỏ filter thì cả 5 cấu hình đều 0đ**: top-3 toàn chunk về "thời lượng/thời gian" (Điều 6, Điều 8 "06 tuần") |
| 2 | Thời lượng tối đa bài tự luận | Fixed, Sentence (2đ); Recursive, Heading (1đ) | Có với cả 5. Fixed và Sentence có đoạn chứa "120 phút" ngay top-1; các cấu hình khác để nó ở hạng 2–3 | Top-1 luôn đúng Điều 6 nhưng có thể là khoản khác (trắc nghiệm, phòng máy): các khoản cùng Điều có điểm sát nhau |
| 3 | Đến muộn bao lâu thì không được thi | Cả 5 (2đ) | Có, top-1 chứa đúng Điều 13 khoản 2 | Câu dễ nhất |
| 4 | Hai GV chấm tiểu luận lệch ≥ 2 điểm | Heading v1/v2 (1đ) | Heading: đoạn Điều 24 ở hạng 3, top-1 là Điều 21. Fixed, Recursive, Sentence: chỉ có Điều 21 (0/2 ý) | **Đúng file nhưng sai Điều**. Bốn chiến lược đều có file `cham-thi` trong top-3 |
| 5 | Lỗi bị đình chỉ thi | Cả 5 (1đ); chỉ Sentence gom đủ 3/3 ý | Fixed, Recursive, Heading thiếu ý "không chấp hành yêu cầu của CBCT". Sentence đủ ý nhưng top-1 là đoạn thủ tục lập biên bản | Điểm c) dài 841 ký tự bị cắt qua 2 chunk ở các chiến lược theo kích thước; nửa sau mất nhãn "c) Đình chỉ thi" nên xếp hạng thấp. Sentence giữ trọn vì cả danh sách là một câu ngăn bằng `;`. Top-3 còn lẫn Điều 30 (vi phạm của **cán bộ**) |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> **Có, và quyết định kết quả ở Q1.**
> - **Có filter** `audience=student`: tập ứng viên giảm từ 118 xuống 24 chunk (Recursive), và Recursive cùng Heading v2 đạt 2/2.
> - **Không filter**: cả 5 cấu hình đều 0/2. Phần "trong thời hạn bao lâu" của câu hỏi kéo lên các chunk về thời lượng thi và hạn nộp tiểu luận (`bien-soan-de-thi`, `hinh-thuc-thoi-luong-thi`); chunk của người học còn không lọt top-3.
>
> **Filter cũng có cái giá.** Nhóm thử câu *"Thi tự luận thì sinh viên được ra về sớm khi nào?"*. Đáp án ("sau 2/3 thời gian làm bài") nằm trong Điều 11 khoản 6, thuộc tài liệu **cán bộ coi thi** (`faculty`).
> - Không filter: chunk này xếp hạng 3–6 tùy chiến lược. Với Sentence nó nằm **ngay trong top-3**.
> - Có filter `student`: chunk này **biến mất hoàn toàn** ở mọi chiến lược, tăng `top_k` bao nhiêu cũng không cứu được.
>
> Ngoài ra, filter so khớp chính xác nên cũng loại luôn tài liệu `audience: all`, ví dụ Điều 6 về thời lượng thi. Filter tăng precision nhưng có thể làm mất recall.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
> 1. **Chấm theo `doc_id` là ảo tưởng.** Cả 4 chiến lược đều 10/10 theo `doc_id`, nhưng chấm đúng rubric chỉ còn 5–7. Q4 là ví dụ điển hình: cả bốn đều có file `cham-thi` trong top-3, nhưng ba chiến lược chỉ lấy được Điều sai.
> 2. **Filter là con dao hai lưỡi.** Q1 từ 0/2 lên 2/2 nhờ `audience=student`, nhưng câu "ra về sớm" thì mất hẳn đáp án (với Sentence, đáp án đang ở hạng 3 thì biến mất) vì quy định nằm trong tài liệu của cán bộ coi thi.
> 3. **Metadata phải nằm trong metadata.** Một dòng ghi nguồn ~400 ký tự nằm trong nội dung chunk đủ làm loãng embedding và đánh rơi đáp án: Heading từ 5/10 lên 7/10 chỉ nhờ bỏ dòng đó ra. Mock embedder đạt **0/10** trên cùng bộ câu hỏi, nên benchmark bắt buộc phải dùng embedder thật.

**Bài học rút ra khi so sánh trong nhóm:**
> Cùng tài liệu và câu hỏi, khác biệt đến từ việc chunk có giữ được **đơn vị ngữ nghĩa của văn bản** hay không. Fixed-size chẻ đôi câu nên quy trình và thời hạn tách rời nhau. Recursive giữ trọn khoản nhưng không biết khoản đó thuộc Điều nào. Heading giữ được cả hai tầng. Sentence cho thấy một đơn vị ngữ nghĩa khác: **câu dài** của văn bản pháp quy (danh sách ngăn bằng `;`) nên nó là chiến lược duy nhất giữ trọn danh sách ở Q5. Mỗi chiến lược thắng ở một kiểu câu hỏi khác nhau, nên kết hợp heading (giữ ngữ cảnh Điều) với ranh giới câu (không cắt danh sách) có thể là hướng tốt nhất.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> 1. Cho `audience` nhận **nhiều giá trị** và cho filter hỗ trợ kiểu "student hoặc all", đồng thời tách riêng các khoản trong tài liệu cán bộ mà người học cũng cần biết (như Điều 11 khoản 6).
> 2. Chunk theo **khoản/điểm** và gắn nhãn điểm (vd. "Điều 29, c) Đình chỉ thi") vào từng mảnh con. Hoặc dùng parent-document retrieval: khớp chunk nhỏ nhưng trả về cả Điều.
> 3. Không để ghi chú nguồn trong phần thân tài liệu ngay từ đầu.
> 4. Kiểm chéo mọi bản tóm tắt trên web với văn bản gốc trước khi đưa vào corpus, vì nhóm đã gặp trường hợp web ghi "14 ngày" còn văn bản gốc ghi "14 ngày làm việc".

### Kịch bản demo (6–8 phút, Nguyễn Trường Bảo dẫn)

Mọi thành viên đều trình bày phần chiến lược của mình. Terminal mở sẵn với `bench.py` đã chạy được.

| Thời lượng | Người trình bày | Nội dung |
|---|---|---|
| 1' | Đỗ Ngọc Phi | Chủ đề, nguồn IUH, cách tách tài liệu theo `audience`, các vấn đề thu thập (robots.txt theo UA, chứng chỉ SSL, PDF scan, bản web lệch với văn bản gốc) |
| 2' | Cả 4 người (~30" mỗi người) | Mỗi người tóm tắt chiến lược và lý do chọn: Bảo (Fixed), Quốc (Recursive), Đại (Heading), Phi (Sentence) |
| 3' | Phạm Cường Quốc, rồi Nguyễn Trường Bảo | Quốc: 5 câu hỏi và cách chấm theo ý chính. Bảo: bảng ba mức chấm (10 → 5–7), A/B filter ở Q1 và câu thăm dò "ra về sớm" |
| 2' | Đỗ Đức Đại | Demo trực tiếp Q4 (đúng file, sai Điều) và Q1 với Heading v1 so với v2 (lỗi dòng ghi nguồn) |
| còn lại | Cả nhóm | Hỏi đáp |

Câu trả lời chuẩn bị sẵn cho ba câu giảng viên hay hỏi:
- *Chuyển chủ đề thì chiến lược nào còn dùng được?* Recursive dùng được gần như mọi nơi. Heading chỉ tốt khi văn bản có Điều/mục rõ ràng. Sentence hợp văn bản pháp quy có câu dài.
- *Metadata filter giúp ở đâu, làm mất kết quả ở đâu?* Giúp ở Q1 (0 → 2 điểm). Làm mất ở câu "ra về sớm" và loại cả tài liệu `audience: all`.
- *Học được gì từ nhóm khác?* Điền sau buổi demo.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá | Căn cứ |
|----------|-------------------|--------|
| Lựa chọn tài liệu (Document Set Quality) | 9 / 10 | 10 tài liệu đủ metadata, `sources.csv` khớp 1-1, script CP2 báo 10/10 OK, ghi rõ quá trình thu thập. Trừ 1 điểm vì 9/10 tài liệu cùng tách từ một PDF scan chép tay, nguồn chưa đa dạng |
| Thiết kế chiến lược (Strategy Design) | 15 / 15 | 4 chiến lược không trùng nhau, baseline có thước đo "trọn câu", HeadingChunker cải tiến từ v1 lên v2 nhờ phân tích lỗi, so sánh điểm mạnh và điểm yếu của từng người |
| Chất lượng truy xuất (Retrieval Quality) | 7 / 10 | Chấm theo rubric chính thức: tốt nhất 7/10 (Heading v2), các chiến lược còn lại 5–6/10 |
| Thuyết trình (Demo) | 4 / 5 | Kịch bản chia đủ người và thời lượng, có câu trả lời chuẩn bị sẵn. Chưa điền phần học được từ nhóm khác |
| **Tổng phần nhóm** | **35 / 40** | |