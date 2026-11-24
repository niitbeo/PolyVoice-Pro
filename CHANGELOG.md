# Changelog — DAU Voice Pro

Tất cả thay đổi đáng chú ý của **DAU Voice Pro** được ghi nhận tại đây.

---

## [1.1.1] — 2026-11-24 *(Bản phát hành chính thức mới nhất)*

Bản phát hành hoàn chỉnh mang đến **Sân khấu LED Hội trường**, **Slide-Aware RAG 100% Offline**, **Bảo mật Wi-Fi LAN hội nghị** và toàn bộ bản vá độ ổn định qua kiểm định đa lớp cho **Windows x64** và **macOS Apple Silicon**.

### ✨ Tính năng mới
- **Sân khấu LED Hội trường (`led-stage`):**
  - Hiển thị 3–4 câu chốt hoàn chỉnh đứng yên cỡ chữ lớn (0% nhảy sửa chữ), kèm thanh nghe sống mờ và nhãn `Slide #N` cập nhật trực tiếp theo diễn giả.
  - **QR góc màn LED** kèm vùng an toàn không che phụ đề (hỗ trợ đủ cả 5 chế độ: `Sân khấu LED`, `Xếp tầng`, `YouTube`, `Chỉ dịch`, `50/50 Song ngữ`).
  - **Tạm ẩn màn chờ DAU (`Display Freeze`):** 1 chạm chuyển tức thì cả **Màn LED**, **điện thoại Khán giả (`/audience`)** và **OBS (`/obs`)** sang màn hình chờ hội nghị khi nghỉ giải lao hoặc xử lý sự cố micro.
  - **Sửa nhanh trực tiếp lên LED & QR (`Pencil`):** Biên tập viên sửa tức thì bản dịch trong Lịch sử và đồng bộ ngay lên màn LED, `/audience` (kể cả máy chọn ngôn ngữ riêng) và `/obs`.
- **Nạp Slide & Bối cảnh Hội nghị 100% Offline (`engine/slide_context.py` + `SlideContextModal`):**
  - Kéo thả trực tiếp `.PPTX`, `.DOCX`, `.PDF` (`pypdf` pure-Python, không cần thư viện C), `.TXT`/`.MD` hoặc dán chương trình hội nghị trước giờ G.
  - Tự động rút tên riêng, chức danh, thuật ngữ song ngữ và cung cấp **DAU Conference Preset (22 thuật ngữ chuẩn ĐH Kiến trúc Đà Nẵng)** nạp 1 chạm.
  - Đối chiếu trang Slide theo thời gian thực (`< 1 ms` CPU, hỗ trợ nhảy phi tuyến Q&A) để bơm **ASR Hotwords** và khóa thuật ngữ cho cả **Hy-MT2**, **EnViT5**, **NLLB-200** và **Cloud API**.

### 🛡️ Bảo mật, Độ ổn định & Hiệu năng
- **Chặn truy cập quản trị từ Wi-Fi LAN (`LoopbackGuardMiddleware`):** Điện thoại khán giả trên mạng Wi-Fi hội trường chỉ truy cập được `/audience`, `/obs`, `/ws/audience`; toàn bộ `/api/*` quản trị và `/ws` bị khóa cứng chỉ cho `localhost` (`403` / `1008`).
- **Chống sai nghĩa đa ngôn ngữ & Slide IN HOA:**
  - Ngăn tiêu đề Slide viết IN HOA khóa nhầm từ thường (`SMART`, `CITY`, `GREEN`, `KHOA`, `XANH`…), giữ nhận dạng viết tắt chữ thường từ `Zipformer-30M` (`bim`, `dau`, `esg`…), và chống rò tiếng Anh từ DAU Preset sang Nhật/Hàn/Trung Phồn thể/Thái/Pháp.
  - Chuẩn hóa **Unicode NFC/NFD** (khắc phục lỗi Keynote macOS xuất NFD), gọt động từ dẫn đầu cặp thuật ngữ song ngữ, và cache regex thuật ngữ (`~15×` nhanh hơn khi nhiều thuật ngữ).
  - Sửa lỗi Moonshine-v2 tách chữ viết tắt `A. I` → `AI`, `A. G. I.` → `AGI` và bổ sung từ nối lửng tiếng Việt (`phải`, `cần`, `luôn`, `vẫn`…) trong `_ends_with_connector`.
- **Chịu tải 50+ điện thoại Wi-Fi & Ổn định dài giờ:**
  - Vá rò rỉ WebSocket khi client ngắt kết nối đột ngột, phát song song có timeout `2.0s` + tự dọn socket chết, gộp yêu cầu dịch lịch sử khi nhiều điện thoại quét QR cùng lúc, sửa lấy LAN IP khi hội trường không có Internet.
  - Vá `NameError: HYMT2_LANG_NAMES` khi đổi ngôn ngữ ở chế độ `hy-mt2`; chống rò tiến trình `llama-server` khi dừng sớm; kiểm tra giới hạn upload 40 MB theo luồng + đẩy nạp hotwords sang luồng nền không làm khựng sự kiện; giới hạn bộ nhớ `_history` (300 câu) và hàng đợi `_final_mt_queue`.
- **Desktop Shell (Tauri/Rust) & Giao diện:**
  - Đóng cửa sổ LED bằng `Alt+F4`/`Cmd+W` chỉ ẩn cửa sổ chứ không hủy vĩnh viễn; sửa Auto-Updater tải xong gói cài mới tắt sidecar (tự khởi động lại engine nếu huỷ UAC/rớt mạng); dọn sạch tiến trình con trên macOS khi thoát.
  - Sửa padding `SlideContextModal`, chống tràn nút Topbar từ `820px` đến `1280px+`, nâng thanh điều khiển LED lên trên màn tạm ẩn để không bị kẹt ở chế độ toàn màn hình.
- **Đóng gói & CI/CD:**
  - Đồng bộ 4 chiều phiên bản trên toàn bộ manifest/lockfile, thêm chốt chặn CI không cho ghi đè bản release đã phát hành, đóng gói `--collect-all pypdf python_multipart`, và mở rộng smoke test sidecar cho toàn bộ API `v1.1.x` trên cả Windows x64 và macOS arm64.

---

## [1.0.0] — 2026-11-23

- Định danh thương hiệu chính thức **DAU Voice Pro** (logo Trường Đại học Kiến trúc Đà Nẵng trong suốt, tối giản thanh trạng thái).
- Đồng bộ toàn bộ icon sang **Lucide Icons** và hộp chọn **StudioSelect** thống nhất 2 giao diện Sáng/Tối.

---

## [0.9.9] — 2026-11-23

- Hoàn thiện bộ tiền phát hành **100% Offline** với 9 bản vá lõi (chống lặp Moonshine-v2, giữ nguyên viết hoa đúng chuẩn, dọn ngữ cảnh khi đổi ngôn ngữ giữa phiên).
- Mở rộng **12 nút ngôn ngữ truy cập nhanh** (thêm **Đài Loan Phồn thể `zho_Hant`**) và tối ưu tương thích đường dẫn Unicode/DLL trên **Windows x64**.

---

## [0.9.8] — 2026-11-23

- Vá 6 nhóm lỗi song chiều **Hàn ⇄ Việt** (0% rò placeholder `XTerm`, 0% rò ngữ cảnh, giảm CER tiếng Hàn thực đo từ `13.57%` xuống `2.84%`).
- Tinh gọn Kho Model còn **10 mô hình cốt lõi** dễ chọn, tự dọn RAM thông minh.

---

## [0.9.7] — 2026-11-22

- Tích hợp **Parakeet-TDT-0.6B-Ultra INT8** (25 ngôn ngữ Châu Âu, nhanh hơn `7×–10×` Whisper, **0% GPU**) vào Gói 3 và Dịch File Offline.

---

## [0.9.6] — 2026-11-22

- Ra mắt kiến trúc đa ngôn ngữ 3 tầng, 3 gói cài đặt 1 chạm, Fast-Lane **Nhật/Hàn/Thái/Trung**, dịch file ghi âm/video nâng cao và chống lùi dòng Spotlight.

---

## [0.9.5] — 2026-11-21

- Ra mắt tính năng **Kiểm định Chất lượng Dịch & Chính xác Ngữ nghĩa bằng Google AI (`Gemini 2.5 Flash`)** ngay trong Lịch sử (chấm điểm `0–100%`, phân biệt lỗi ASR vs lỗi dịch, sửa 1 chạm & thêm Từ điển).

---

## [0.9.0 – 0.9.4] — 2026-11-20

- Ra mắt **Chế độ Phụ đề YouTube**, kiến trúc **Zero-Overlap Silero VAD v5 + Moonshine-v2**, luồng dịch nháp/chốt tách đôi (91–193 ms), mặc định siêu nhẹ **EnViT5 + NLLB-200 (0% GPU)**, sửa lỗi PortAudio SIGSEGV trên macOS ARM64 và khử spin-wait OpenMP 200% CPU.

---

## [0.8.7 – 0.8.8] — 2026-11-20

- Bổ sung **Whisper Large-v3-Turbo + Silero VAD v5** (99 ngôn ngữ offline), tự chốt câu cho ngôn ngữ không dấu cách, dịch file đa ngôn ngữ và vá lỗi rò ngữ cảnh đoạn dài trên Hy-MT2.
