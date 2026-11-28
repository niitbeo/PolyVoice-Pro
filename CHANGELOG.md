# Changelog — DAU Voice Pro

Tất cả thay đổi đáng chú ý của **DAU Voice Pro** được ghi nhận tại đây.

---

## [1.3.0] — 2026-11-28 *(Bản phát hành chính thức mới nhất)*

### 🎬 Thanh Phụ Đề Kính Mờ Trong Suốt 1 Chạm — Dưới Slide (Màn 2) & Nổi Trên Video
- **1 chạm ghim Dưới Slide / Video ngay thanh công cụ chính:** Tự động phát hiện màn hình thứ 2 (máy chiếu / màn hình trình chiếu PowerPoint, Keynote, Google Slides) để ghim dải phụ đề kính mờ gọn đẹp ở mép dưới **Màn 2** và **tự bật Xuyên chuột** giúp diễn giả bấm chuyển slide mượt mà không bị vướng; khi dùng 1 màn hình sẽ mở dải phụ đề nổi tự do để kéo thả lên **YouTube, Zoom, hoặc video MP4**.
- **Cửa sổ trong suốt hoàn toàn 100% không viền, không bóng đổ (`macOS` & `Windows`):** Hỗ trợ 4 mức nền kính mờ linh hoạt (**Mờ 75% mặc định / Đậm 90% / Trong suốt 0% chữ có viền sắc nét / Mờ 55%**), 2 màu chữ tương phản cao (**Trắng / Vàng** đồng bộ cả chữ chốt lẫn chữ đang dịch trực tiếp), bật/tắt nhanh dòng câu gốc (**Gốc: Bật/Tắt**), tăng/giảm cỡ chữ (`16px–56px`), đổi vị trí **Mép dưới ↔ Mép trên**, và theo mọi Space trên macOS (`visible_on_all_workspaces`).
- **Thanh điều khiển tự ẩn thông minh & Chuyển mượt sang Màn LED lớn:** Rê chuột vào thanh phụ đề để chỉnh nhanh hoặc kéo vị trí, tự ẩn hoàn toàn sau khi rời chuột hoặc đứng yên > 3,2 giây để màn hình chỉ còn phụ đề thuần sạch; bấm **`Màn LED`** để mở rộng lại đầy đủ 6 chế độ hội trường (và ngược lại) mà không mất cấu hình kích thước riêng.
- **Phím tắt toàn cục `Cmd + Shift + X` (macOS) / `Ctrl + Shift + X` (Windows):** Bật hoặc tắt nhanh chế độ **Xuyên chuột** từ bất kỳ ứng dụng nào (kể cả khi đang trình chiếu PowerPoint toàn màn hình hoặc xem YouTube).

---

## [1.2.0] — 2026-11-26

### 🌐 Phát Mã QR Hội nghị Công khai 4G/5G Miễn phí HTTPS (Zero-Server Tunnel)
- **Phát QR Công khai 1 chạm (`*.trycloudflare.com`, \$0, không cần tài khoản):** Đại biểu dùng **4G/5G hoặc Wi-Fi cách ly hội trường** quét QR xem phụ đề đa ngôn ngữ tức thì mà không cần mở port router hay thuê máy chủ. Tự tải `cloudflared` chính chủ (~18 MB, chỉ 1 lần duy nhất cho macOS & Windows) kèm thanh tiến độ và xác minh DNS qua `1.1.1.1` DoH trước khi hiển thị QR.
- **Tùy chọn Tên miền cố định của trường (`Cloudflare Named Tunnel`):** Hỗ trợ dán Token để phát trên tên miền cố định (VD: `https://live.dau.edu.vn`) in sẵn lên standee/backdrop hội nghị.
- **Bảo mật cô lập 2 lớp & Khóa mã PIN 4 số:** Mở riêng cổng ASGI cô lập `127.0.0.1:18768` chỉ phục vụ `/audience` và `/ws/audience` (tuyệt đối không lộ `/api/*`, `/ws`, API Key hay log); gia cố `LoopbackGuardMiddleware` chặn mọi header reverse-proxy (`CF-Connecting-IP`, `CF-Ray`, `X-Forwarded-For`); tùy chọn khóa PIN 4 số tự nhúng vào QR; tự động đồng bộ mã QR góc trên Màn hình LED và bổ sung WebSocket Heartbeat 25 giây chống ngắt kết nối nhàn rỗi.

### ☁️ Tích hợp Cloud AI Toàn diện — Soniox Real-Time Pro & 100% Google Cloud
- **Soniox Real-Time Pro (`stt-rt-v6`, 60 ngôn ngữ, ~0% CPU):** Nghe + dịch trực tiếp trên mây, hỗ trợ **Dịch 2 chiều tự động (`two_way`)** trên 1 Micro/Mixer, phân biệt nhiều diễn giả (`S1`/`S2`), tự nối phiên WebSocket mượt mỗi 4 phút giữ nguyên ngữ cảnh 500 ký tự, và nạp sẵn Từ điển & Slide DAU.
- **100% Google Cloud (Speech-to-Text + Translation API):** Gỡ khóa tuần tự HTTP để dịch nháp & câu chốt chạy song song trên Google Cloud (`95–165 ms`, khớp 0% giật chữ khi chốt), bổ sung `GoogleCloudStreamingASR` (`speech.googleapis.com/v1`) dùng chung 1 API Key, giao diện Cloud gọn chỉ còn 2 tab chính (**Soniox Real-Time** & **Google Cloud**).

### 🖥️ Màn hình LED Toàn văn, Thư viện Phiên Họp & Chống Giật Chữ Chuẩn Quốc Tế
- **Chế độ `Toàn văn` (`full-text`) trên Màn hình LED & Lịch sử:** Chạy văn bản song ngữ 2 cột (hoặc 1 cột) nối liền mạch không thẻ/không timestamp, cỡ chữ lên tới `140px`, 3 màu chữ (Trắng / Vàng / Xanh), tự ẩn thanh trên sau 3 giây khi đang phát, tự động chừa vùng đệm an toàn cho `QR góc` & thanh điều khiển nổi.
- **Thư viện Phiên Họp (`Thư viện`):** Xem lại, tìm kiếm, sao chép và tải lại toàn bộ biên bản `.TXT` / phụ đề `.SRT` đã tự lưu ngay trong ứng dụng.
- **Khử giật khung & Chế độ Êm mắt chuẩn Wordly (0% nhảy chữ):** Khóa chiều cao khung Spotlight (`0,0px` rung lắc), đồng nhất độ đậm font chữ nháp (`font-weight: 700`), tối ưu nhịp câu tiếng Việt chống treo câu dài 30–49 giây, và đưa công tắc **`Dịch nháp khi đang nói (Tắt = 0% nhảy chữ)`** ra ngay cột trái cùng thanh ticker mảnh ở đáy khung.
- **Tối ưu hiển thị điện thoại QR (`390px`) & LED chế độ Sáng:** Hiển thị rõ nhãn `Song ngữ` trên màn hình iPhone `390px`, đồng bộ nhãn diễn giả `S1`/`S2`, và nâng độ tương phản chế độ `Sáng` cho máy chiếu hội trường ban ngày.

---

## [1.1.4] — 2026-11-25

### 🍎 Khắc phục triệt để quyền `Âm thanh máy` (ScreenCaptureKit) trên macOS
- **Niêm phong chữ ký gói `.app` chuẩn Apple Silicon (`APPLE_SIGNING_IDENTITY="-"` + `Entitlements.plist`):** Khắc phục lỗi macOS TCC từ chối `Âm thanh máy` ngay cả khi người dùng đã cấp quyền trong *Ghi màn hình & Âm thanh hệ thống* (do bản build CI trước đó thiếu niêm phong `CodeResources` & `Info.plist=not bound`).
- **Ký chuẩn `voicerecore-macos-helper`:** Bổ sung `codesign --force --sign -` gắn chặt `Info.plist` vào helper Swift thu âm thanh hệ thống.

### 🎯 Nâng cấp Tự nhận diện 99 ngôn ngữ (`auto`), Tiếng Nhật & Dịch File song ngữ
- **Hỗ trợ đầy đủ `Tự nhận diện (99 NN)` trong Dịch File Offline (`file_translator.py`):** Tự động nhận ngôn ngữ từng phân đoạn trong file hội nghị song ngữ (`Nhật + Anh + Việt`), tự xả ngữ cảnh khi đổi ngôn ngữ (vá lỗi HTTP 500 khi chọn `auto`), bổ sung chế độ siêu tốc `Moonshine-JA` / `Moonshine-KO` khi dịch file Nhật/Hàn.
- **Tối ưu tốc độ & chống nuốt câu `Whisper-Turbo (auto)`:** Bỏ qua bước gọi SLID thừa với các hệ chữ không nhập nhằng (Nhật/Hàn/Thái/Lào/Khmer/Việt — tiết kiệm ~1.5–2s/câu), trả từng câu VAD độc lập để không bao giờ xóa hàng đợi câu kế tiếp hay trộn 2 ngôn ngữ, bỏ Pass-2 `Moonshine-EN` và carry-over khi đang chạy `Whisper-Turbo`, lọc lặp dấu `... ... ...` và phân biệt chuẩn tiếng Pháp/Bồ Đào Nha với tiếng Việt (`â/ê/ô/ã/õ`).
- **Làm sạch đầu ra `Moonshine-JA`:** Chuẩn hóa số & thương hiệu tách rời (`i P h o n e 6` → `iPhone 6`, `2 0 1 2 年` → `2012年`, `５Ｇ` → `5G`, `１，０００` → `1,000`), chặn chuỗi phiên âm Katakana-chấm rác khi có người nói tiếng Anh vào mode Nhật, nối mượt cụm CJK không chèn dấu cách ASCII.
- **Chuẩn hóa thuật ngữ DAU đa ngữ & Màn LED gọn hơn:** Thêm quy tắc 1 chiều `ダナン建築大学` / `Danang University of Architecture` → `Đại học Kiến trúc Đà Nẵng` (chống dịch lệch thành `Dannan`/`Danan`), và tự động ẩn dòng câu gốc trùng lặp trên Màn LED khi diễn giả nói trùng ngôn ngữ đích (`passthrough`).

---

## [1.1.3] — 2026-11-25

### ✨ Tính năng mới
- **Tự động nhận diện ngôn ngữ nói (99 NN — `Nói: Tự nhận diện (99 NN)`):**
  - Chạy 100% Offline trên `Whisper Large-v3-Turbo` kết hợp xác thực chữ viết đa ngữ (`infer_text_nllb_lang`) cho Anh, Việt, Nhật, Hàn, Trung, Thái, Lào, Khmer, Pháp, Đức, Nga…
  - Tự động chốt đoạn cũ, xả ngữ cảnh MT cũ và hiển thị nhãn sống **`Tự nhận: Tiếng Anh / Tiếng Nhật / ...`** trên cả Giao diện chính lẫn Màn LED ngay khi diễn giả quốc tế đổi ngôn ngữ trên sân khấu.
  - Hỗ trợ cả hộp dịch nhanh văn bản khi đang ở chế độ `Tự nhận diện`.

### 🖥️ Khắc phục triệt để Màn hình LED (`led-stage`) không còn nuốt chữ nói
- **Thanh `Đang nghe:` ở đáy màn LED:** Bỏ cắt cụt 1 dòng `text-overflow: ellipsis`, nâng cấp hiển thị mượt tối đa 2 dòng và tự cuộn theo đuôi câu mới nhất (`tailClipLiveText`), đảm bảo chữ diễn giả vừa nói luôn hiện rõ 100%.
- **Hiển thị câu gốc ngay trên bản dịch LED (`Gốc: Bật / Tắt`):** Mặc định bật dòng lời nói gốc gọn phía trên mỗi câu chốt LED (có nút bật/tắt 1 chạm ngay trên thanh công cụ Overlay).
- **Sửa lỗi Flexbox cắt đầu câu & tự cuộn:** Thay `justify-content: flex-end` bằng `margin-top: auto` chuẩn W3C kèm `ledScrollRef` tự cuộn xuống câu mới nhất ở mọi cỡ chữ LED (`16–56px`).

### 🛡️ Cải thiện độ chính xác ASR Tiếng Nhật & Dịch Việt ⇄ Anh (`EnViT5`)
- **Nâng cấp `Moonshine-JA` & `Whisper-Turbo`:** Thêm bộ đệm tiền âm `320ms preroll` chống nuốt âm đầu câu, nới giới hạn câu (`9.5s` / `8.5s`) tránh cắt cụt giữa từ, truyền `rms` thực vào bộ lọc, chặn 100% ảo giác sách nói Aozora Bunko / YouTube outro bị cắt đuôi / rác Latinh toàn chiều rộng và khử lặp cụm 2x–3x tiếng Nhật.
- **Chốt chặn `EnViT5` chống nhả tiếng Việt & mất placeholder `XTerm`:** Tự động phát hiện khi `EnViT5` làm rơi `XTerm` (lỗi biến tên trường thành `"Max"`) hoặc nhả ngược tiếng Việt sang bản dịch tiếng Anh (`"Rector lên nói..."`), lập tức dịch lại bằng `beam=1` / dự phòng sạch sang `NLLB-200-600M`.

---

## [1.1.2] — 2026-11-24

### ✨ Tinh chỉnh giao diện & Tài liệu
- **Hộp thoại thoát ứng dụng gọn, chuyên nghiệp:** Thu gọn hộp thoại (`390px`), tiêu đề `Thoát DAU Voice Pro?`, thông điệp 1 dòng rõ ràng theo ngữ cảnh (bình thường / đang dịch live / đang chạy tác vụ), bỏ các ghi chú thừa và căn phải 2 nút `Hủy` | `Thoát`.
- **Hoàn thiện tài liệu phát hành:** Bổ sung `CHANGELOG.md` đầy đủ và cập nhật `README.md` cho dòng `v1.1.x`.

---

## [1.1.1] — 2026-11-24

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
