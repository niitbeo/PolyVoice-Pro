# PolyVoice Pro — Phiên Dịch Hội Nghị Song Ngữ Real-Time (202 Ngôn Ngữ • Win & Mac)

Ứng dụng desktop đa nền tảng (**Windows `.exe`/`.msi` & macOS `.dmg`**) phiên dịch giọng nói 2 chiều theo thời gian thực (**20 ngôn ngữ nói ⇄ 202 ngôn ngữ dịch**) với kiến trúc **100% Offline** kết hợp **Công tắc bật/tắt Cloud API Chính Thức** (Google Gemini / Microsoft Azure Free F0 2M ký tự / Google Cloud) và **Tự động cập nhật 1 chạm qua GitHub Releases**.

## Tính năng nổi bật (v0.8.0)

- **Chủ động chuyển `Offline | Cloud` 1 chạm:**
  - **Chế độ Offline (0 đ, không cần Internet):** Gói cốt lõi siêu gọn chỉ **~1.18 GB** (`Zipformer-30M Việt` 49MB + `Zipformer-20M / Moonshine-v2 Anh` 265MB + `VietAI EnViT5 Việt⇄Anh` 266MB + `NLLB-200 600M Lào & 202 NN` 604MB), tùy chọn mở rộng `Hy-MT2-1.8B` (+1.1 GB cho Tiếng Thái / 38 NN chuẩn bản xứ).
  - **Chế độ Cloud API Chính Thức (icon bánh răng `⚙️`):** Hỗ trợ **Microsoft Azure Translator (2M free/tháng)**, **Google Cloud Translation (500K free/tháng)** và **Google Gemini (`gemini-2.5-flash`)**. Nháp khi đang nói luôn chạy Offline (tiết kiệm ~85% hạn mức) và tự khóa ở ngưỡng miễn phí \$0.
- **Chuẩn Hội trường & Thuyết trình Song ngữ:**
  - **Màn hình Phụ đề Nổi chia đôi 50/50** kéo sang màn hình thứ 2 (`Đầy màn hình`, `Xuyên chuột`).
  - **Mã QR Wi-Fi Nội bộ (`/audience`)**: Khán giả quét QR bằng điện thoại (không cần Internet) tự chọn ngôn ngữ phụ đề riêng (**Anh / Thái / Lào / Khmer / Trung / Nhật / Hàn...**).
  - **OBS Browser Source (`/obs`)**, **Dịch File ghi âm/video (`.mp3/.wav/.mp4 → .SRT`)**, **Thu âm 2 chiều Zoom/Meet (`[Bạn]` + `[Đối tác]`)**, **Từ điển Họp + BPE Hotwords**, **Tự lưu biên bản `.TXT`/`.SRT`** và **Tóm tắt cuộc họp song ngữ**.
- **Giao diện Crimson Academic Pro (Sáng & Tối):** Tự đồng bộ chế độ Sáng/Tối trên cả App chính, Cửa sổ Phụ đề Nổi và Trang điện thoại khán giả.
- **Tự động cập nhật 1 chạm:** Khi phát hành phiên bản mới trên [GitHub Releases](https://github.com/niitbeo/PolyVoice-Pro/releases), người dùng mở app sẽ nhận thông báo và bấm **Cập nhật & Khởi động lại** ngay trong ứng dụng (giữ nguyên toàn bộ model offline & từ điển).

## Chạy chế độ phát triển (macOS & Windows)

```bash
# 1. Cài backend AI local (Python 3.12 qua uv, không cần PyTorch)
cd engine && uv sync && cd ..

# 2. Cài frontend & chạy Desktop App Native
pnpm install
pnpm tauri dev
```

## Phát hành phiên bản mới & Tự động cập nhật cho mọi người dùng

Chỉ cần chạy **1 lệnh duy nhất** sau khi chỉnh sửa code:

```bash
bash scripts/release.sh 0.8.1 "Mô tả tính năng hoặc bản sửa lỗi mới"
```

GitHub Actions (`.github/workflows/release.yml`) sẽ tự động build bản cài đặt **Windows `.exe`/`.msi`** & **macOS `.dmg`**, ký số bảo mật **Ed25519** và phát hành kèm `latest.json` lên GitHub Releases để toàn bộ máy đã cài đặt tự nhận bản cập nhật.
