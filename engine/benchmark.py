"""
End-to-End Local Benchmark CLI for VOICERECORE.
Measures Vietnamese Streaming ASR speed + NLLB-200 INT8 translation speed across 10 languages.

Usage:
    uv run --project engine python engine/benchmark.py
"""

from __future__ import annotations

import platform
import time

from asr_engine import VietASRStreamingEngine
from languages import LANGUAGES
from model_manager import ModelManager
from translator import NLLBTranslator


def main() -> None:
    print(f"=== VOICERECORE Benchmark ({platform.system()} {platform.machine()}) ===")
    print(f"Total supported languages in catalog: {len(LANGUAGES)}\n")

    t0 = time.perf_counter()
    asr = VietASRStreamingEngine(provider="cpu")
    print(f"[1/2] VietASR loaded in {(time.perf_counter() - t0) * 1000:.1f} ms (provider={asr.active_provider})")
    asr.close()

    mm = ModelManager()
    model_dir = mm.ensure_nllb_600m(
        on_progress=lambda p: print(f"  download [{p['progress']:5.1f}%] {p['message']}")
    )
    t1 = time.perf_counter()
    tr = NLLBTranslator(model_dir, device="cpu")
    print(f"[2/2] NLLB-200-distilled-600M INT8 loaded in {(time.perf_counter() - t1) * 1000:.1f} ms\n")

    sample = (
        "Xin chào mọi người, hôm nay chúng ta đang thử nghiệm hệ thống nhận dạng giọng nói "
        "tiếng Việt và dịch trực tiếp sang hai trăm ngôn ngữ trên máy tính."
    )
    targets = [
        ("eng_Latn", "English"),
        ("jpn_Jpan", "Japanese"),
        ("kor_Hang", "Korean"),
        ("zho_Hans", "Chinese (Simp)"),
        ("fra_Latn", "French"),
        ("deu_Latn", "German"),
        ("spa_Latn", "Spanish"),
        ("tha_Thai", "Thai"),
        ("rus_Cyrl", "Russian"),
        ("arb_Arab", "Arabic"),
    ]
    print(f"Source (VI): {sample}\n")
    print(f"{'Lang':<16} {'Beam=1 (ms)':>12} {'Beam=2 (ms)':>12}  Translation")
    print("-" * 92)
    for code, label in targets:
        r1 = tr.translate(sample, "vie_Latn", code, beam_size=1)
        r2 = tr.translate(sample, "vie_Latn", code, beam_size=2)
        print(f"{label:<16} {r1.latency_ms:>10.1f}ms {r2.latency_ms:>10.1f}ms  {r2.translated_text}")


if __name__ == "__main__":
    main()
