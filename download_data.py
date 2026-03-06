#!/usr/bin/env python3
"""
Download a conversational dataset for BissiMamba training.

Strategy (tries each source in order until one succeeds):
  1. HuggingFace `datasets` library  (pip install datasets)
  2. HuggingFace Datasets Server API  (public JSON endpoint, no auth)
  3. Cornell Movie Dialogs Corpus     (direct GitHub raw download)

Output: data/train.txt  — Human:/Bot: conversation pairs, UTF-8

Usage:
    python3 download_data.py
"""

import os, sys, json, time, zipfile, io
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: pip install requests", file=sys.stderr); sys.exit(1)

OUT_DIR  = Path("data")
OUT_FILE = OUT_DIR / "train.txt"
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Shared formatter
# ─────────────────────────────────────────────────────────────────────

def clean(s: str) -> str:
    return " ".join(s.strip().split())

def turns_to_text(turns: list) -> str:
    parts = []
    for i in range(0, len(turns) - 1, 2):
        h = clean(turns[i])
        b = clean(turns[i + 1])
        if h and b:
            parts.append(f"Human: {h}\nBot: {b}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Source 1: HuggingFace `datasets` library
# ─────────────────────────────────────────────────────────────────────

def try_hf_library() -> bool:
    try:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "datasets"],
            check=True, capture_output=True
        )
        from datasets import load_dataset          # type: ignore
        print("Using HuggingFace `datasets` library…")
        ds = load_dataset("daily_dialog", trust_remote_code=True)
        count = 0
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            for split_name in ("train", "validation", "test"):
                split = ds.get(split_name)
                if split is None:
                    continue
                for row in split:
                    text = turns_to_text(row["dialog"])
                    if text:
                        f.write(text + "\n\n")
                        count += 1
                print(f"  {split_name}: {count} dialogs so far")
        print(f"Done — {count} dialogs → {OUT_FILE}")
        return True
    except Exception as exc:
        print(f"  [hf-library] skipped: {exc}", file=sys.stderr)
        return False


# ─────────────────────────────────────────────────────────────────────
# Source 2: HuggingFace Datasets Server  (JSON REST API)
# ─────────────────────────────────────────────────────────────────────

def _hf_api_rows(dataset, config, split, offset, length, timeout=20):
    """Single page fetch from the HuggingFace Datasets Server."""
    url = "https://datasets-server.huggingface.co/rows"
    r = requests.get(url, params={
        "dataset": dataset, "config": config,
        "split": split, "offset": offset, "length": length,
    }, timeout=timeout)
    r.raise_for_status()
    return r.json()

def try_hf_api() -> bool:
    """Try several dataset / config combinations."""
    candidates = [
        ("daily_dialog",         "default"),
        ("daily_dialog",         "all"),
        ("Aeala/daily-dialog",   "default"),
        ("roskoN/daily_dialog",  "default"),
        ("blended_skill_talk",   "default"),
    ]
    for dataset, config in candidates:
        try:
            print(f"  Trying HF API: {dataset} / {config} …")
            page = _hf_api_rows(dataset, config, "train", 0, 1, timeout=10)
            rows = page.get("rows", [])
            if not rows:
                continue
            # Found a working combo — fetch everything
            print(f"Using HF Datasets Server: {dataset} / {config}")
            count  = 0
            offset = 0
            PAGE   = 100
            with open(OUT_FILE, "w", encoding="utf-8") as f:
                while True:
                    page = _hf_api_rows(dataset, config, "train", offset, PAGE)
                    rows = page.get("rows", [])
                    if not rows:
                        break
                    for item in rows:
                        row    = item.get("row", item)
                        dialog = row.get("dialog") or row.get("turns") or []
                        text   = turns_to_text(dialog)
                        if text:
                            f.write(text + "\n\n")
                            count += 1
                    offset += len(rows)
                    print(f"  {offset} rows / {count} dialogs", end="\r")
                    if len(rows) < PAGE:
                        break
                    time.sleep(0.05)
            print(f"\nDone — {count} dialogs → {OUT_FILE}")
            return True
        except Exception as exc:
            print(f"    skipped: {exc}", file=sys.stderr)
    return False


# ─────────────────────────────────────────────────────────────────────
# Source 3: Cornell Movie Dialogs (GitHub raw)
# ─────────────────────────────────────────────────────────────────────

CORNELL_LINES  = (
    "https://raw.githubusercontent.com/suriyadeepan/"
    "datasets/master/seq2seq/cornell_movie_corpus/movie_lines.txt"
)
CORNELL_CONVOS = (
    "https://raw.githubusercontent.com/suriyadeepan/"
    "datasets/master/seq2seq/cornell_movie_corpus/movie_conversations.txt"
)

def _fetch_text(url: str, timeout=30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

def try_cornell() -> bool:
    try:
        print("Using Cornell Movie Dialogs (GitHub raw)…")
        lines_raw  = _fetch_text(CORNELL_LINES)
        convos_raw = _fetch_text(CORNELL_CONVOS)

        # Parse lines: id → text
        id2text: dict = {}
        for line in lines_raw.splitlines():
            parts = line.split(" +++$+++ ")
            if len(parts) >= 5:
                id2text[parts[0].strip()] = parts[4].strip()

        # Parse conversations
        count = 0
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            for line in convos_raw.splitlines():
                parts = line.split(" +++$+++ ")
                if len(parts) < 4:
                    continue
                # ids are like "['L194', 'L195', 'L196']"
                raw = parts[3].strip().strip("[]").replace("'", "")
                ids = [x.strip() for x in raw.split(",")]
                turns = [id2text[i] for i in ids if i in id2text]
                text = turns_to_text(turns)
                if text:
                    f.write(text + "\n\n")
                    count += 1
        print(f"Done — {count} dialogs → {OUT_FILE}")
        return True
    except Exception as exc:
        print(f"  [cornell] skipped: {exc}", file=sys.stderr)
        return False


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("BissiMamba — Dataset Downloader")
    print("=" * 40)

    for attempt in (try_hf_library, try_hf_api, try_cornell):
        if attempt():
            break
    else:
        print("\nERROR: all download sources failed.", file=sys.stderr)
        print("Check your internet connection or manually place a text", file=sys.stderr)
        print("file of Human:/Bot: exchanges at data/train.txt", file=sys.stderr)
        sys.exit(1)

    size_kb = OUT_FILE.stat().st_size // 1024
    print(f"\nFile size : {size_kb} KB")
    print("\nWorkflow:")
    print("  # CPU small model:")
    print("  make mamba_lm_train && ./mamba_lm_train")
    print()
    print("  # CUDA 1B model (after installing NVIDIA driver + CUDA toolkit):")
    print("  make train_large && ./train_large              # 1 B params")
    print("  make train_large && ./train_large data/train.txt ckpt.bin 10 --small  # 7 M params")


if __name__ == "__main__":
    main()
