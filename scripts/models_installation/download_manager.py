# scripts/models_installation/download_manager.py
"""Download manager for pretrained pose estimation backbone weights.

Supports robust chunked streaming, SSL fallback, User-Agent spoofing,
and mirror retries for HRNet-W32, ResNet50, and ViTPose-Small.
"""

import os
import sys
import ssl
import urllib.request

# Ensure UTF-8 output encoding on all consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- MIRROR CONFIGURATIONS ---
MODELS = {
    "ResNet50": {
        "mirrors": [
            "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth",
        ],
        "path": "models/weights/resnet50.pth",
    },
    "HRNet-W32": {
        "mirrors": [
            "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
        ],
        "path": "models/weights/hrnet_w32.pth",
    },
    "ViTPose-Small": {
        "mirrors": [
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"
        ],
        "path": "models/weights/vitpose_small_mmpose.pth",
        "manual_url": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth",
    },
}


def report_progress(downloaded, total_size):
    """Visual progress bar."""
    if total_size > 0:
        percent = downloaded * 100 / total_size
        bar_length = 40
        filled_length = int(bar_length * percent / 100)
        bar = "=" * filled_length + "-" * (bar_length - filled_length)
        sys.stdout.write(
            f"\r[{bar}] {percent:.1f}% ({downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB)"
        )
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB")
        sys.stdout.flush()


def download_model(name, info):
    dest = info["path"]
    mirrors = info["mirrors"]

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"\n[INFO] Processing {name}...")

    # 1. Check if checkpoint already exists and is valid
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size < 10 * 1024 * 1024:
            print(f"   [WARN] Existing file corrupt ({size} bytes). Removing...")
            try:
                os.remove(dest)
            except Exception:
                pass
        else:
            print(
                f"   [OK] {name} already exists and is valid ({size / (1024*1024):.1f} MB). Skipping."
            )
            return

    # 2. Retry loop across mirrors
    success = False
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    for i, url in enumerate(mirrors):
        print(f"   [DOWNLOAD] Attempt {i+1}: Fetching from {url}...")
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
            )
            with urllib.request.urlopen(
                req, context=context, timeout=90
            ) as response, open(dest, "wb") as out_file:
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024 * 1024  # 1MB
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    report_progress(downloaded, total_size)

            if os.path.exists(dest) and os.path.getsize(dest) > 10 * 1024 * 1024:
                print("\n   [SUCCESS] Download completed successfully.")
                success = True
                break
        except Exception as e:
            print(f"\n   [ERROR] Mirror {i+1} failed: {e}")
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except Exception:
                    pass

    if not success:
        print(f"\n   [ERROR] Could not automatically download {name}.")
        print("   [INFO] Reason: Upstream links may be restricted or blocked by firewall.")
        if "manual_url" in info:
            print(f"\n   [MANUAL RESOLUTION]:")
            print(f"   1. Open link: {info['manual_url']}")
            print(f"   2. Download checkpoint weights file")
            print(f"   3. Place it at: {os.path.abspath(dest)}")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print("   MODEL WEIGHTS DOWNLOAD MANAGER")
    print("=" * 60)
    for name, info in MODELS.items():
        download_model(name, info)
    print("\n" + "=" * 60)
    print("[SUCCESS] All model checkpoints ready.")
    print("=" * 60)