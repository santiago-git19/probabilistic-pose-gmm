# scripts/models_installation/download_manager.py
import urllib.request
import os
import sys

# --- MIRROR CONFIGURATIONS ---
MODELS = {
    "ResNet50": {
        "mirrors": [
            "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth",
        ],
        "path": "models/weights/resnet50.pth"
    },
    "HRNet-W32": {
        "mirrors": [
            "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
        ],
        "path": "models/weights/hrnet_w32.pth"
    },
    "ViTPose-Small": {
        "mirrors": [
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"
        ],
        "path": "models/weights/vitpose_small_mmpose.pth",
        "manual_url": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"
    }
}

def report_progress(block_num, block_size, total_size):
    """Visual progress bar."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        bar_length = 40
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)')
        sys.stdout.flush()

def download_model(name, info):
    dest = info['path']
    mirrors = info['mirrors']
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"\n🔍 Processing {name}...")
    
    # 1. Check if checkpoint already exists and is valid
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if size < 10 * 1024 * 1024: 
            print(f"   ⚠️ Existing file corrupt ({size} bytes). Removing...")
            try:
                os.remove(dest)
            except:
                pass
        else:
            print(f"   ✅ {name} already exists and is valid ({size / (1024*1024):.1f} MB). Skipping.")
            return

    # 2. Retry loop across mirrors
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)

    success = False
    for i, url in enumerate(mirrors):
        print(f"   ⬇️ Attempt {i+1}: Downloading from mirror {i+1}...")
        try:
            urllib.request.urlretrieve(url, dest, report_progress)
            if os.path.exists(dest) and os.path.getsize(dest) > 10 * 1024 * 1024:
                print("\n   ✅ Download completed successfully.")
                success = True
                break
        except Exception as e:
            print(f"\n   ❌ Mirror {i+1} failed: {e}")
            if os.path.exists(dest):
                try: os.remove(dest)
                except: pass
    
    if not success:
        print(f"\n   ❌ Could not automatically download {name}.")
        print("   ⚠️  Reason: Upstream links may be restricted or blocked by firewall.")
        if "manual_url" in info:
            print(f"\n   👉 MANUAL RESOLUTION:")
            print(f"   1. Open link: {info['manual_url']}")
            print(f"   2. Download checkpoint weights file")
            print(f"   3. Place it at: {os.path.abspath(dest)}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 MODEL WEIGHTS DOWNLOAD MANAGER")
    for name, info in MODELS.items():
        download_model(name, info)
    print("\n✨ Download process finished.")