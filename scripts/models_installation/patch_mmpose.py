# scripts/models_installation/patch_mmpose.py
import os
from pathlib import Path

def patch_setup_py():
    target_file = os.path.join("models", "mmpose", "setup.py")
    
    if not os.path.exists(target_file):
        print(f"[ERROR] Cannot find {target_file}")
        return

    print(f"[INFO] Patching {target_file} for PEP 517 compatibility...")

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Original code that triggers execution scope issues in newer Python
    bad_code = "exec(compile(f.read(), version_file, 'exec'))"
    
    # Robust replacement utilizing isolated dictionary
    good_code = "version_vars = {}; exec(compile(f.read(), version_file, 'exec'), version_vars); locals().update(version_vars)"

    # Check if already patched
    if good_code in content:
        print("[OK] setup.py is already patched.")
        return

    # Check if bad_code is present
    if bad_code not in content:
        if "locals()['__version__']" in content:
            print("[WARN] Exact block pattern not found, attempting fallback...")
        else:
            print("[ERROR] Target code pattern not found. Check MMPose version.")
            return

    # Apply replacement
    new_content = content.replace(bad_code, good_code)
    new_content = new_content.replace("return locals()['__version__']", "return version_vars['__version__']")

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("[SUCCESS] setup.py patch applied successfully!")


def patch_mmpose_configs():
    """Patch MMPose model configs to enable DARK (unbiased=True) and UDP codecs."""
    mmpose_configs = Path("models") / "mmpose" / "configs" / "body_2d_keypoint" / "topdown_heatmap" / "coco"
    
    if not mmpose_configs.exists():
        print(f"[WARN] MMPose configs directory not found at {mmpose_configs}")
        return

    print("[INFO] Patching MMPose configs for DARK (unbiased=True) and UDP codecs...")

    # 1. HRNet-W32 (DARK codec)
    hrnet_cfg = mmpose_configs / "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    if hrnet_cfg.exists():
        with open(hrnet_cfg, "r", encoding="utf-8") as f:
            content = f.read()
        if "unbiased=True" not in content and "type='MSRAHeatmap'" in content:
            new_content = content.replace("sigma=2)", "sigma=2, unbiased=True)")
            with open(hrnet_cfg, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("[SUCCESS] Patched HRNet-W32 config with DARK (unbiased=True).")
        else:
            print("[OK] HRNet-W32 config already configured with DARK (unbiased=True).")

    # 2. ResNet-50 (DARK codec)
    resnet_cfg = mmpose_configs / "td-hm_res50_8xb64-210e_coco-256x192.py"
    if resnet_cfg.exists():
        with open(resnet_cfg, "r", encoding="utf-8") as f:
            content = f.read()
        if "unbiased=True" not in content and "type='MSRAHeatmap'" in content:
            new_content = content.replace("sigma=2)", "sigma=2, unbiased=True)")
            with open(resnet_cfg, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("[SUCCESS] Patched ResNet-50 config with DARK (unbiased=True).")
        else:
            print("[OK] ResNet-50 config already configured with DARK (unbiased=True).")

    # 3. ViTPose-Small (UDP codec)
    vitpose_cfg = mmpose_configs / "td-hm_ViTPose-small_8xb64-210e_coco-256x192.py"
    if vitpose_cfg.exists():
        with open(vitpose_cfg, "r", encoding="utf-8") as f:
            content = f.read()
        if "type='MSRAHeatmap'" in content:
            new_content = content.replace("type='MSRAHeatmap'", "type='UDPHeatmap'")
            with open(vitpose_cfg, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("[SUCCESS] Patched ViTPose-Small config with UDP codec.")
        else:
            print("[OK] ViTPose-Small config already configured with UDP codec.")


if __name__ == "__main__":
    patch_setup_py()
    patch_mmpose_configs()