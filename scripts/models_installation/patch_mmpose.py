# scripts/models_installation/patch_mmpose.py
import os

def patch_setup_py():
    target_file = os.path.join("models", "mmpose", "setup.py")
    
    if not os.path.exists(target_file):
        print(f"❌ Error: Cannot find {target_file}")
        return

    print(f"🔧 Patching {target_file} for PEP 517 compatibility...")

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Original code that triggers execution scope issues in newer Python
    bad_code = "exec(compile(f.read(), version_file, 'exec'))"
    
    # Robust replacement utilizing isolated dictionary
    good_code = "version_vars = {}; exec(compile(f.read(), version_file, 'exec'), version_vars); locals().update(version_vars)"

    # Check if already patched
    if good_code in content:
        print("✅ The file is already patched.")
        return

    # Check if bad_code is present
    if bad_code not in content:
        if "locals()['__version__']" in content:
            print("⚠️ Warning: Exact block pattern not found, attempting fallback...")
        else:
            print("❌ Error: Target code pattern not found. Check MMPose version.")
            return

    # Apply replacement
    new_content = content.replace(bad_code, good_code)
    
    # Robust return handling
    new_content = new_content.replace("return locals()['__version__']", "return version_vars['__version__']")

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("✅ Patch applied successfully!")

if __name__ == "__main__":
    patch_setup_py()