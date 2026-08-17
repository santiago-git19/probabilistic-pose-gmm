# src/tests/check_env.py
import os
import sys
import torch
import warnings

# Ignore initialization warnings
warnings.filterwarnings("ignore")

def main():
    print("=== ENVIRONMENT HEALTH CHECK ===")
    
    # 1. HARDWARE VERIFICATION
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    
    # 2. DEFINE ABSOLUTE PATHS
    PROJECT_ROOT = os.path.abspath(os.getcwd())
    MMPOSE_ROOT = os.path.join(PROJECT_ROOT, "models", "mmpose")
    
    # Absolute paths to config and weights
    config_file = os.path.join(
        MMPOSE_ROOT, "configs", 
        "body_2d_keypoint", "topdown_heatmap", "coco", 
        "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    )
    checkpoint_file = os.path.join(PROJECT_ROOT, "models", "weights", "hrnet_w32.pth")

    # 3. VERIFY PHYSICAL ASSETS EXISTENCE
    if not os.path.exists(MMPOSE_ROOT):
        print(f"❌ ERROR: Cannot find models/mmpose directory at:\n   {MMPOSE_ROOT}")
        return
    if not os.path.exists(config_file):
        print(f"❌ ERROR: Cannot find config file at:\n   {config_file}")
        return
    if not os.path.exists(checkpoint_file):
        print(f"❌ ERROR: Cannot find weights file at:\n   {checkpoint_file}")
        return

    # 4. LOAD MODEL WITH WORKING DIRECTORY CONTEXT SWITCH
    print("🔄 Attempting to load HRNet-W32...")
    
    try:
        from mmpose.apis import init_model
        from mmpose.utils import register_all_modules
        
        register_all_modules()
        
        # Switch working directory to models/mmpose so internal relative imports resolve
        print(f"   📂 Switching context to: {MMPOSE_ROOT}")
        os.chdir(MMPOSE_ROOT)
        
        model = init_model(config_file, checkpoint_file, device='cpu')
        
        # Restore project root working directory
        os.chdir(PROJECT_ROOT)
        print(f"   📂 Context restored to: {PROJECT_ROOT}")
        
        print("✅ SUCCESS! Model initialized into memory.")
        print(f"   (Model: HRNet initialized with {sum(p.numel() for p in model.parameters()):,} parameters)")
        
    except Exception as e:
        os.chdir(PROJECT_ROOT)
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()