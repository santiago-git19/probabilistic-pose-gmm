"""Helper script to execute specific subsets of ablation experiments.

Usage:
    # Run baseline + each individual component
    python src/experiments/run_ablation_subset.py --subset basic

    # Run scale experiments across varying configurations
    python src/experiments/run_ablation_subset.py --subset scale

    # Run primary ablation combinations
    python src/experiments/run_ablation_subset.py --subset combined

    # Run custom list
    python src/experiments/run_ablation_subset.py --experiments "00,01,02,14,24"
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.experiments.compare_experiments import EXPERIMENTS, main

# Define experiment subsets
SUBSETS = {
    "basic": [
        "00_Baseline_NoTTA",
        "01_FlipOnly",
        "02_ScaleOnly_Conservative",
        "09_PhotometricOnly_Brightness",
        "10_PhotometricOnly_Contrast",
        "11_PhotometricOnly_Noise",
        "12_PhotometricOnly_Blur",
    ],
    "scale": [
        "00_Baseline_NoTTA",
        "02_ScaleOnly_Conservative",
        "03_ScaleOnly_Moderate",
        "04_ScaleOnly_Aggressive",
        "05_ScaleOnly_HighPeakExp",
        "06_ScaleOnly_HighSharpExp",
        "07_ScaleOnly_GentleSharpness",
        "08_ScaleOnly_MaxConfidence",
    ],
    "photometric": [
        "00_Baseline_NoTTA",
        "09_PhotometricOnly_Brightness",
        "10_PhotometricOnly_Contrast",
        "11_PhotometricOnly_Noise",
        "12_PhotometricOnly_Blur",
        "13_PhotometricOnly_AllCombined",
    ],
    "combined": [
        "00_Baseline_NoTTA",
        "01_FlipOnly",
        "14_FlipScale_Conservative",
        "15_FlipScale_Moderate",
        "20_FlipPhoto_AllCombined",
        "22_ScalePhoto_AllCombined",
        "23_FullStack_Conservative",
        "24_FullStack_Moderate",
        "25_FullStack_Aggressive",
    ],
    "quick": [
        "00_Baseline_NoTTA",
        "01_FlipOnly",
        "03_ScaleOnly_Moderate",
        "13_PhotometricOnly_AllCombined",
        "24_FullStack_Moderate",
    ],
}


def filter_experiments(keys):
    """Filter EXPERIMENTS dict to only include specified keys."""
    return {k: v for k, v in EXPERIMENTS.items() if k in keys}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study subsets")
    parser.add_argument(
        "--subset",
        choices=list(SUBSETS.keys()),
        help="Predefined subset to run",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        help="Comma-separated list of experiment prefixes (e.g., '00,01,14')",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("\n=== ALL EXPERIMENTS ===")
        for i, name in enumerate(EXPERIMENTS.keys(), 1):
            print(f"{i:2d}. {name}")
        print(f"\nTotal: {len(EXPERIMENTS)} experiments")
        
        print("\n=== AVAILABLE SUBSETS ===")
        for subset_name, exp_list in SUBSETS.items():
            print(f"\n{subset_name} ({len(exp_list)} experiments):")
            for exp in exp_list:
                print(f"  - {exp}")
        sys.exit(0)

    # Determine which experiments to run
    if args.subset:
        selected = SUBSETS[args.subset]
        print(f"\n>>> Running subset '{args.subset}' ({len(selected)} experiments)")
    elif args.experiments:
        prefixes = args.experiments.split(",")
        selected = [
            name for name in EXPERIMENTS.keys()
            if any(name.startswith(p.strip()) for p in prefixes)
        ]
        print(f"\n>>> Running custom selection ({len(selected)} experiments)")
    else:
        print("ERROR: Must specify --subset or --experiments")
        parser.print_help()
        sys.exit(1)

    print("Selected experiments:")
    for name in selected:
        print(f"  - {name}")
    print()

    # Temporarily replace global EXPERIMENTS
    import src.experiments.compare_experiments as cmp_module
    original_experiments = cmp_module.EXPERIMENTS.copy()
    cmp_module.EXPERIMENTS = filter_experiments(selected)

    try:
        main()
    finally:
        # Restore original
        cmp_module.EXPERIMENTS = original_experiments
