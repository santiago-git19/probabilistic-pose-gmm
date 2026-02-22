"""
Quick reference for interpreting TTA ablation study results.

After running experiments, analyze the CSV output to answer key questions.
"""

import pandas as pd
from pathlib import Path

# Example: Load the latest comparison results
results_path = Path("outputs/comparisons/comparison_results_latest.csv")

if results_path.exists():
    df = pd.read_csv(results_path)
    
    print("=" * 80)
    print("TTA ABLATION STUDY - ANALYSIS GUIDE")
    print("=" * 80)
    
    # 1. Best overall improvement
    print("\n1. BEST DELTA OKS (Improvement over baseline)")
    print("-" * 80)
    best = df.nlargest(5, 'delta_oks_mean')[['experiment', 'delta_oks_mean', 'oks_ours_mean', 'elapsed_s']]
    print(best.to_string(index=False))
    
    # 2. Individual components
    print("\n\n2. INDIVIDUAL COMPONENTS (Flip, Scale, Photometric only)")
    print("-" * 80)
    components = df[df['experiment'].str.match(r'^(01_FlipOnly|0[2-8]_ScaleOnly|0[9-9]_PhotometricOnly|1[0-3]_PhotometricOnly)')]
    if not components.empty:
        print(components[['experiment', 'delta_oks_mean', 'elapsed_s']].to_string(index=False))
    
    # 3. Scale configurations comparison
    print("\n\n3. SCALE TTA CONFIGURATIONS")
    print("-" * 80)
    scale_exps = df[df['experiment'].str.contains('Scale', case=False)]
    if not scale_exps.empty:
        print(scale_exps[['experiment', 'delta_oks_mean', 'nll_mean', 'elapsed_s']].to_string(index=False))
    
    # 4. Efficiency analysis
    print("\n\n4. EFFICIENCY (Delta OKS per second)")
    print("-" * 80)
    df_eff = df.copy()
    df_eff['efficiency'] = df_eff['delta_oks_mean'] / df_eff['elapsed_s']
    efficient = df_eff.nlargest(5, 'efficiency')[['experiment', 'delta_oks_mean', 'elapsed_s', 'efficiency']]
    print(efficient.to_string(index=False))
    
    # 5. Uncertainty quality (NLL)
    print("\n\n5. UNCERTAINTY QUALITY (Lower NLL = better)")
    print("-" * 80)
    best_nll = df.nsmallest(5, 'nll_mean')[['experiment', 'nll_mean', 'entropy_mean', 'delta_oks_mean']]
    print(best_nll.to_string(index=False))
    
    # 6. Summary statistics
    print("\n\n6. SUMMARY STATISTICS")
    print("-" * 80)
    baseline = df[df['experiment'].str.contains('Baseline', case=False)]
    if not baseline.empty:
        baseline_oks = baseline['oks_base_mean'].iloc[0]
        print(f"Baseline OKS:        {baseline_oks:.4f}")
        print(f"Best OKS achieved:   {df['oks_ours_mean'].max():.4f}  (+{(df['oks_ours_mean'].max() - baseline_oks)*100:.2f}%)")
        print(f"Median improvement:  {df['delta_oks_mean'].median():.4f}")
        print(f"Max improvement:     {df['delta_oks_mean'].max():.4f}")
        print(f"Min improvement:     {df['delta_oks_mean'].min():.4f}")
    
    # 7. Recommendations
    print("\n\n7. RECOMMENDATIONS")
    print("=" * 80)
    
    # Best overall
    best_overall = df.loc[df['delta_oks_mean'].idxmax()]
    print(f"\n✓ Best accuracy:     {best_overall['experiment']}")
    print(f"  Delta OKS: +{best_overall['delta_oks_mean']:.4f}, Time: {best_overall['elapsed_s']:.1f}s")
    
    # Best efficiency
    df_eff = df.copy()
    df_eff['efficiency'] = df_eff['delta_oks_mean'] / df_eff['elapsed_s']
    best_eff = df_eff.loc[df_eff['efficiency'].idxmax()]
    print(f"\n✓ Best efficiency:   {best_eff['experiment']}")
    print(f"  Delta OKS: +{best_eff['delta_oks_mean']:.4f}, Time: {best_eff['elapsed_s']:.1f}s, Eff: {best_eff['efficiency']:.6f}")
    
    # Best uncertainty
    best_unc = df.loc[df['nll_mean'].idxmin()]
    print(f"\n✓ Best uncertainty:  {best_unc['experiment']}")
    print(f"  NLL: {best_unc['nll_mean']:.4f}, Delta OKS: +{best_unc['delta_oks_mean']:.4f}")
    
    print("\n" + "=" * 80)

else:
    print(f"ERROR: Results file not found at {results_path}")
    print("\nRun experiments first:")
    print("  python src/experiments/compare_experiments.py")
    print("or")
    print("  python src/experiments/run_ablation_subset.py --subset quick")
