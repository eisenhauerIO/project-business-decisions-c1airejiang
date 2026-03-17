# SERA Figure Replication
# Reproduces all figures from arxiv 2601.20789 using the paper's raw data.
 
# Figures produced:
#   1. figure1a_glm45.png  — Cost vs Accuracy scaling (GLM-4.5-Air teacher)
#   2. figure1a_glm46.png  — Cost vs Accuracy scaling (GLM-4.6 teacher)
#   3. truncation.png      — Context truncation ratio ablation
#   4. figure1b_spec.png   — Repository specialization α=1.0 vs α=0.0
#   5. snr_table.png       — Signal-to-noise ratio summary table
 
# Usage:
#   pip install matplotlib numpy
#   python sera_figures.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "figure.facecolor":  "#0e0f13",
    "axes.facecolor":    "#16181f",
    "axes.edgecolor":    "#2a2d3a",
    "axes.labelcolor":   "#8890a8",
    "axes.grid":         True,
    "grid.color":        "#2a2d3a",
    "grid.linewidth":    0.8,
    "xtick.color":       "#4a5068",
    "ytick.color":       "#4a5068",
    "text.color":        "#e8eaf0",
    "legend.facecolor":  "#1e2029",
    "legend.edgecolor":  "#2a2d3a",
    "legend.labelcolor": "#8890a8",
    "font.family":       "monospace",
    "font.size":         10,
    "savefig.facecolor": "#0e0f13",
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
})

# Palette 

C_BLUE    = "#7eb8f7"
C_CORAL   = "#f57c6b"
C_GOLD    = "#f5c542"
C_LAVENDER = "#b39ddb"
C_GREEN   = "#6fcf97"
C_ORANGE  = "#ffb347"
C_DIM     = "#8890a8"
C_FAINT   = "#4a5068"

# Raw data extracted from the paper (Tables 1, 2, 3, and Appendix D.4)

COST_VLLM     = 0.1307 + 0.056
COST_ZAI      = 0.0358 + 0.056
COST_GLM46    = (748881*0.11 + 24327*0.60 + 7673*2.20) / 1e6 + 0.056
COST_GLM46_VLLM = (8*8*43 / 19200) * 2 + 0.056
 
SAMPLE_SIZES_45 = np.array([400, 750, 1500, 3000, 4200, 7400, 16000, 20000])
ALL_SEEDS_45 = np.array([
    [34.40, 36.80, 38.20, 40.60, 40.60, 43.20, 47.00, 44.20],
    [33.00, 35.00, 40.20, 37.80, 45.80, 45.40, 47.00, 42.00],
    [33.00, 37.40, 38.20, 40.60, 39.00, 43.40, 45.80, 46.40],
])
 
SAMPLE_SIZES_46 = np.array([1500, 3000, 4200, 7400, 16000, 25000])
ALL_SEEDS_46 = np.array([
    [38.40, 43.60, 40.40, 44.80, 45.80, 51.20],
    [39.00, 42.60, 41.00, 42.40, 47.20, 47.40],
    [37.00, 42.00, 44.20, 45.60, 48.20, 50.00],
])
 
TRUNCATION_RATIOS = np.array([1.00, 0.95, 0.92, 0.86, 0.81, 0.76, 0.70])
TRUNCATION_PERF = np.array([
    [40.20, 41.40, 40.20],
    [44.40, 43.80, 40.80],
    [43.00, 40.80, 43.00],
    [40.60, 43.20, 42.40],
    [42.00, 42.00, 42.20],
    [41.20, 41.80, 41.40],
    [39.00, 42.00, 40.60],
])
 
# 5 checkpoints × 3 seeds
SPEC_1_0 = np.array([
    [25.11, 34.63, 43.04, 42.86, 54.11],
    [28.57, 41.13, 43.72, 42.00, 51.51],
    [28.57, 40.69, 41.99, 45.89, 51.08],
])
SPEC_0_0 = np.array([
    [22.94, 31.60, 41.13, 37.83, 48.03],
    [19.91, 31.30, 39.39, 35.93, 48.05],
    [23.58, 31.30, 35.06, 39.13, 47.62],
])

def add_title (ax, fig_label, title, subtitle=None):
    ax.set_title(
        f"[{fig_label}] {title}" + (f"\n{subtitle}" if subtitle else ""),
        loc="left", pad=10, fontsize=12, color="#e8eaf0", fontweight="bold")
    
def finalize(fig, path) :
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved figure to {path}")

# GLM-4.5-Air

def figure1a_glm45():
    costs_vllm = SAMPLE_SIZES_45 * COST_VLLM / 1000
    costs_zai  = SAMPLE_SIZES_45 * COST_ZAI  / 1000
 
    avg = ALL_SEEDS_45.mean(axis=0)
    std = ALL_SEEDS_45.std(axis=0)
 
    fig, ax = plt.subplots(figsize=(7, 4.5))
    add_title(ax, "Fig 1a", "Cost vs. Accuracy — GLM-4.5-Air Teacher",
              "SWE-bench Verified · 3 seeds per sample size")
 
    # Seed scatter
    for seed in ALL_SEEDS_45:
        ax.scatter(costs_vllm, seed, color=C_BLUE,  alpha=0.25, s=22, zorder=2)
        ax.scatter(costs_zai,  seed, color=C_CORAL, alpha=0.25, s=22, zorder=2)
 
    # Mean lines with error bands
    ax.fill_between(costs_vllm, avg - std, avg + std, color=C_BLUE,  alpha=0.08)
    ax.fill_between(costs_zai,  avg - std, avg + std, color=C_CORAL, alpha=0.08)
 
    ax.plot(costs_vllm, avg, "-o", color=C_BLUE,  lw=2.2, ms=6, label="vLLM (mean ± σ)", zorder=4)
    ax.plot(costs_zai,  avg, "-o", color=C_CORAL, lw=2.2, ms=6, label="z.ai API (mean ± σ)", zorder=4)
 
    # Annotate sample sizes
    for i, n in enumerate(SAMPLE_SIZES_45[::2]):
        ax.annotate(f"{n//1000}k", (costs_vllm[i*2], avg[i*2]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=8, color=C_FAINT, ha="center")
 
    ax.set_xlabel("Training Data Cost ($k)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(30, 50)
    ax.legend(framealpha=0.6, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
 
    finalize(fig, "figure1a_glm45.png")

# GLM-4.6

def figure1a_glm46():
    costs_zai  = SAMPLE_SIZES_46 * COST_GLM46      / 1000
    costs_vllm = SAMPLE_SIZES_46 * COST_GLM46_VLLM / 1000
 
    avg = ALL_SEEDS_46.mean(axis=0)
    std = ALL_SEEDS_46.std(axis=0)
 
    fig, ax = plt.subplots(figsize=(7, 4.5))
    add_title(ax, "Fig 1a", "Cost vs. Accuracy — GLM-4.6 Teacher",
              "SWE-bench Verified · 3 seeds per sample size")
 
    for seed in ALL_SEEDS_46:
        ax.scatter(costs_zai,  seed, color=C_GOLD,   alpha=0.25, s=22, zorder=2)
        ax.scatter(costs_vllm, seed, color=C_LAVENDER, alpha=0.25, s=22, zorder=2)

    ax.fill_between(costs_zai,  avg - std, avg + std, color=C_GOLD,   alpha=0.1)
    ax.fill_between(costs_vllm, avg - std, avg + std, color=C_LAVENDER, alpha=0.1)

    ax.plot(costs_zai,  avg, "-o", color=C_GOLD,   lw=2.2, ms=6, label="z.ai API (mean ± σ)", zorder=4)
    ax.plot(costs_vllm, avg, "-o", color=C_LAVENDER, lw=2.2, ms=6, label="vLLM (mean ± σ)",     zorder=4)
 
    for i, n in enumerate(SAMPLE_SIZES_46):
        ax.annotate(f"{n//1000}k", (costs_zai[i], avg[i]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=8, color=C_FAINT, ha="center")
 
    ax.set_xlabel("Training Data Cost ($k)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(34, 55)
    ax.legend(framealpha=0.6, fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
 
    finalize(fig, "figure1a_glm46.png")

# Truncation Ablation

def figure_truncation():
    avg = TRUNCATION_PERF.mean(axis=1)
    std = TRUNCATION_PERF.std(axis=1)
 
    fig, ax = plt.subplots(figsize=(7, 4.5))
    add_title(ax, "Ablation", "Context Truncation Ratio vs. Accuracy",
              "3 seeds per truncation ratio · GLM-4.5-Air")
 
    # Individual seeds
    for s in range(3):
        ax.scatter(TRUNCATION_RATIOS, TRUNCATION_PERF[:, s],
                   color=C_GREEN, alpha=0.3, s=25, zorder=2)
 
    ax.fill_between(TRUNCATION_RATIOS, avg - std, avg + std,
                    color=C_GREEN, alpha=0.12)
    ax.plot(TRUNCATION_RATIOS, avg, "-o", color=C_GREEN,
            lw=2.2, ms=7, label="Mean accuracy", zorder=4)
    ax.errorbar(TRUNCATION_RATIOS, avg, yerr=std,
                fmt="none", color=C_GREEN, alpha=0.5, capsize=4, lw=1.2)
 
    # Mark best
    best_idx = np.argmax(avg)
    ax.annotate(f"  peak {avg[best_idx]:.1f}%\n  @ ratio={TRUNCATION_RATIOS[best_idx]}",
                (TRUNCATION_RATIOS[best_idx], avg[best_idx]),
                fontsize=8.5, color=C_GREEN, va="bottom")
 
    ax.set_xlabel("Truncation Ratio")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(37, 47)
    ax.invert_xaxis()   # 1.0 on left → 0.70 on right matches paper layout
    ax.legend(framealpha=0.6, fontsize=9)
 
    finalize(fig, "truncation.png")

# Repository Specialization

def figure1b_spec():
    xs = np.arange(5)
    avg10 = SPEC_1_0.mean(axis=0)
    std10 = SPEC_1_0.std(axis=0)
    avg00 = SPEC_0_0.mean(axis=0)
    std00 = SPEC_0_0.std(axis=0)
 
    fig, ax = plt.subplots(figsize=(7, 4.5))
    add_title(ax, "Fig 1b", "Repository Specialization — α=1.0 vs α=0.0",
              "SWE-bench Verified · 3 seeds · 5 checkpoints")
 
    for seed in SPEC_1_0:
        ax.scatter(xs, seed, color=C_BLUE,  alpha=0.25, s=22, zorder=2)
    for seed in SPEC_0_0:
        ax.scatter(xs, seed, color=C_CORAL, alpha=0.25, s=22, zorder=2)
 
    ax.fill_between(xs, avg10 - std10, avg10 + std10, color=C_BLUE,  alpha=0.1)
    ax.fill_between(xs, avg00 - std00, avg00 + std00, color=C_CORAL, alpha=0.1)
 
    ax.plot(xs, avg10, "-o", color=C_BLUE,  lw=2.2, ms=7,
            label=f"SERA  α=1.0  (final={avg10[-1]:.1f}%)", zorder=4)
    ax.plot(xs, avg00, "-o", color=C_CORAL, lw=2.2, ms=7,
            label=f"Baseline α=0.0  (final={avg00[-1]:.1f}%)", zorder=4)
 
    # Delta annotation at final checkpoint
    delta = avg10[-1] - avg00[-1]
    ax.annotate(f"  Δ={delta:+.1f}pp", (xs[-1], (avg10[-1]+avg00[-1])/2),
                fontsize=9, color=C_DIM, va="center")
 
    ax.set_xticks(xs)
    ax.set_xticklabels([f"ckpt {i}" for i in xs])
    ax.set_xlabel("Specialization Checkpoint")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(18, 58)
    ax.legend(framealpha=0.6, fontsize=9)
 
    finalize(fig, "figure1b_spec.png")

# Signal-to-Noise Ratio Table

def figure_snr_table():
    def snr(g1, g2):
        diff    = abs(np.mean(g1) - np.mean(g2))
        avg_std = (np.std(g1) + np.std(g2)) / 2
        return diff / avg_std if avg_std > 0 else np.inf, diff, avg_std
 
    rows = [
        ("Truncation 0.95 vs 1.00",
         TRUNCATION_PERF[1], TRUNCATION_PERF[0],
         "Does 5% truncation help?"),
 
        ("Truncation 0.92 vs 1.00",
         TRUNCATION_PERF[2], TRUNCATION_PERF[0],
         "8% truncation effect"),
 
        ("SERA-32B 16k vs GLM-4.5-Air teacher",
         ALL_SEEDS_45[:, 6], np.array([51.2, 51.4, 49.0]),
         "Gap to teacher model"),
 
        ("GLM-4.6: 25k vs 16k samples",
         ALL_SEEDS_46[:, 5], ALL_SEEDS_46[:, 4],
         "Marginal gain from more data"),
 
        ("GLM-4.6 vs GLM-4.5-Air @ 16k",
         ALL_SEEDS_46[:, 4], ALL_SEEDS_45[:, 6],
         "Teacher model upgrade"),
 
        ("SERA α=1.0 vs α=0.0 (final ckpt)",
         SPEC_1_0[:, 4], SPEC_0_0[:, 4],
         "Specialization benefit"),
    ]
 
    def verdict(s):
        if s < 1:   return "Noise dominates", C_CORAL
        if s < 2:   return "Borderline",      C_ORANGE
        if s < 3:   return "Decent signal",   C_GREEN
        return              "Strong signal",   C_BLUE
 
    col_labels = ["Comparison", "Δ Mean", "Avg σ", "SNR", "Verdict", "Note"]
    cell_data  = []
    row_colors = []
    snr_vals   = []
 
    for name, g1, g2, note in rows:
        s, diff, avg_s = snr(g1, g2)
        verdict_str, _ = verdict(s)
        snr_vals.append((s, verdict(s)))
        cell_data.append([
            name,
            f"{diff:.1f}%",
            f"±{avg_s:.1f}%",
            f"{s:.1f}×" if np.isfinite(s) else "∞",
            verdict_str,
            note,
        ])
        row_colors.append(["#16181f"] * 6)
 
    fig, ax = plt.subplots(figsize=(13, 4.2))
    fig.patch.set_facecolor("#0e0f13")
    ax.axis("off")
 
    ax.set_title(
        "[SNR Analysis]  Signal-to-Noise Ratio — Key Comparisons\n"
        "SNR = |ΔMean| / avg seed σ  ·  higher = more reliable effect",
        loc="left", pad=12, fontsize=11, color="#e8eaf0", fontfamily="monospace",
    )
 
    col_widths = [0.27, 0.07, 0.07, 0.07, 0.13, 0.27]
    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)
 
    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1e2029")
        cell.set_text_props(color=C_FAINT, fontfamily="monospace", fontsize=8.5)
        cell.set_edgecolor("#2a2d3a")
 
    # Style data rows
    for i, (_, (_, color)) in enumerate(snr_vals):
        for j in range(len(col_labels)):
            cell = table[i+1, j]
            cell.set_facecolor("#16181f" if i % 2 == 0 else "#1a1c24")
            cell.set_edgecolor("#2a2d3a")
            txt_color = "#e8eaf0" if j in (0, 5) else C_DIM
            if j == 3:   txt_color = color   # SNR value gets verdict color
            if j == 4:   txt_color = color   # verdict text
            cell.set_text_props(color=txt_color, fontfamily="monospace")
 
    finalize(fig, "snr_table.png")

# Summary of all figures

def figure_combined():
    """All 4 main plots in a 2×2 grid (convenience overview)."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
 
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    # Plot 0 with GLM-4.5-Air teacher
    ax = axes[0]
    costs_v = SAMPLE_SIZES_45 * COST_VLLM / 1000
    costs_z = SAMPLE_SIZES_45 * COST_ZAI  / 1000
    avg45 = ALL_SEEDS_45.mean(axis=0)
    std45 = ALL_SEEDS_45.std(axis=0)
    for seed in ALL_SEEDS_45:
        ax.scatter(costs_v, seed, color=C_BLUE,  alpha=0.2, s=18, zorder=2)
        ax.scatter(costs_z, seed, color=C_CORAL, alpha=0.2, s=18, zorder=2)
    ax.fill_between(costs_v, avg45-std45, avg45+std45, color=C_BLUE,  alpha=0.1)
    ax.fill_between(costs_z, avg45-std45, avg45+std45, color=C_CORAL, alpha=0.1)
    ax.plot(costs_v, avg45, "-o", color=C_BLUE,  lw=2, ms=5, label="vLLM")
    ax.plot(costs_z, avg45, "-o", color=C_CORAL, lw=2, ms=5, label="z.ai")
    ax.set_title("[Fig 1a]  GLM-4.5-Air Scaling", loc="left", fontsize=10, color="#e8eaf0")
    ax.set_xlabel("Cost ($k)"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(30, 50)
    ax.legend(fontsize=8, framealpha=0.5)

    # Plot 1 with GLM-4.6 teacher
    ax = axes[1]
    costs_z46 = SAMPLE_SIZES_46 * COST_GLM46      / 1000
    costs_v46 = SAMPLE_SIZES_46 * COST_GLM46_VLLM / 1000
    avg46 = ALL_SEEDS_46.mean(axis=0)
    std46 = ALL_SEEDS_46.std(axis=0)
    for seed in ALL_SEEDS_46:
        ax.scatter(costs_z46, seed, color=C_GOLD,   alpha=0.2, s=18, zorder=2)
        ax.scatter(costs_v46, seed, color=C_LAVENDER, alpha=0.2, s=18, zorder=2)
    ax.fill_between(costs_z46, avg46-std46, avg46+std46, color=C_GOLD,   alpha=0.12)
    ax.fill_between(costs_v46, avg46-std46, avg46+std46, color=C_LAVENDER, alpha=0.12)
    ax.plot(costs_z46, avg46, "-o", color=C_GOLD,   lw=2, ms=5, label="z.ai")
    ax.plot(costs_v46, avg46, "-o", color=C_LAVENDER, lw=2, ms=5, label="vLLM")
    ax.set_title("[Fig 1a]  GLM-4.6 Scaling", loc="left", fontsize=10, color="#e8eaf0")
    ax.set_xlabel("Cost ($k)"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(34, 55)
    ax.legend(fontsize=8, framealpha=0.5)

    # Plot 2 with truncation ablation
    ax = axes[2]
    avg_t = TRUNCATION_PERF.mean(axis=1)
    std_t = TRUNCATION_PERF.std(axis=1)
    for s in range(3):
        ax.scatter(TRUNCATION_RATIOS, TRUNCATION_PERF[:, s], color=C_GREEN, alpha=0.3, s=20)
    ax.fill_between(TRUNCATION_RATIOS, avg_t-std_t, avg_t+std_t, color=C_GREEN, alpha=0.12)
    ax.plot(TRUNCATION_RATIOS, avg_t, "-o", color=C_GREEN, lw=2, ms=6)
    ax.errorbar(TRUNCATION_RATIOS, avg_t, yerr=std_t, fmt="none",
                color=C_GREEN, alpha=0.5, capsize=3)
    ax.invert_xaxis()
    ax.set_title("[Ablation]  Context Truncation Ratio", loc="left", fontsize=10, color="#e8eaf0")
    ax.set_xlabel("Truncation Ratio"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(37, 47)

    # Plot 3 with repository specialization
    ax = axes[3]
    xs = np.arange(5)
    avg10 = SPEC_1_0.mean(axis=0);  std10 = SPEC_1_0.std(axis=0)
    avg00 = SPEC_0_0.mean(axis=0);  std00 = SPEC_0_0.std(axis=0)
    for seed in SPEC_1_0: ax.scatter(xs, seed, color=C_BLUE,  alpha=0.2, s=18)
    for seed in SPEC_0_0: ax.scatter(xs, seed, color=C_CORAL, alpha=0.2, s=18)
    ax.fill_between(xs, avg10-std10, avg10+std10, color=C_BLUE,  alpha=0.1)
    ax.fill_between(xs, avg00-std00, avg00+std00, color=C_CORAL, alpha=0.1)
    ax.plot(xs, avg10, "-o", color=C_BLUE,  lw=2, ms=6, label="α=1.0 (SERA)")
    ax.plot(xs, avg00, "-o", color=C_CORAL, lw=2, ms=6, label="α=0.0 (baseline)")
    ax.set_xticks(xs); ax.set_xticklabels([f"ckpt {i}" for i in xs])
    ax.set_title("[Fig 1b]  Repository Specialization", loc="left", fontsize=10, color="#e8eaf0")
    ax.set_xlabel("Checkpoint"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(18, 58)
    ax.legend(fontsize=8, framealpha=0.5)

    fig.suptitle("SERA — arxiv 2601.20789  ·  Figure Replication",
                 y=0.99, fontsize=13, color="#e8eaf0", fontfamily="monospace")

    finalize(fig, "sera_combined.png")
    print("  → sera_combined.png contains all 4 plots in one overview figure")

# SNR Table

def print_snr_analysis():
    def snr(g1, g2):
        diff    = abs(np.mean(g1) - np.mean(g2))
        avg_std = (np.std(g1) + np.std(g2)) / 2
        return (diff / avg_std if avg_std > 0 else np.inf), diff, avg_std
 
    def verdict(s):
        if s < 1:  return "❌ Noise dominates"
        if s < 2:  return "⚠️  Borderline"
        if s < 3:  return "✅ Decent signal"
        return             "🟢 Strong signal"
 
    print("\n" + "="*72)
    print("  SIGNAL-TO-NOISE RATIO  (SNR = |ΔMean| / avg seed σ)")
    print("="*72)
 
    comparisons = [
        ("Truncation 0.95 vs 1.00",      TRUNCATION_PERF[1], TRUNCATION_PERF[0]),
        ("Truncation 0.92 vs 1.00",      TRUNCATION_PERF[2], TRUNCATION_PERF[0]),
        ("SERA-32B 16k vs teacher",       ALL_SEEDS_45[:, 6], np.array([51.2, 51.4, 49.0])),
        ("GLM-4.6: 25k vs 16k",          ALL_SEEDS_46[:, 5], ALL_SEEDS_46[:, 4]),
        ("GLM-4.6 vs GLM-4.5-Air @16k",  ALL_SEEDS_46[:, 4], ALL_SEEDS_45[:, 6]),
        ("SERA α=1.0 vs α=0.0 (final)",  SPEC_1_0[:, 4],     SPEC_0_0[:, 4]),
    ]
 
    for name, g1, g2 in comparisons:
        s, diff, avg_s = snr(g1, g2)
        print(f"\n  {name}")
        print(f"    Δ mean = {diff:.1f}%   avg σ = ±{avg_s:.1f}%   SNR = {s:.1f}×")
        print(f"    {verdict(s)}")
    print()

# Main execution

if __name__ == "__main__":
    print("\nSERA Figure Replication — arxiv 2601.20789\n")
    print("-" * 45)

    print("\nGenerating Figures...")
    figure1a_glm45()
    figure1a_glm46()
    figure_truncation()
    figure1b_spec()
    figure_snr_table()
    figure_combined()

    print_snr_analysis()

    print("All done!")
    print("  → Check the current directory for the generated figure PNG files.")
    print("  → sera_combined.png contains all 4 plots in one overview figure")
    print("  → snr_table.png contains the signal-to-noise ratio analysis table")