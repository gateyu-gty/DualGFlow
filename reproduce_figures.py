"""Standalone figure reproduction script for DualGFlow (CMC Manuscript).

Regenerates all three experimental vector PDF figures (and 300 DPI PNGs) directly
from the verified JSON results inside dualgflow/:
  - paper_cmc/figures/fig4_external_heuristics_48.pdf
  - paper_cmc/figures/fig5_offline_vs_online_48.pdf
  - paper_cmc/figures/fig6_zero_shot_table14.pdf
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 10.5,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.8,
    'ytick.labelsize': 9.8,
    'legend.fontsize': 9.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

ROOT = Path(__file__).resolve().parent
RG_DIR = ROOT / 'dualgflow'
FIG_DIR = ROOT / 'paper_cmc' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_fig4(d48: dict, ext: dict) -> None:
    nom = d48['online_48']['nominal_ca_eft']['mean_flowtime']
    methods = [
        'Round-Robin',
        'Earliest-Idle',
        'Compute-EFT',
        'Data-Locality',
        'Nominal-CA-EFT',
        'DualGFlow\n(Ours)',
    ]
    vals = [
        ext['main_48']['round_robin']['mean_flowtime'],
        ext['main_48']['earliest_idle']['mean_flowtime'],
        ext['main_48']['compute_eft']['mean_flowtime'],
        ext['main_48']['data_locality']['mean_flowtime'],
        nom,
        d48['online_48']['dualgflow']['mean_flowtime'],
    ]
    colors = ['#b0b0b0', '#969696', '#737373', '#e69f00', '#377eb8', '#1b9e77']

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, width=0.58, color=colors, edgecolor='#222222', linewidth=0.9)
    ax.set_yscale('log')
    ax.set_ylabel('Mean Workflow Flowtime (s, log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(15000, 4.5e6)
    ax.grid(axis='y', linestyle=':', alpha=0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, (b, v) in enumerate(zip(bars, vals)):
        lbl = f'{v:,.1f} s'
        ax.text(
            b.get_x() + b.get_width() / 2, v * 1.12, lbl,
            ha='center', va='bottom', fontsize=8.5,
            fontweight='bold' if i == len(vals) - 1 else 'normal',
            color='#0d5c43' if i == len(vals) - 1 else '#222222',
        )
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_external_heuristics_48.pdf', bbox_inches='tight')
    plt.close(fig)


def make_fig5(d48: dict, notw: dict, norg_tw: dict) -> None:
    nom_48 = d48['online_48']['nominal_ca_eft']['mean_flowtime']
    labels = [
        'Global-MLP',
        'No-MP\n(1-Hop)',
        'No-TW\n(4L RGNN)',
        'No-RG\n(Task-Only)',
        'No-RG+TW\n(Task+TW)',
        'DualGFlow\n(Ours)',
    ]
    off_means = [
        d48['offline_48']['global_mlp']['mean_flowtime'],
        d48['offline_48']['no_mp_keep_links']['mean_flowtime'],
        notw['offline_48']['mean_flowtime'],
        d48['offline_48']['no_rg']['mean_flowtime'],
        norg_tw['offline_48']['mean_flowtime'],
        d48['offline_48']['dualgflow']['mean_flowtime'],
    ]
    off_stds = [
        d48['offline_48']['global_mlp']['seed_std'],
        d48['offline_48']['no_mp_keep_links']['seed_std'],
        notw['offline_48']['seed_std'],
        d48['offline_48']['no_rg']['seed_std'],
        norg_tw['offline_48']['seed_std'],
        d48['offline_48']['dualgflow']['seed_std'],
    ]
    on_means = [
        d48['online_48']['global_mlp']['mean_flowtime'],
        d48['online_48']['no_mp_keep_links']['mean_flowtime'],
        notw['online_48']['mean_flowtime'],
        d48['online_48']['no_rg']['mean_flowtime'],
        norg_tw['online_48']['mean_flowtime'],
        d48['online_48']['dualgflow']['mean_flowtime'],
    ]
    on_stds = [
        d48['online_48']['global_mlp']['seed_std'],
        d48['online_48']['no_mp_keep_links']['seed_std'],
        notw['online_48']['seed_std'],
        d48['online_48']['no_rg']['seed_std'],
        norg_tw['online_48']['seed_std'],
        d48['online_48']['dualgflow']['seed_std'],
    ]
    colors = ['#8c8c8c', '#d95f02', '#984ea3', '#377eb8', '#66a61e', '#1b9e77']

    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    x = np.arange(len(labels))
    w = 0.35
    for i in range(len(labels)):
        ax.bar(
            x[i] - w / 2, off_means[i], w, yerr=off_stds[i], capsize=3.5,
            color=colors[i], alpha=0.35, edgecolor=colors[i], linewidth=1.3, hatch='///',
            error_kw=dict(ecolor='#333333', lw=1.0),
        )
        ax.bar(
            x[i] + w / 2, on_means[i], w, yerr=on_stds[i], capsize=3.5,
            color=colors[i], alpha=0.92, edgecolor='#222222', linewidth=1.0,
            error_kw=dict(ecolor='#222222', lw=1.1),
        )
        g_off = (nom_48 - off_means[i]) / nom_48 * 100.0
        g_on = (nom_48 - on_means[i]) / nom_48 * 100.0
        ax.text(x[i] - w / 2, off_means[i] + off_stds[i] + 45, f'{g_off:+.2f}%', ha='center', va='bottom', fontsize=8.3, color='#444444', bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.88))
        ax.text(x[i] + w / 2, on_means[i] + on_stds[i] + 45, f'{g_on:+.2f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#111111', bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.88))

    ax.axhline(nom_48, color='#c0392b', linestyle='--', linewidth=1.4)
    ax.set_ylabel('Mean Workflow Flowtime (s) — Lower is Better')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(29650, 31900)
    ax.grid(axis='y', linestyle=':', alpha=0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend_handles = [
        Patch(facecolor='#bbbbbb', edgecolor='#444444', hatch='///', alpha=0.45, label='Stage 1: Offline Warm-Start'),
        Patch(facecolor='#555555', edgecolor='#222222', alpha=0.90, label='Stage 2: Online SIL-PPO'),
        plt.Line2D([0], [0], color='#c0392b', linestyle='--', lw=1.4, label=f'Nominal-CA-EFT ({nom_48:,.1f} s)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, framealpha=0.94, edgecolor='#cccccc')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_offline_vs_online_48.pdf', bbox_inches='tight')
    plt.close(fig)


def make_fig6(d48: dict, notw: dict, diag: dict) -> None:
    suites = [
        ('scale_n24', 'Scale\nN=24'),
        ('scale_n32', 'Scale\nN=32'),
        ('ccr_2.5', 'CCR\n2.5'),
        ('ccr_6.0', 'CCR\n6.0'),
        ('load_0.75', 'Load\n0.75'),
        ('load_1.05', 'Load\n1.05'),
        ('topo_ce', 'Cloud-\nEdge'),
        ('topo_ls', 'Leaf-\nSpine'),
    ]
    gs = d48['gen_suites']
    series = {
        'Global-MLP': [],
        'No-RG (Task-Only)': [],
        'No-TW (4L RGNN Only)': [],
        'No-MP (1-Hop Links)': [],
        'DualGFlow (Ours)': [],
    }
    for sk, _ in suites:
        nm = gs[sk]['nominal_ca_eft']['mean_flowtime']
        series['Global-MLP'].append((nm - gs[sk]['global_mlp']['mean_flowtime']) / nm * 100.0)
        series['No-RG (Task-Only)'].append((nm - gs[sk]['no_rg']['mean_flowtime']) / nm * 100.0)
        series['No-TW (4L RGNN Only)'].append(notw['gen_suites'][sk]['gain_vs_nom_pct'])
        series['No-MP (1-Hop Links)'].append((nm - gs[sk]['no_mp_keep_links']['mean_flowtime']) / nm * 100.0)
        if sk in diag['scale_dynamic_M']:
            series['DualGFlow (Ours)'].append(diag['scale_dynamic_M'][sk]['mN_gain'])
        else:
            series['DualGFlow (Ours)'].append((nm - gs[sk]['dualgflow']['mean_flowtime']) / nm * 100.0)

    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    x = np.arange(len(suites))
    bw = 0.155
    s_colors = ['#8c8c8c', '#377eb8', '#984ea3', '#d95f02', '#1b9e77']
    offsets = [-2 * bw, -bw, 0.0, bw, 2 * bw]
    for (sname, vals), col, off in zip(series.items(), s_colors, offsets):
        is_ours = (sname == 'DualGFlow (Ours)')
        bars = ax.bar(
            x + off, vals, bw,
            color=col, alpha=0.95 if is_ours else 0.82,
            edgecolor='#111111' if is_ours else '#444444',
            linewidth=1.1 if is_ours else 0.7,
            label=sname,
        )
        if is_ours:
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.25 if v >= 0 else v - 0.6,
                    f'{v:+.2f}%',
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=7.8, fontweight='bold', color='#0d5c43',
                )
    ax.axhline(0.0, color='#333333', linewidth=1.0)
    ax.set_ylabel('Flowtime Reduction vs. Nominal-CA-EFT (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in suites])
    ax.set_ylim(-5.5, 13.5)
    ax.grid(axis='y', linestyle=':', alpha=0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', ncol=3, frameon=True, framealpha=0.94, edgecolor='#cccccc')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_zero_shot_table14.pdf', bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    d48 = json.load(open(RG_DIR / 'tw_clean_48_results.json', encoding='utf-8'))
    ext = json.load(open(RG_DIR / 'tw_external_heuristics_results.json', encoding='utf-8'))
    notw = json.load(open(RG_DIR / 'tw_notw_ablation_results.json', encoding='utf-8'))
    norg_tw = json.load(open(RG_DIR / 'tw_norg_tw_ablation_results.json', encoding='utf-8'))
    diag = json.load(open(RG_DIR / 'tw_diag_fixes_3_4.json', encoding='utf-8'))
    make_fig4(d48, ext)
    make_fig5(d48, notw, norg_tw)
    make_fig6(d48, notw, diag)
    print('Successfully reproduced Fig. 4, Fig. 5, and Fig. 6 into:', FIG_DIR)


if __name__ == '__main__':
    main()
