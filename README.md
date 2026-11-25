# DualGFlow: Dual-Graph Flow-Aware Reinforcement Learning with Tropical Wave Propagation for Multi-Workflow DAG Scheduling

Official reproducibility package for the manuscript submitted to *Computers, Materials & Continua (CMC)*.

---

## 1. Repository Structure

```text
DualGFlow_Release/
├── README.md                                 # Reproducibility & expert audit guide
├── requirements.txt                          # Python dependencies
├── verify_reproducibility.py                 # ~10-second live verification script (30 checkpoints + live sim + Tables 2–4)
├── reproduce_figures.py                      # One-command reproduction of Fig. 4, Fig. 5, and Fig. 6 (lossless vector PDFs)
├── env/workflow_scheduling_v3/dax/           # Real Pegasus DAX workflow XMLs (CyberShake, Montage, Inspiral, Sipht)
├── dualgflow/
│   ├── topologies.py                         # Multi-Hop Mesh, Leaf-Spine, and Cloud-Edge topology & scenario generator
│   ├── simulator_flow.py                     # Continuous-flow max-min fair sharing (progressive filling) simulator
│   ├── models_rg.py                          # DualGFlow (dynamic zero-collision M=max(16,N) Tropical Wave + 4L Edge-Attentive RGNN) & ablations
│   ├── batch_rg.py                           # Batched dual-graph collation & forward pass
│   ├── run_tw_clean_3seed.py                 # 3-seed symmetric two-stage training (DualGFlow, No-MP, Global-MLP, No-RG)
│   ├── run_notw_ablation_3seed.py            # 3-seed standalone No-TW ablation (4-Layer RGNN without Tropical Wave)
│   ├── eval_tw_clean_48.py                   # 48 unseen Multi-Hop Mesh scenarios evaluation + paired 95% bootstrap CI
│   ├── eval_external_heuristics.py           # External baselines (Round-Robin, Earliest-Idle, Compute-EFT, Data-Locality)
│   ├── diag_fixes_3_4.py                     # End-to-end timing breakdown & zero-collision M=N scale verification
│   ├── checkpoints_tw_clean/                 # All 30 trained checkpoints (5 models x 2 stages x 3 seeds: 101, 202, 303)
│   ├── checkpoints_mesh_hc/                  # Cached Stage-1 offline counterfactual dataset
│   ├── tw_clean_48_results.json              # Verified 48-scene main, ablation, and 8-suite zero-shot results
│   ├── tw_notw_ablation_results.json         # Verified 3-seed No-TW standalone ablation results
│   ├── tw_diag_fixes_3_4.json                # Verified end-to-end timing & zero-collision M=N results
│   └── tw_external_heuristics_results.json   # Verified external heuristic results
└── paper_cmc/                                # Complete CMC LaTeX manuscript, 32 verified references, and 6 lossless vector PDF figures
```

---

## 2. Environment Setup

Tested with Python 3.10+ and PyTorch 2.1+ (CPU or CUDA GPU):

```bash
pip install -r requirements.txt
```

---

## 3. One-Command Verification & Figure Reproduction

Run from the `DualGFlow_Release/` root directory:

```bash
# 1. Verify all 30 checkpoints, live simulation exact match (diff = 0.0000s), and Tables 2–4 (~10 seconds)
python verify_reproducibility.py

# 2. Reproduce all experimental vector PDF figures (Fig. 4, Fig. 5, Fig. 6 -> paper_cmc/figures/)
python reproduce_figures.py
```

---

## 4. Full Experiment Reproduction Commands

Execute from the `DualGFlow_Release/` root directory with `PYTHONPATH=.`:

```bash
# Re-evaluate all 48 unseen scenarios & external baselines using saved 3-seed checkpoints
python -m dualgflow.eval_tw_clean_48 --seeds 101 202 303 --device cuda
python -m dualgflow.eval_external_heuristics

# Re-run end-to-end timing breakdown and zero-collision M=N scale tests
python -m dualgflow.diag_fixes_3_4

# Re-train all 5 neural models from scratch (remove dualgflow/checkpoints_tw_clean/ first)
python -m dualgflow.run_tw_clean_3seed --seeds 101 202 303 --device cuda
python -m dualgflow.run_notw_ablation_3seed
```
