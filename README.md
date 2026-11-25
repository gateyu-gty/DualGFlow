# DualGFlow

Source code for DualGFlow, a scheduler for concurrent DAG workflows under multi-hop network contention.

Install the Python dependencies with `pip install -r requirements.txt`. Evaluation, training, and figure reproduction require the corresponding workflow inputs, checkpoints, or result files from the separate reproducibility package. Those artifacts are held separately from this repository.

Run `python -m dualgflow.eval_tw_clean_48 --help` for the checkpoint evaluation command and `python -m dualgflow.run_tw_clean_3seed --help` for the training command.
