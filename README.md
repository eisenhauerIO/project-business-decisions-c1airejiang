
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/g5Rk6CRe)
[![Run Notebook](https://github.com/eisenhauerIO/projects-businss-decisions/actions/workflows/run-notebook.yml/badge.svg)](https://github.com/eisenhauerIO/projects-businss-decisions/actions/workflows/run-notebook.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Main Notebook**: [sera_replication.py](https://github.com/eisenhauerIO/project-business-decisions-c1airejiang/blob/main/sera_replication.py)

The python project contains an example project by [Claire Jiang](https://github.com/c1airejiang) from the 2026 iteration of the [ECON 481: Data Science Computing for Economics] course at the University of Washington. It replicates the results from the following paper:

* Shen, E., Tormoen, D., Shah, S., Farhadi, A., & Dettmers, T. (2026). [SERA: Soft-Verified Efficient Repository Agents](https://arxiv.org/pdf/2601.20789). *Allen Institute for AI, University of Washington, and Carnegie Mellon University

sera_replication.py replicates key figures and statistical analyses from the paper SERA: Soft-Verified Efficient Repository Agents. SERA proposes a pipeline for training small coding models by generating fine-tuning data from a large teacher model in order to achieve strong performance on SWE- bench Verified at a fraction of the cost of frontier models. 

This replication will focus on the empirical results reported in the paper: scaling behavior, repository specialization, truncation robustness, and the statistical reliability of these findings.

**Summary**
SERA trains SERA-32B (a student model) to fix model bugs by:
1. Using a teacher model to generate candidate patches on a target repository
2. Filtering those patches through soft verification rather than full test execution, which keeps costs low
3. Fine tuning the student model on the filtered data, with an optional specialization phase that mixes repository specific and general training data
SWE-bench Verified is used throughout this replication process; a model must produce a code patch that resolves a real GitHub issue.

**Data**
| Variable | Description |
|------|-------------|
| `ALL_SEEDS_45` | Accuracy % at each sample size - GLM - 4.5 - Air, 3 seeds |
| `ALL_SEEDS_46` | Accuracy % at each sample size - GLM - 4.6, 3 seeds |
| `TRUNCATION_PERF` | Accuracy % at each truncation ratio, 3 seeds |
| `SPEC_1_0/SPEC_0_0` | Accuracy % across checkpoints - α = 1.0 and α = 0.0, 3 seeds |
