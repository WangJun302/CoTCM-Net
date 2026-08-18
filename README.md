# CoTCM-Net

**CoTCM-Net** is the official core repository for a research study on
symptom-guided Traditional Chinese Medicine (TCM) prescription reasoning. The
paper studies how structured TCM knowledge, molecular generation, retrieval,
expert routing, and recursive reasoning can be coupled into a unified framework
for generating herb recommendations from symptom descriptions.

The central motivation of the study is that TCM prescription generation should
not be treated as a direct black-box text generation problem. Instead, the
reasoning process is decomposed into a clinically interpretable chain:

```text
Symptoms -> Therapeutic functions -> Molecules -> Herbs -> Prescriptions
```

To support this chain, the study constructs multi-level knowledge graphs,
fine-tunes a function-to-molecule generation model, grounds generated molecules
back to herb entities, and evaluates the resulting herb recommendations under
internal benchmark settings.

This repository contains the core resources used to construct the knowledge
graphs, fine-tune the molecular generation model, and build the internal
50-case symptom benchmark for herb recommendation analysis.

## Overview

CoTCM-Net connects four reasoning levels:

1. **Symptom-to-function reasoning**: maps symptom descriptions to therapeutic
   functions.
2. **Function-to-molecule generation**: learns the association between
   therapeutic functions and molecular SMILES.
3. **Molecule-to-herb grounding**: maps generated or retrieved molecules back
   to herb entities.
4. **Herb-to-prescription retrieval**: links recommended herbs to prescription
   evidence from the constructed knowledge graph.

The model code implements a T5-based generation framework with Mixture-of-
Experts (MoE) and Mixture-of-Recursion (MoR) components. The included data files
support graph construction, fine-tuning, and internal benchmark evaluation.

## Repository Structure

```text
CoTCM/
data/
  fine_tuning/
    train_data_separated.jsonl
    dev_data_separated.jsonl
    test_data_separated.jsonl
  kg_sources/
    chufang.xlsx
    functiontodisease.xlsx
    function_texts.json
    herb_compound_texts.json
    smiles_list.json
    smiles2herb.json
  kg/
    degree_counts.csv
    unique_herbs.csv
  standard_set_50/
    standard_50_symptom_eval.jsonl
    standard_50_symptom_provenance.csv
    standard_50_symptom_set.json
    standard_50_symptom_set.xlsx
    standard_50_symptom_set_flat.csv
    standard_50_symptom_set_summary.csv
src/
  models/
    cotcm_moe_mor_t5.py
    cotcm_moe_t5_legacy.py
  train_cotcm_finetune.py
requirements.txt
```

## Data

### Fine-tuning Data

`data/fine_tuning/` contains the supervised function-to-molecule generation
split. Each JSONL row has the following format:

```json
{"input": "therapeutic function text", "target": "canonical SMILES"}
```

The current split contains:

| Split | File | Number of samples |
|---|---:|---:|
| Training | `train_data_separated.jsonl` | 4,196 |
| Development | `dev_data_separated.jsonl` | 780 |
| Test | `test_data_separated.jsonl` | 700 |

### Knowledge-Graph Sources

`data/kg_sources/` provides the core source files used to construct the
multi-level knowledge graphs:

- `functiontodisease.xlsx`: symptom/function-related source table.
- `chufang.xlsx`: prescription and herb source table.
- `function_texts.json`: normalized therapeutic-function descriptions.
- `herb_compound_texts.json`: herb-compound textual resources.
- `smiles_list.json`: molecular SMILES candidates.
- `smiles2herb.json`: molecule-to-herb mapping dictionary.

### Graph Summaries

`data/kg/` contains compact graph-level summaries, including the canonical herb
list and degree statistics used for auditing the graph structure.

### Internal 50-Case Benchmark

`data/standard_set_50/` contains the internal symptom benchmark used for
symptom-to-herb recommendation evaluation and closed-loop case analysis. This
benchmark is derived from the project knowledge resources and should be treated
as an **internal benchmark**, not as independent external clinical validation.

## Model Code

- `src/models/cotcm_moe_mor_t5.py`: main CoTCM-Net implementation with MoE and
  MoR components.
- `src/models/cotcm_moe_t5_legacy.py`: legacy T5-MoE variant retained for
  ablation and comparison.
- `src/train_cotcm_finetune.py`: fine-tuning entry point for the
  function-to-molecule generation task.

## Installation

Create a clean Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The repository does not pin a CUDA-specific PyTorch build. Please install the
PyTorch version that matches your local CUDA/GPU environment if GPU training is
required.

## Fine-tuning

Run the main fine-tuning script from the repository root:

```bash
python src/train_cotcm_finetune.py \
  --pth_train data/fine_tuning/train_data_separated.jsonl \
  --pth_dev data/fine_tuning/dev_data_separated.jsonl \
  --pth_test data/fine_tuning/test_data_separated.jsonl \
  --save_dir checkpoints \
  --batch_size 1 \
  --epoch 100 \
  --seed 1111
```


## Citation

If you use this repository, please cite the associated CoTCM-Net manuscript.

