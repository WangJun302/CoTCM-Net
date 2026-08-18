# CoTCM-Net Core Repository

This repository provides the compact public package for **CoTCM-Net**, a
Traditional Chinese Medicine framework for symptom-guided function reasoning,
molecular generation, herb grounding, and prescription retrieval.

## Contents

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

## Data Overview

### Fine-tuning data
`data/fine_tuning/` contains the main supervised molecular-generation split.
Each JSONL row maps a therapeutic-function description to a canonical SMILES
string.

### Knowledge-graph sources
`data/kg_sources/` contains the raw files used to construct the three TCM
knowledge graphs:

- symptom-function
- function-molecule-herb
- herb-prescription

### Graph summaries
`data/kg/` contains the canonical herb list and graph-level statistics.

### Internal 50-case benchmark
`data/standard_set_50/` contains the internal symptom benchmark used for the
herb-ranking evaluation and closed-loop case analysis. It is an internal
benchmark, not external clinical validation.



## Code

- `src/models/cotcm_moe_mor_t5.py`: main CoTCM-Net model
- `src/models/cotcm_moe_t5_legacy.py`: legacy ablation model
- `src/train_cotcm_finetune.py`: training entry for the MoE-MoR-T5 fine-tuning

## Installation

```bash
pip install -r requirements.txt
```
