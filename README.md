# CoTCM-Net Core Data Package

This repository provides the compact public package for **CoTCM-Net**, a
traditional Chinese medicine prescription-generation framework that connects
symptoms, therapeutic functions, molecular compounds, herbs, and prescriptions.


## What Is Included

```text
CoTCM-Net-github-core/
  data/
    kg_sources/
      chufang.xlsx
      functiontodisease.xlsx
      function_texts.json
      herb_compound_texts.json
      smiles_list.json
      smiles2herb.json
    kg/
      degree_counts.csv
      unique_195_herbs.csv
    molecular_text/
      PCtrain.txt
      PCdev.txt
      PCtest.txt
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
```

## Knowledge-Graph Data

The released data support the construction of the following knowledge links:

| Layer | Main relation | Source files |
|---|---|---|
| Symptom-Function | symptom or disease expression to therapeutic function | `data/kg_sources/functiontodisease.xlsx`, `data/kg_sources/chufang.xlsx` |
| Function-Molecule-Herb | therapeutic function to molecular compounds and herbs | `data/kg_sources/chufang.xlsx`, `data/kg_sources/smiles_list.json`, `data/kg_sources/smiles2herb.json` |
| Herb-Prescription | herbs contained in prescription records | `data/kg_sources/chufang.xlsx` |

The primary source table is `data/kg_sources/chufang.xlsx`. It contains 1,288
rows and links prescription names, herbs, molecular compounds, functions, and
symptom evidence. The file `data/kg_sources/functiontodisease.xlsx` contains
1,287 function-symptom records.

The canonical herb list is provided in `data/kg/unique_195_herbs.csv`. In this
released version, the table contains 191 rows after canonicalization and
deduplication. The filename is kept for consistency with the manuscript
workflow.

## Internal 50 Symptom-Set Benchmark

The folder `data/standard_set_50/` contains the internal 50-case symptom-set
benchmark used for symptom-to-herb evaluation and closed-loop case analysis.

Important files:

| File | Description |
|---|---|
| `standard_50_symptom_eval.jsonl` | Compact evaluation file. Each row contains a case id, symptom input, target herbs, and reference prescription. |
| `standard_50_symptom_set.json` | Complete evidence file with symptoms, herbs, molecules, functions, and prescription links. |
| `standard_50_symptom_set_flat.csv` | Flattened case-herb-molecule table for inspection. |
| `standard_50_symptom_set_summary.csv` | Case-level summary table. |
| `standard_50_symptom_set.xlsx` | Spreadsheet version for manual checking. |

This benchmark should be interpreted as an **internal benchmark**, not as
independent clinical external validation, because it is constructed from the
same knowledge-resource ecosystem as the training and graph resources.

## Model Architecture Files

Two model-definition files are included:

| File | Description |
|---|---|
| `src/models/cotcm_moe_mor_t5.py` | Main CoTCM-Net architecture with T5-style sequence generation, MoE expert routing, and MoR recursive refinement. |
| `src/models/cotcm_moe_t5_legacy.py` | Earlier MoE-T5 version retained for comparison and ablation reference. |

The main model uses a T5-style encoder-decoder backbone, a mixture-of-experts
encoder, top-k expert routing, and a recursion-depth controller for iterative
refinement.
