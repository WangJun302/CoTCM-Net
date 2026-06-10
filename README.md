# CoTCM-Net
Traditional Chinese Medicine (TCM) prescriptions are grounded in a diagnostic reasoning chain that connects symptoms to prescriptions through layered associations among therapeutic functions, molecular representations, herbal ingredients, and prescription compatibility. However, this implicit reasoning process lacks a clear and verifiable generation pathway and has not been effectively formalized within modern intelligent healthcare systems. To address this challenge, we propose a knowledge-enhanced multimodal reasoning framework termed Chain-of-TCM Reasoning (CTR), aiming to establish an explainable pathway from symptom semantics to prescription generation. CTR formalizes the diagnostic logic as a hierarchical sequence--Symptom -> Function -> Molecule -> Herb -> Prescription--and constructs three heterogeneous knowledge graphs (Symptom--Function, Function--Molecule--Herb, Herb--Prescription) to support cross-modal semantic, structural, and pharmacological alignment. We further design the CoTCM-Net (Chain-of-TCM Reasoning Network), where RAG anchors sparse and heterogeneous therapeutic-function descriptions in local TCM knowledge, MoE routes different syndrome and efficacy patterns to specialized functional experts, and MoR performs recursive refinement for symptom-function mappings that cannot be resolved in a single pass. This domain-specific coordination constrains generation by the TCM reasoning chain and links generated molecules back to herbs and prescriptions through the knowledge graph, yielding a traceable route for evaluating pharmacological consistency and prescription rationality.
<img width="1472" height="607" alt="5194951d-b994-4db2-8ee6-156b2219245c" src="https://github.com/user-attachments/assets/5c385904-b837-4ab9-ac26-add08f1b5739" />
Explicit TCM reasoning chain from symptoms to prescription candidates.} The figure illustrates the conceptual hierarchy underlying the proposed framework, in which symptom inputs are progressively mapped to therapeutic functions, grounded into candidate molecular representations (SMILES), back-mapped to herbs, and finally aggregated into prescription candidates. The Mahuang Tang pathway is shown as the primary route, while Xiao Qing Long Tang is included as a related prescription candidate, highlighting the interpretable symptom--function--molecule--herb--prescription chain formalized in this study.

## Requirement
Main packages:
python==3.9.0;
torch==1.13.0;
transformers==4.28.0;
pandas==1.5.2;
numpy==1.23.5;
nltk==3.7

