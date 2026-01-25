# GENERator Atlas Fine-tuning Records

This repository is a **fine-tuning and experiment record repository** based on  
the official **GENERator** project:

https://github.com/GenerTeam/GENERator.git

The **usage, command-line interface, and overall workflow are identical to the original GENERator repository**.  
This repository mainly serves as a record and extension of fine-tuning experiments conducted on top of GENERator.

---

## 1. Base Repository

This work is built upon the official GENERator codebase:

- **GENERator GitHub**: https://github.com/GenerTeam/GENERator.git  
- **Paper**: *GENERator: A Long-Context Generative Genomic Foundation Model* (Wu et al., 2025)

All scripts in this repository follow the same execution logic as the original GENERator repository unless explicitly stated otherwise.

---

## 2. Fine-tuning Dataset

The fine-tuning dataset used in this repository is located in:

```text
dataset/
````

### Dataset source

The dataset is derived from **CRISPR-Cas Atlas**:

* **CRISPR-Cas Atlas GitHub**:
  [https://github.com/Profluent-AI/CRISPR-Cas-Atlas.git](https://github.com/Profluent-AI/CRISPR-Cas-Atlas.git)

The dataset was preprocessed and reorganized for causal language model fine-tuning on genomic sequences.

---

## 3. Backbone Model

The backbone model used for fine-tuning is:

* **GENERator-eukaryote-3b-base**
  [https://huggingface.co/GenerTeam/GENERator-eukaryote-3b-base](https://huggingface.co/GenerTeam/GENERator-eukaryote-3b-base)

Fine-tuning was performed on top of this pretrained checkpoint without changing the model architecture.

---

## 4. Code Modifications and Extensions

Compared with the original GENERator repository, this repository includes **minor but practical modifications**, mainly for engineering compatibility and experiment management:

### Key differences from the original repo

* Adjustments to **better support specific PyTorch versions**
* Additional functionality added to existing scripts, for example:

  * Fine-tuning scripts now support **saving checkpoints by training steps**
  * Additional utility scripts for generation, embedding extraction, and log processing
* Several **new scripts** are included for:

  * Atlas fine-tuning
  * Fast sanity-check experiments
  * Result analysis and visualization

These changes do **not** alter the core modeling logic of GENERator.

---

## 5. Fine-tuned Model

The fine-tuned model produced in this repository is publicly available at:

* **generator-v2-prokaryote-3b-atlas-ft**
  [https://huggingface.co/metaXu264/generator-v2-prokaryote-3b-atlas-ft](https://huggingface.co/metaXu264/generator-v2-prokaryote-3b-atlas-ft)

This model is a GENERator v2 prokaryote model fine-tuned on CRISPR-Cas Atlas–derived data.

---

## 6. Usage

Since this repository is based on the official GENERator codebase,
**all usage instructions are identical to the original repository**:

👉 Please refer to the official GENERator README for detailed usage instructions:

[https://github.com/GenerTeam/GENERator.git](https://github.com/GenerTeam/GENERator.git)

All downstream tasks (sequence recovery, variant effect prediction, generation, etc.) can be executed using the same commands.

---

## 7. Licenses

This repository follows the licenses of the original projects it is based on.

### GENERator License

Please refer to the original GENERator repository for license details:

[https://github.com/GenerTeam/GENERator.git](https://github.com/GenerTeam/GENERator.git)

### CRISPR-Cas Atlas License

Please refer to the CRISPR-Cas Atlas repository for license details:

[https://github.com/Profluent-AI/CRISPR-Cas-Atlas.git](https://github.com/Profluent-AI/CRISPR-Cas-Atlas.git)

No additional license restrictions are imposed by this repository.

---

## 8. Citation

If you use this repository or the associated models, please cite the following works.

### GENERator

```bibtex
@misc{wu2025generator,
      title={GENERator: A Long-Context Generative Genomic Foundation Model}, 
      author={Wei Wu and Qiuyi Li and Mingyang Li and Kun Fu and Fuli Feng and Jieping Ye and Hui Xiong and Zheng Wang},
      year={2025},
      eprint={2502.07272},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.07272}, 
}
```

### CRISPR-Cas Atlas / OpenCRISPR

```bibtex
@article{profluent2024opencrispr,
  title={Design of highly functional genome editors by modeling the universe of CRISPR-Cas sequences},
  author={Ruffolo, Jeffrey A and Nayfach, Stephen and Gallagher, Joseph and Bhatnagar, Aadyot and Beazer, Joel and Hussain, Riffat and Russ, Jordan and Yip, Jennifer and Hill, Emily and Pacesa, Martin and others},
  journal={bioRxiv},
  pages={2024--04},
  year={2024},
  publisher={Cold Spring Harbor Laboratory }
}
```
