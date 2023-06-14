# Training-Free Composition of Parameter-Efficient Modules with Arithmetic Operation

This repository contains code for reproducing training-free composition of parameter-efficient modules (PEMs) with addition, negation and multicombination. They are based on [adapter-transformers](https://github.com/adapter-hub/adapter-transformers).

## Installation


`adapter-transformers` currently supports **Python 3.7+** and **PyTorch 1.3.1+**.
After downloading and unzipping, you can install by:

```
cd supplementary_material
cd code
pip install .
```

## Scripts

Tuning and merging shell scripts for main tasks are listed in exps: composition for distribution generalization, composition for multitasking, compostion for unlearning and composition for domain transfer. Composition operation are realised in `merge.py`, `analogy.py` and `negation.py`(applied in `exps\composition_for_unlearning\gpt2_scale.py`).

```
.
└── exps
├── composition_for_distribution_generalization
│   ├── dataset_split_merge.sh
│   ├── fft_run_glue.sh
│   ├── ftdataset_split_merge.sh
│   ├── ia3_run_glue.sh
│   ├── lora_run_glue.sh
│   └── split_data.sh
├── composition_for_domain_transfer
│   ├── fft_polarity_classify.sh
│   ├── fft_polarity_lm.sh
│   ├── ia3_polarity_classify.sh
│   ├── ia3_polarity_lm.sh
│   ├── lora_polarity_classify.sh
│   ├── lora_polarity_lm.sh
│   └── vary_merge_analogy.sh
├── composition_for_multitasking
│   ├── fft_prompt_run_glue.sh
│   ├── ia3_prompt_run_glue.sh
│   ├── lora_prompt_run_glue.sh
│   └── vary_merge_prompt_run_glue.sh
├── composition_for_unlearning
│   ├── composition_for_unlearning\fft.sh
│   ├── gpt2_scale.py
│   ├── README.md
│   ├── requirements.txt
│   ├── run_clm_noconcat.py
│   ├── run_prediction.py
│   ├── runscore.sh
│   └── trainadapter.sh
└── run_glue.sh
```

## Instruction Datasets
The instruction pair with toxic civil comment dataset we created via ChatGPT is in `openai_generate_datasets`, together with toxic and non-toxic instructions for evaluation.