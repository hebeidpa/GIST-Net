# GIST-Net

GIST-Net is a frozen research inference pipeline for contrast-enhanced CT and a tumour mask. It provides four outputs: GIST probability, three-level risk probabilities, molecular-subtype probabilities, and recurrence probability. It can also generate task-specific Grad-CAM maps in axial, coronal and sagittal views.

This public repository is inference-only. It contains one frozen model release and does not include model fitting, parameter selection, or patient-level research data.

## Outputs

- GIST versus non-GIST: one probability.
- Risk: low, intermediate and high probabilities. The low-risk output represents the combined original categories 0 and 1.
- Molecular subtype: KIT, PDGFRA, wild type and other probabilities.
- Prognosis: recurrence-within-24-months probability.

## Installation

```bash
git clone https://github.com/hebeidpa/GIST-Net.git
cd GIST-Net
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the Merlin medical foundation-model package according to its upstream instructions and place its locally acquired checkpoint in a directory accessible to the inference process. GIST-Net never downloads it at runtime.

Download `gist_net_stage1.pt` and `gist_net_stage2.pt` from the latest GitHub Release. Verify them with `weights/SHA256SUMS`.

## Input

Prepare a CSV with `id`, `image`, and `mask` columns. Image and mask must be aligned 3D NIfTI files; the mask must contain at least one non-zero voxel. See `examples/cases.template.csv`.

## Prediction

```bash
gist-net-infer \
  --csv examples/cases.csv \
  --stage1-ckpt weights/gist_net_stage1.pt \
  --stage2-ckpt weights/gist_net_stage2.pt \
  --merlin-local-dir /path/to/Merlin \
  --output outputs/predictions.csv
```

## Grad-CAM

```bash
gist-net-gradcam \
  --csv examples/cases.csv \
  --case_id example_case \
  --stage1_ckpt weights/gist_net_stage1.pt \
  --stage2_ckpt weights/gist_net_stage2.pt \
  --merlin_local_dir /path/to/Merlin \
  --out_dir outputs/example_case
```

The Grad-CAM command saves separate base and overlay PNG files for all three anatomical planes, task prediction CSV files, NIfTI heatmaps and optional 3D renderings.

## Intended use

GIST-Net is provided for research and reproducibility only. It is not a medical device and must not be used as the sole basis for diagnosis, treatment, surveillance or patient counselling. Independent validation is required before use in a new institution or acquisition setting.

See [MODEL_CARD.md](MODEL_CARD.md) for limitations and label definitions.
