# GIST-Net model card

## Model summary

GIST-Net is a frozen two-stage 3D imaging pipeline. The first stage uses a 3D DenseNet121 with CT and tumour-mask channels for GIST classification. The second stage combines a 3D DenseNet121 representation with a frozen Merlin imaging representation through gated residual fusion and exposes risk, molecular-subtype and prognosis heads.

## Label definitions

- Risk class 0: low risk, formed by merging the original label values 0 and 1.
- Risk class 1: intermediate risk, corresponding to the original label value 2.
- Risk class 2: high risk, corresponding to the original label value 3.
- Molecular subtype: KIT, PDGFRA, wild type and other.
- Prognosis: binary recurrence within 24 months.

The probability assigned to the merged low-risk category is learned directly by the three-class model; it is not computed by adding two post-hoc output probabilities.

## Preprocessing

The tumour bounding box is expanded by a configurable margin, CT values are clipped to -200 to 300 HU and scaled to [0, 1], and the CT crop and binary mask are resized to 128 x 128 x 96 voxels. CT uses trilinear interpolation and the mask uses nearest-neighbour interpolation.

## Limitations

- Input performance depends on acquisition, contrast phase, reconstruction and mask quality.
- The model may not generalize to populations or scanners absent from development data.
- Grad-CAM is a qualitative localization aid and does not establish causal importance.
- Predicted probabilities may require calibration in a target population.
- The released software does not replace pathological, molecular or multidisciplinary assessment.

## Data and privacy

No patient images, masks, identifiers, manifests or patient-level predictions are distributed in this repository.
