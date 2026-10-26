import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from .gradcam import (build_stage2_model, get_bbox, load_stage1_model,
                      prepare_model_input, read_csv_auto)

RISK_LABELS = ("low", "intermediate", "high")
MUTATION_LABELS = ("KIT", "PDGFRA", "wild_type", "other")


def predict_case(row, stage1, stage2, device, margin=20):
    ct = nib.load(str(row["image"])).get_fdata().astype(np.float32)
    mask = nib.load(str(row["mask"])).get_fdata() > 0
    x_np, _, _ = prepare_model_input(ct, mask, get_bbox(mask, margin=margin))
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    with torch.inference_mode():
        gist_prob = torch.sigmoid(stage1(x).reshape(-1)[0]).item()
        outputs = stage2(x)
        risk = torch.softmax(outputs["risk"], dim=1)[0].cpu().numpy()
        mutation = torch.softmax(outputs["mutation"], dim=1)[0].cpu().numpy()
        prognosis_prob = torch.sigmoid(outputs["prognosis"].reshape(-1)[0]).item()
    result = {"id": str(row["id"]), "is_gist_probability": gist_prob,
              "prognosis_probability": prognosis_prob}
    result.update({f"risk_{name}_probability": float(value)
                   for name, value in zip(RISK_LABELS, risk)})
    result.update({f"mutation_{name}_probability": float(value)
                   for name, value in zip(MUTATION_LABELS, mutation)})
    return result


def main():
    parser = argparse.ArgumentParser(description="Run frozen GIST-Net inference")
    parser.add_argument("--csv", required=True, help="CSV with id,image,mask columns")
    parser.add_argument("--stage1-ckpt", required=True)
    parser.add_argument("--stage2-ckpt", required=True)
    parser.add_argument("--merlin-local-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--margin", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    stage1 = load_stage1_model(args.stage1_ckpt, args.device)
    stage2 = build_stage2_model(args.stage2_ckpt, args.merlin_local_dir, args.device)
    records = [predict_case(row, stage1, stage2, args.device, args.margin)
               for _, row in read_csv_auto(args.csv).iterrows()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(f"Saved {len(records)} predictions to {output}")


if __name__ == "__main__":
    main()
