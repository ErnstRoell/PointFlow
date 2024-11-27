import torch
import json
from metrics.evaluation_metrics import EMD_CD

DEVICE = "cuda:0"
# SUFFIX="_normalized"
SUFFIX = ""
MODELS = ["Encoder", "EncoderScaled","VAE", "PointFlow", "SoftFlow", "ShapeGF"]
CATES = ["Airplane", "Chair", "Car"]


losses = []
# "Encoder","VAE","PointFlow",
for cat in CATES:
    for model in MODELS:
        sample_pcs = torch.load(f"./results/{model}/samples_{cat}{SUFFIX}.pt")
        ref_pcs = torch.load(f"./results/{model}/ref_{cat}{SUFFIX}.pt")

        result = EMD_CD(sample_pcs, ref_pcs, 10, reduced=False, accelerated_cd=True)
        cd_loss = result["MMD-CD"]
        emd_loss = result["MMD-EMD"]

        losses.extend(
            [
                {
                    "idx": idx,
                    "Loss_emd": e.item(),
                    "Loss_cd": c.item(),
                    "Model": model,
                    "Category": cat,
                }
                for idx, (c, e) in enumerate(zip(cd_loss, emd_loss))
            ]
        )

with open(f"losses{SUFFIX}.json", "w") as f:
    json.dump(losses, f)
