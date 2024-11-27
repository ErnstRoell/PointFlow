import json
import torch
from metrics.evaluation_metrics import compute_all_metrics
import argparse


def compute_pc_metrics(sample_pcs, ref_pcs):
    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, 32, accelerated_cd=True)
    results = {
        k: (v.cpu().detach().item() if not isinstance(v, float) else v)
        for k, v in results.items()
    }
    return results


def main(args):
    results = []

    model = args.model 
    cate = args.cate 
    normalize = args.normalize 
    suffix = ""
    if normalize:
        suffix = "_normalized"

    for i in range(5):
        # Load results here.
        sample_pcs = torch.load(f"./results_gen/{model}/samples_{cate}_{i}{suffix}.pt").detach().cuda()
        ref_pcs = torch.load(f"./results_gen/{model}/ref_{cate}_{i}{suffix}.pt").detach().cuda()
        
        if args.fast_run: 
            sample_pcs = sample_pcs[:100]
            ref_pcs = ref_pcs[:100]

        result = compute_pc_metrics(sample_pcs, ref_pcs)
        result["model"] = model
        result["cate"] = cate
        result["normalized"] = normalize
        result["run"] = i
        results.append(result)

    with open(
        f"./results_gen/computed_metrics/{args.model}_{args.cate}{suffix}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f)

    print(json.dumps(results,indent=4))


if __name__ == "__main__":
    # Use nargs to specify how many arguments an option should take.
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("-n", "--normalize", action="store_true")
    parser.add_argument("-c", "--cate")
    parser.add_argument("-f", "--fast_run",action="store_true")
    args = parser.parse_args()
    main(args)
