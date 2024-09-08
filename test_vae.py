import os
import torch
import numpy as np
from datasets import get_datasets, synsetid_to_cate, get_test_loader
from args import get_args
from pprint import pprint
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
from models.vae import BaseModel as VAE
from models.encoder import BaseModel as Encoder
from test import evaluate_recon


################################################################################
# Import model and additional stuff
################################################################################

from model_wrapper import TopologicalModelVAEScaled, TopologicalModelEncoderScaled

################################################################################
################################################################################


def evaluate_gen(model, args):
    print("=====GEN======")
    loader = get_test_loader(args)
    all_sample = []
    all_ref = []
    for data in loader:
        idx_b, te_pc, tr_pc = data["idx"], data["test_points"], data["train_points"]
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)

        B, N = te_pc.size(0), te_pc.size(1)
        _, out_pc = model.sample(B, N)

        # # denormalize
        m, s = data["mean"].float(), data["std"].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        # ########################
        # ## Insert
        # ########################

        # out_pc = normalize(out_pc)
        # te_pc = normalize(te_pc)

        # ########################
        # ## End insert
        # ########################

        all_sample.append(out_pc)
        all_ref.append(te_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    # print("====== Sample Stats ======")
    # print(sample_pcs.norm(dim=-1).max(dim=-1)[0])
    # print(sample_pcs.mean(axis=1))
    #
    # print("====== Ref Stats ======")
    # print(ref_pcs.norm(dim=-1).max(dim=-1)[0])
    # print(ref_pcs.mean(axis=1))

    # print("REF", ref_pcs.shape)
    # r_mean = ref_pcs[0].mean(axis=1)
    # r_max = ref_pcs[0].mean(axis=1)
    # print(ref_norm.max(axis=-1)[0])

    # sample_pcs = torch.cat(all_sample, dim=0)
    # ref_pcs = torch.cat(all_ref, dim=0)
    # print(sample_pcs.shape)
    # samples_norm = sample_pcs.norm(dim=-1)
    # ref_norm = ref_pcs.norm(dim=-1)
    # print(ref_norm.shape)
    # print(samples_norm.shape)
    # print(ref_norm.max(axis=-1)[0])
    # print(samples_norm.max(axis=-1)[0])

    print(
        "Generation sample size:%s reference size: %s"
        % (sample_pcs.size(), ref_pcs.size())
    )

    # Save the generative output
    save_dir = os.path.dirname(args.resume_checkpoint)
    print(save_dir)
    np.save(
        os.path.join(save_dir, "model_out_smp.npy"), sample_pcs.cpu().detach().numpy()
    )
    np.save(os.path.join(save_dir, "model_out_ref.npy"), ref_pcs.cpu().detach().numpy())

    print("======COMPUTING METRICS=======")
    # Compute metrics
    results = compute_all_metrics(
        sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True
    )
    results = {
        k: (v.cpu().detach().item() if not isinstance(v, float) else v)
        for k, v in results.items()
    }
    pprint(results)

    print("======COMPUTING JSD=======")
    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    print("JSD:%s" % jsd)


def main(args):

    # Load the model 
    if args.model == "encoder": 
        print("Loading Encoder")
        encoder_model = Encoder.load_from_checkpoint(
            checkpoint_path=f"./trained_models/ectencoder_shapenet_{args.cates[0]}.ckpt"
        ).cuda()
        model = TopologicalModelEncoderScaled(encoder_model.cuda())
    elif args.model == "vae":
        print("Loading VAE")
        encoder_model = Encoder.load_from_checkpoint(
            checkpoint_path=f"./trained_models/ectencoder_shapenet_{args.cates[0]}.ckpt"
        ).cuda()
        vae = VAE.load_from_checkpoint(
            f"./trained_models/vae_shapenet_{args.cates[0]}.ckpt"
        ).cuda()
        model = TopologicalModelVAEScaled(encoder_model, vae)
        model.vae.eval()
    else: 
        raise ValueError()


    with torch.no_grad():
        if args.evaluate_recon:
            results = []
            # Evaluate reconstruction
            for _ in range(args.num_reruns):
                # Evaluate reconstruction
                result, sample_pc, ref_pc = evaluate_recon(model, args)
                result["model"] = args.model
                result["cate"] = args.cates[0]
                result["normalized"] = args.normalize

                results.append(result)

            suffix = ""
            if args.normalize:
                suffix = "_normalized"

            with open(
                f"./results_pointflow/{args.model}_{args.cates[0]}{suffix}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(results, f)
            torch.save(
                sample_pc, f"./results_pointflow/sample_{args.model}_{args.cates[0]}{suffix}.pt"
            )
            torch.save(ref_pc, f"./results_pointflow/ref_{args.model}_{args.cates[0]}{suffix}.pt")

        else:
            # Evaluate generation
            evaluate_gen(model, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
