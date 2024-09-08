import json
from datasets import get_test_loader, synsetid_to_cate
from args import get_args
import torch
import torch.nn as nn
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import compute_all_metrics
from models.networks import PointFlow
from models.vae import BaseModel as VAE
from models.encoder import BaseModel as Encoder
from model_wrapper import TopologicalModelVAE, TopologicalModelEncoder
from models.networks import PointFlow

from normalization import normalize


@torch.no_grad()
def evaluate_recon(model, args):
    if "all" in args.cates:
        cates = list(synsetid_to_cate.values())
    else:
        cates = args.cates
    cate_to_len = {}
    for cate in cates:
        args.cates = [cate]
        loader = get_test_loader(args)

        all_sample = []
        all_ref = []
        for data in loader:
            idx_b, tr_pc, te_pc = data["idx"], data["train_points"], data["test_points"]
            te_pc = te_pc.cuda()  # if args.gpu is None else te_pc.cuda(args.gpu)
            tr_pc = tr_pc.cuda()  # if args.gpu is None else tr_pc.cuda(args.gpu)
            B, N = te_pc.size(0), te_pc.size(1)
            out_pc = model.reconstruct(tr_pc, num_points=N)
            m, s = data["mean"].float(), data["std"].float()
            m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
            s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
            out_pc = out_pc * s + m
            te_pc = te_pc * s + m

            if args.normalize:
                te_pc, means, norms = normalize(te_pc.clone())
                out_pc -= means
                out_pc /= norms

            all_sample.append(out_pc)
            all_ref.append(te_pc)

            if args.fast_run:
                break

        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)

        cate_to_len[cate] = int(sample_pcs.size(0))

        results = EMD_CD(
            sample_pcs, ref_pcs, args.batch_size, reduced=True, accelerated_cd=True
        )
        results = {
            ("%s" % k): (v if isinstance(v, float) else v.item())
            for k, v in results.items()
        }

    return results, sample_pcs, ref_pcs


def evaluate_gen(model, args):
    loader = get_test_loader(args)
    all_sample = []
    all_ref = []
    for i, data in enumerate(loader):
        idx_b, te_pc, tr_pc = data["idx"], data["test_points"], data["train_points"]
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        B, N = te_pc.size(0), te_pc.size(1)
        _, out_pc = model.sample(B, N)

        # denormalize
        m, s = data["mean"].float(), data["std"].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        if args.normalize:
            te_pc, means, norms = normalize(te_pc)
            out_pc -= means
            out_pc /= norms

        all_sample.append(out_pc)
        all_ref.append(te_pc)


        if args.fast_run and i==2:
            break

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    # Compute metrics
    results = compute_all_metrics(
        sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True
    )
    results = {
        k: (v.cpu().detach().item() if not isinstance(v, float) else v)
        for k, v in results.items()
    }
    return results, sample_pcs, ref_pcs


def compute_pc_metrics(sample_pcs,ref_pcs):
    # Compute metrics
    results = compute_all_metrics(
        sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True
    )
    results = {
        k: (v.cpu().detach().item() if not isinstance(v, float) else v)
        for k, v in results.items()
    }
    return results, sample_pcs, ref_pcs

def main(args):
    # Load the model 
    if args.model == "Encoder": 
        print("Loading Encoder")
        encoder_model = Encoder.load_from_checkpoint(
            checkpoint_path=f"./trained_models/ectencoder_shapenet_{args.cates[0]}.ckpt"
        ).cuda()
        model = TopologicalModelEncoder(encoder_model.cuda())
    elif args.model == "VAE":
        print("Loading VAE")
        encoder_model = Encoder.load_from_checkpoint(
            checkpoint_path=f"./trained_models/ectencoder_shapenet_{args.cates[0]}.ckpt"
        ).cuda()
        vae = VAE.load_from_checkpoint(
            f"./trained_models/vae_shapenet_{args.cates[0]}.ckpt"
        ).cuda()
        model = TopologicalModelVAE(encoder_model, vae)
        model.vae.eval()
    elif args.model == "PointFlow":
        model = PointFlow(args)

        def _transform_(m):
            return nn.DataParallel(m, device_ids=[0])

        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

        print("Resume Path:%s" % args.resume_checkpoint)
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint)
        model.eval()
    else: 
        raise ValueError()


    with torch.no_grad():
        results = []
        if args.evaluate_recon:
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

            with open(f"./results/{args.model}/{args.cates[0]}{suffix}.json","w",
                encoding="utf-8",
            ) as f:
                json.dump(results, f)
            torch.save(
                sample_pc, f"./results/{args.model}/samples_{args.cates[0]}{suffix}.pt"
            )
            torch.save(ref_pc, f"./results/{args.model}/ref_{args.cates[0]}{suffix}.pt")

        else:
            if args.model == "Encoder":
                raise ValueError()

            for _ in range(args.num_reruns):
                result, sample_pc, ref_pc = evaluate_gen(model, args)
                result["model"] = args.model
                result["cate"] = args.cates[0]
                result["normalized"] = args.normalize
                
                results.append(result)
             
            suffix = ""
            if args.normalize:
                suffix = "_normalized"

            with open(f"./results_gen/{args.model}/{args.cates[0]}{suffix}.json","w",
                encoding="utf-8",
            ) as f:
                json.dump(results, f)
            torch.save(
                sample_pc, f"./results_gen/{args.model}/samples_{args.cates[0]}{suffix}.pt"
            )
            torch.save(ref_pc, f"./results_gen/{args.model}/ref_{args.cates[0]}{suffix}.pt")


if __name__ == "__main__":
    args = get_args()
    main(args)
