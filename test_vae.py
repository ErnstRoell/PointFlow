from datasets import get_datasets, synsetid_to_cate
from args import get_args
from pprint import pprint
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
from models.networks import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn

################################################################################
# Import model and additional stuff
################################################################################

from omegaconf import OmegaConf
from model_wrapper import TopologicalModelVAE
from load_models import load_encoder, load_vae
from normalization import normalize

################################################################################
################################################################################


def get_test_loader(args):
    _, te_dataset = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        te_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def evaluate_recon(model, args):
    # TODO: make this memory efficient
    if "all" in args.cates:
        cates = list(synsetid_to_cate.values())
    else:
        cates = args.cates
    all_results = {}
    cate_to_len = {}
    save_dir = os.path.dirname(args.resume_checkpoint)
    for cate in cates:
        args.cates = [cate]
        loader = get_test_loader(args)

        all_sample = []
        all_ref = []
        for data in loader:
            idx_b, tr_pc, te_pc = data["idx"], data["train_points"], data["test_points"]

            te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
            tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)
            B, N = te_pc.size(0), te_pc.size(1)
            out_pc = model.reconstruct(tr_pc, num_points=N)

            m, s = data["mean"].float(), data["std"].float()
            m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
            s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
            out_pc = out_pc * s + m
            te_pc = te_pc * s + m


            ########################
            ## Insert
            ########################

            out_pc = normalize(out_pc)
            te_pc = normalize(te_pc)

            ########################
            ## End insert
            ########################

            all_sample.append(out_pc)
            all_ref.append(te_pc)

        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)
        cate_to_len[cate] = int(sample_pcs.size(0))

        print(
            "Cate=%s Total Sample size:%s Ref size: %s"
            % (cate, sample_pcs.size(), ref_pcs.size())
        )

        # Save it
        np.save(
            os.path.join(save_dir, "%s_out_smp.npy" % cate),
            sample_pcs.cpu().detach().numpy(),
        )
        np.save(
            os.path.join(save_dir, "%s_out_ref.npy" % cate),
            ref_pcs.cpu().detach().numpy(),
        )

        results = EMD_CD(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
        results = {
            k: (v.cpu().detach().item() if not isinstance(v, float) else v)
            for k, v in results.items()
        }
        pprint(results)
        all_results[cate] = results

    # Save final results
    print("=" * 80)
    print("All category results:")
    print("=" * 80)
    pprint(all_results)
    save_path = os.path.join(save_dir, "percate_results.npy")
    np.save(save_path, all_results)

    # Compute weighted performance
    ttl_r, ttl_cnt = defaultdict(lambda: 0.0), defaultdict(lambda: 0.0)
    for catename, l in cate_to_len.items():
        for k, v in all_results[catename].items():
            ttl_r[k] += v * float(l)
            ttl_cnt[k] += float(l)
    ttl_res = {k: (float(ttl_r[k]) / float(ttl_cnt[k])) for k in ttl_r.keys()}
    print("=" * 80)
    print("Averaged results:")
    pprint(ttl_res)
    print("=" * 80)

    save_path = os.path.join(save_dir, "results.npy")
    np.save(save_path, all_results)


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

        ########################
        ## Insert
        ########################

        out_pc = normalize(out_pc)
        te_pc = normalize(te_pc)

        ########################
        ## End insert
        ########################

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

    # model = PointFlow(args)

    # def _transform_(m):
    #     return nn.DataParallel(m)

    # model = model.cuda()
    # model.multi_gpu_wrapper(_transform_)

    # print("Resume Path:%s" % args.resume_checkpoint)
    # checkpoint = torch.load(args.resume_checkpoint)
    # model.load_state_dict(checkpoint)
    # model.eval()

    # Instead of PointFlow we load our own modelwrapper, that
    # handles the function signatures of input and output.
    ect_config = OmegaConf.load(
        f"./configs/config_encoder_shapenet_{args.cates[0]}.yaml"
    )
    vae_config = OmegaConf.load(f"./configs/config_vae_shapenet_{args.cates[0]}.yaml")

    encoder_model = load_encoder(ect_config)
    vae = load_vae(vae_config)
    model = TopologicalModelVAE(encoder_model, vae)
    # model.vae.eval()

    with torch.no_grad():
        if args.evaluate_recon:
            # Evaluate reconstruction
            evaluate_recon(model, args)
        else:
            # Evaluate generation
            evaluate_gen(model, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
