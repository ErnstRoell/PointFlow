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


def evaluate_gen(model, args):
    loader = get_test_loader(args)
    all_sample = []
    pvd_pcs = torch.load("./pvd/samples.pth").cuda()
    all_ref = []
    idx_start = 0 
    for data in loader:
        idx_b, te_pc, tr_pc = data["idx"], data["test_points"], data["train_points"]
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        B, N = te_pc.size(0), te_pc.size(1)

        # Loading the smaples from pvd
        out_pc = pvd_pcs[idx_start,idx_start+B] 
        idx_start += B


        # denormalize
        m, s = data["mean"].float(), data["std"].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        # out_pc -= out_pc.mean(dim=-2,keepdim=True)
        # out_pc /= out_pc.norm(dim=-1,keepdim=True)
        #
        # te_pc -= te_pc.mean(dim=-2,keepdim=True)
        # te_pc /= te_pc.norm(dim=-1,keepdim=True)

        all_sample.append(out_pc)
        all_ref.append(te_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print(
        "Generation sample size:%s reference size: %s"
        % (sample_pcs.size(), ref_pcs.size())
    )

    # Save the generative output
    save_dir = os.path.dirname(args.resume_checkpoint)
    np.save(
        os.path.join(save_dir, "model_out_smp.npy"), sample_pcs.cpu().detach().numpy()
    )
    np.save(os.path.join(save_dir, "model_out_ref.npy"), ref_pcs.cpu().detach().numpy())

    # Compute metrics
    results = compute_all_metrics(
        sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True
    )
    results = {
        k: (v.cpu().detach().item() if not isinstance(v, float) else v)
        for k, v in results.items()
    }
    pprint(results)

    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    print("JSD:%s" % jsd)


def main(args):
    model = PointFlow(args)

    def _transform_(m):
        return nn.DataParallel(m, device_ids=[0])

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        if args.evaluate_recon:
            # Evaluate reconstruction
            res = [evaluate_recon(model, args) for _ in range(10)]
            res_cd = torch.stack([r[args.cates[0]]["MMD-CD"] for r in res])
            res_emd = torch.stack([r[args.cates[0]]["MMD-EMD"] for r in res])
            res_cd_mean = res_cd.mean()
            res_cd_std = res_cd.std()

            res_emd_mean = res_emd.mean()
            res_emd_std = res_emd.std()

            print("===========RESULTS=============")
            print("MMD-CD-Mean",res_cd_mean.item())
            print("MMD-CD-STD",res_cd_std.item())
            print("MMD-EMD-Mean",res_emd_mean.item())
            print("MMD-EMD-STD",res_emd_std.item())
            print("===============================")
            torch.save(res,"res_pointflow")

        else:
            # Evaluate generation
            evaluate_gen(model, args)


if __name__ == "__main__":
    args = get_args()
    main(args)
