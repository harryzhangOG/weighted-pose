import os

import numpy as np
import pytorch3d
import torch.optim
import torch_geometric.loader as tgl
from tqdm import tqdm

from part_embedding.flowtron.models.weighted_pose import GCGoalFlowNet
from part_embedding.goal_inference.dataset_v2 import CATEGORIES
from part_embedding.taxpose4art.train_utils import create_segmenter_dataset


def load_model(ckpt):
    model_path = (
        f"part_embedding/flowtron/pretraining/checkpoints/{ckpt}/weights_060.pt"
    )
    model = GCGoalFlowNet()
    model.load_state_dict(torch.load(model_path))

    return model.cuda()


def n_rot(r):
    return r / r.norm(dim=0).unsqueeze(0)


def theta_err(R_pred, R_gt):
    assert len(R_pred) == 1
    dR = n_rot(R_pred[0].squeeze()).T @ n_rot(R_gt.squeeze())
    dR = n_rot(dR)
    th = torch.arccos((torch.trace(dR) - 1) / 2.0)

    if torch.isnan(th):
        breakpoint()

    return torch.rad2deg(th).cpu().item()


def calculate_err(gt_rot, pred_rot, gt_t, pred_t, pcd):
    rot_err = theta_err(pred_rot, gt_rot)

    trans_err = torch.norm(gt_t - pred_t).cpu().item()
    Rp = pred_rot.detach().cpu().squeeze()
    Rgt = gt_rot.detach().cpu().squeeze()
    tp = pred_t.detach().cpu().squeeze()
    tgt = gt_t.detach().cpu().squeeze()
    pcd = pcd.detach().cpu()

    pos_gt = pcd @ Rgt.T + tgt
    pos_pred = pcd @ Rp.T + tp
    mse_err = np.linalg.norm(pos_gt - pos_pred, axis=1).mean()

    # import trimesh

    # scene = trimesh.Scene(
    #     [
    #         trimesh.points.PointCloud(pos_gt),
    #         trimesh.points.PointCloud(pos_pred, colors=(255, 0, 0)),
    #     ]
    # )
    # scene.show("gl")
    # breakpoint()

    return np.array([rot_err, trans_err, mse_err])


def meta_result(res_dict):
    meta_dict = {}
    for obj_id in res_dict:
        cat = CATEGORIES[obj_id]
        if cat not in meta_dict:
            meta_dict[cat] = np.mean(res_dict[obj_id], axis=0).reshape(1, 3)
        else:
            temp = meta_dict[cat]
            meta_dict[cat] = np.vstack([temp, np.mean(res_dict[obj_id], axis=0)])
    for cat in meta_dict:
        meta_dict[cat] = np.mean(meta_dict[cat], axis=0)
    a = np.zeros((1, 3))
    num = 0
    for cat in meta_dict:
        a += meta_dict[cat]
        num += 1

    meta_dict["Average"] = (a / num)[0]

    return meta_dict


def main(ckpt, dset_name, root, n_repeat=1, n_proc=10):
    model = load_model(ckpt)
    model.eval()
    train_dset, test_dset, unseen_dset = create_segmenter_dataset(
        dset_name, root, True, n_repeat, False, n_proc, True, False
    )  # Third one is process
    train_loader = tgl.DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = tgl.DataLoader(
        test_dset, batch_size=min(len(test_dset), 1), shuffle=True, num_workers=0
    )
    device = "cuda:0"
    pbar = tqdm(train_loader)
    pbar_val = tqdm(test_loader)
    res_ff_train = {}
    res_art_train = {}
    res_ff_val = {}
    res_art_val = {}
    for (
        action_art,
        anchor_art,
        anchor_demo_art,
        action_ff,
        anchor_ff,
        anchor_demo_ff,
    ) in pbar:
        action = action_art.to(device)
        anchor = anchor_art.to(device)
        anchor_demo = anchor_demo_art.to(device)

        R_gt = anchor.R_action_anchor.reshape(-1, 3, 3)
        t_gt = anchor.t_action_anchor
        mat = torch.zeros(action.num_graphs, 4, 4).to(device)
        mat[:, :3, :3] = R_gt
        mat[:, :3, 3] = t_gt
        mat[:, 3, 3] = 1
        gt_T_action = pytorch3d.transforms.Transform3d(
            device=device, matrix=mat.transpose(-1, -2)
        )
        flow_pred = model(anchor)
        flow_gt = torch.zeros_like(flow_pred).cuda().reshape(-1, 2000, 3)
        flow_gt[:, :500, :] = gt_T_action.transform_points(
            action.pos.reshape(-1, 500, 3)
        ) - action.pos.reshape(-1, 500, 3)
        flow_gt = flow_gt.reshape(-1, 3)

        R_pred, t_pred = model.svd(action.pos, flow_pred, flow_gt)

        errs = calculate_err(R_gt, R_pred, t_gt, t_pred, action.pos)
        if anchor.obj_id[0].split("_")[0] not in res_art_train:
            res_art_train[anchor.obj_id[0].split("_")[0]] = errs
        else:
            temp = res_art_train[anchor.obj_id[0].split("_")[0]]
            res_art_train[anchor.obj_id[0].split("_")[0]] = np.vstack([temp, errs])

        flow_gt = flow_gt.reshape(-1, 2000, 3)
        flow_pred = flow_pred.reshape(-1, 2000, 3)
        flow_pred[0][500:] = 0

        action = action_ff.to(device)
        anchor = anchor_ff.to(device)
        anchor_demo = anchor_demo_ff.to(device)

        R_gt = anchor.R_action_anchor.reshape(-1, 3, 3)
        t_gt = anchor.t_action_anchor
        mat = torch.zeros(action.num_graphs, 4, 4).to(device)
        mat[:, :3, :3] = R_gt
        mat[:, :3, 3] = t_gt
        mat[:, 3, 3] = 1
        gt_T_action = pytorch3d.transforms.Transform3d(
            device=device, matrix=mat.transpose(-1, -2)
        )
        flow_pred = model(anchor)
        flow_gt = torch.zeros_like(flow_pred).cuda().reshape(-1, 2000, 3)
        flow_gt[:, :500, :] = gt_T_action.transform_points(
            action.pos.reshape(-1, 500, 3)
        ) - action.pos.reshape(-1, 500, 3)
        flow_gt = flow_gt.reshape(-1, 3)

        R_pred, t_pred = model.svd(action.pos, flow_pred, flow_gt)

        errs = calculate_err(R_gt, R_pred, t_gt, t_pred, action.pos)

        if anchor.obj_id[0].split("_")[0] not in res_ff_train:
            res_ff_train[anchor.obj_id[0].split("_")[0]] = errs
        else:
            temp = res_ff_train[anchor.obj_id[0].split("_")[0]]
            res_ff_train[anchor.obj_id[0].split("_")[0]] = np.vstack([temp, errs])

    meta_ff = meta_result(res_ff_train)
    meta_art = meta_result(res_art_train)

    print("===================== TRAINING OBJECTS =====================")
    print("Free Floating:")
    for cat in meta_ff:
        print(
            f"{cat:<20}\ttheta error: {meta_ff[cat][0]:.2f} \ttrans error: {meta_ff[cat][1]:.2f} \tMSE: {meta_ff[cat][2]:.2f}"
        )
    print("Articulated:")
    for cat in meta_art:
        print(
            f"{cat:<20}\ttheta error: {meta_art[cat][0]:.2f} \ttrans error: {meta_art[cat][1]:.2f} \tMSE: {meta_art[cat][2]:.2f}"
        )

    for (
        action_art,
        anchor_art,
        anchor_demo_art,
        action_ff,
        anchor_ff,
        anchor_demo_ff,
    ) in pbar_val:
        action = action_art.to(device)
        anchor = anchor_art.to(device)
        anchor_demo = anchor_demo_art.to(device)

        R_gt = anchor.R_action_anchor.reshape(-1, 3, 3)
        t_gt = anchor.t_action_anchor
        mat = torch.zeros(action.num_graphs, 4, 4).to(device)
        mat[:, :3, :3] = R_gt
        mat[:, :3, 3] = t_gt
        mat[:, 3, 3] = 1
        gt_T_action = pytorch3d.transforms.Transform3d(
            device=device, matrix=mat.transpose(-1, -2)
        )
        flow_pred = model(anchor)
        flow_gt = torch.zeros_like(flow_pred).cuda().reshape(-1, 2000, 3)
        flow_gt[:, :500, :] = gt_T_action.transform_points(
            action.pos.reshape(-1, 500, 3)
        ) - action.pos.reshape(-1, 500, 3)
        flow_gt = flow_gt.reshape(-1, 3)

        R_pred, t_pred = model.svd(action.pos, flow_pred, flow_gt)

        errs = calculate_err(R_gt, R_pred, t_gt, t_pred, action.pos)
        if anchor.obj_id[0].split("_")[0] not in res_art_val:
            res_art_val[anchor.obj_id[0].split("_")[0]] = errs
        else:
            temp = res_art_val[anchor.obj_id[0].split("_")[0]]
            res_art_val[anchor.obj_id[0].split("_")[0]] = np.vstack([temp, errs])

        action = action_ff.to(device)
        anchor = anchor_ff.to(device)
        anchor_demo = anchor_demo_ff.to(device)

        R_gt = anchor.R_action_anchor.reshape(-1, 3, 3)
        t_gt = anchor.t_action_anchor
        mat = torch.zeros(action.num_graphs, 4, 4).to(device)
        mat[:, :3, :3] = R_gt
        mat[:, :3, 3] = t_gt
        mat[:, 3, 3] = 1
        gt_T_action = pytorch3d.transforms.Transform3d(
            device=device, matrix=mat.transpose(-1, -2)
        )
        flow_pred = model(anchor)
        flow_gt = torch.zeros_like(flow_pred).cuda().reshape(-1, 2000, 3)
        flow_gt[:, :500, :] = gt_T_action.transform_points(
            action.pos.reshape(-1, 500, 3)
        ) - action.pos.reshape(-1, 500, 3)
        flow_gt = flow_gt.reshape(-1, 3)

        R_pred, t_pred = model.svd(action.pos, flow_pred, flow_gt)

        errs = calculate_err(R_gt, R_pred, t_gt, t_pred, action.pos)
        if anchor.obj_id[0].split("_")[0] not in res_ff_val:
            res_ff_val[anchor.obj_id[0].split("_")[0]] = errs
        else:
            temp = res_ff_val[anchor.obj_id[0].split("_")[0]]
            res_ff_val[anchor.obj_id[0].split("_")[0]] = np.vstack([temp, errs])

    meta_ff = meta_result(res_ff_val)
    meta_art = meta_result(res_art_val)

    print("===================== VAL OBJECTS =====================")
    print("Free Floating:")
    for cat in meta_ff:
        print(
            f"{cat:<20}\ttheta error: {meta_ff[cat][0]:.2f} \ttrans error: {meta_ff[cat][1]:.2f} \tMSE: {meta_ff[cat][2]:.2f}"
        )
    print("Articulated:")
    for cat in meta_art:
        print(
            f"{cat:<20}\ttheta error: {meta_art[cat][0]:.2f} \ttrans error: {meta_art[cat][1]:.2f} \tMSE: {meta_art[cat][2]:.2f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        help="checkpoint dir name",
    )
    args = parser.parse_args()
    ckpt = args.ckpt

    main(ckpt, "taxpose-new", os.path.expanduser("~/partnet-mobility"))
