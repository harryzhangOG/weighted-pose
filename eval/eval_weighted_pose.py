import os

import numpy as np
import torch.optim
import torch_geometric.loader as tgl
from tqdm import tqdm

from part_embedding.flowtron.models.weighted_pose import WeightedPose
from part_embedding.goal_inference.dataset_v2 import CATEGORIES
from part_embedding.taxpose4art.train_utils import create_segmenter_dataset


def load_model(ckpt):
    pretrained = False if ckpt else True
    model = WeightedPose(pretrained=pretrained)

    if ckpt:
        print("Running trained model")
        model_path = f"part_embedding/flowtron/checkpoints/{ckpt}/weights_060.pt"
        model.load_state_dict(torch.load(model_path))
        # state_dict = {}
        # loaded_dict = torch.load(model_path)
        # for k in loaded_dict:
        #     if "taxpose_emb" in k:
        #         state_dict[k[12:]] = loaded_dict[k]
        # model.taxpose_emb.load_state_dict(state_dict)
        # state_dict = {}
        # for k in loaded_dict:
        #     if "goal_flow" in k:
        #         state_dict[k[10:]] = loaded_dict[k]
        # model.goal_flow.load_state_dict(state_dict)
        model.taxpose_emb.eval()
        model.goal_flow.eval()
        model.seg_net.eval()
        model.eval()

    else:
        print("Running pretrained baseline")
        model.taxpose_emb.eval()
        model.goal_flow.eval()
        model.seg_net.eval()

    return model.cuda()


def n_rot(r):
    return r / r.norm(dim=0).unsqueeze(0)


def theta_err(R_pred, R_gt):
    assert len(R_pred) == 1
    dR = n_rot(R_pred[0].squeeze()).T @ n_rot(R_gt.squeeze())
    dR = n_rot(dR)
    th = torch.arccos((torch.trace(dR) - 1) / 2.0)

    if torch.isnan(th):
        return 0

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

        R_pred, t_pred, pred_T_action, Fx, Fy, goal_flow, learned_w = model(
            action, anchor
        )
        errs = calculate_err(R_gt, R_pred, t_gt, t_pred, action.pos)
        if anchor.obj_id[0].split("_")[0] not in res_art_train:
            res_art_train[anchor.obj_id[0].split("_")[0]] = errs
        else:
            temp = res_art_train[anchor.obj_id[0].split("_")[0]]
            res_art_train[anchor.obj_id[0].split("_")[0]] = np.vstack([temp, errs])

        action = action_ff.to(device)
        anchor = anchor_ff.to(device)
        anchor_demo = anchor_demo_ff.to(device)

        R_gt = anchor.R_action_anchor.reshape(-1, 3, 3)
        t_gt = anchor.t_action_anchor

        R_pred, t_pred, pred_T_action, Fx, Fy, goal_flow, learned_w = model(
            action, anchor
        )
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

        R_pred, t_pred, pred_T_action, Fx, Fy, goal_flow, learned_w = model(
            action, anchor
        )
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

        R_pred, t_pred, pred_T_action, Fx, Fy, goal_flow, learned_w = model(
            action, anchor
        )
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
    parser.add_argument("--ckpt", type=str, help="checkpoint dir name", default=None)
    args = parser.parse_args()

    # If ckpt is None, then it runs the pretrained version.
    ckpt = args.ckpt

    main(ckpt, "weighted-pose-snap", os.path.expanduser("~/partnet-mobility"))

"""
To test finetuned model:

python -m part_embedding.flowtron.eval.eval_weighted_pose --ckpt wp_new-super-flower-5

To test pretrained model:
python -m part_embedding.flowtron.eval.eval_weighted_pose
"""
