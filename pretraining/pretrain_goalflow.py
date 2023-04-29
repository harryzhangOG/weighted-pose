import os

import pytorch3d
import torch.optim
import torch_geometric.loader as tgl
import wandb
from tqdm import tqdm

from part_embedding.flowtron.models.loss import TAXPoseLoss  # noqa
from part_embedding.flowtron.models.taxpose import DemoCondTAXPose  # noqa

# from part_embedding.flowtron.models.flowbotv2 import GoalFlowModel as Model
from part_embedding.flowtron.models.weighted_pose import GCGoalFlowNet as Model
from part_embedding.goal_inference.model_sg import dcp_sg_plot  # noqa
from part_embedding.losses.formnet_loss import artflownet_loss
from part_embedding.taxpose4art.train_utils import (
    create_segmenter_dataset,
    goalflow_hybrid_loss,
    goalflow_plot,
)

"""
This script pretrains the goal flow component of weighted pose on articulated objects only.
"""


def main(
    dset_name: str,
    root="/home/harry/partnet-mobility/raw",
    batch_size: int = 16,
    use_bc_loss: bool = True,
    n_epochs: int = 50,
    lr: float = 0.001,
    n_repeat: int = 50,
    embedding_dim: int = 512,
    n_proc: int = 60,
    sem_label=None,
    wandb_log=False,
    frac=1,
):
    torch.autograd.set_detect_anomaly(True)
    n_print = 1

    device = "cuda:0"

    # Set up wandb config
    dict_config = {
        "learning_rate": lr,
        "n_repeat": n_repeat,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "dataset": dset_name,
    }

    train_dset, test_dset, unseen_dset = create_segmenter_dataset(
        dset_name,
        root,
        True,
        n_repeat,
        False,
        n_proc,
        True,
        False,
        sem_label=sem_label,
        fraction=frac,
    )  # Third one is process

    train_loader = tgl.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = tgl.DataLoader(
        test_dset, batch_size=min(len(test_dset), 4), shuffle=True, num_workers=0
    )
    if unseen_dset:
        unseen_loader = tgl.DataLoader(
            unseen_dset,
            batch_size=min(len(unseen_dset), 4),
            shuffle=True,
            num_workers=0,
        )
    if wandb_log:
        wandb.init(project="pretrain", entity="harryzhangog", config=dict_config)
        run_name = wandb.run.name
        run_name_log = f"{dset_name}-{run_name}"
        wandb.run.name = run_name_log
    else:
        run_name_log = "debug"

    model = Model().to(device)

    # opt = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    d = f"part_embedding/flowtron/pretraining/checkpoints/{run_name_log}"
    os.makedirs(d, exist_ok=True)

    ## Train
    train_step = 0
    val_step = 0
    unseen_step = 0
    for i in range(1, n_epochs + 1):
        pbar = tqdm(train_loader)
        pbar_val = tqdm(test_loader)

        for (
            action_art,
            anchor_art,
            anchor_demo_art,
            action_ff,
            anchor_ff,
            anchor_demo_ff,
        ) in pbar:
            train_step += 1
            action = action_art
            anchor = anchor_art
            anchor_demo = anchor_demo_art

            action = action.to(device)
            anchor = anchor.to(device)
            anchor_demo = anchor_demo.to(device)

            opt.zero_grad()

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
            n_nodes = torch.as_tensor([d.num_nodes for d in anchor.to_data_list()]).to(
                device
            )

            # Hybrid loss training
            hybrid_loss, idx = goalflow_hybrid_loss(flow_pred, flow_gt, n_nodes)
            R_pred, t_pred = model.svd(action.pos, flow_pred, flow_gt)
            pred_pose = (
                (
                    torch.bmm(action.pos.reshape(-1, 500, 3), R_pred.transpose(-1, -2))
                    + t_pred
                )
                .reshape(-1, 3)
                .to(device)
            )

            n_nodes_action = torch.as_tensor(
                [d.num_nodes for d in action.to_data_list()]
            ).to(device)
            loss = (
                hybrid_loss
                + artflownet_loss(
                    pred_pose, action.pos + flow_gt[idx], None, n_nodes_action
                )
                + artflownet_loss(
                    pred_pose, action.pos + flow_pred[idx], None, n_nodes_action
                )
            )
            # loss = artflownet_loss(flow_pred, flow_gt, None, n_nodes)

            if wandb_log:
                wandb.log({"train_loss": loss, "train-x-axis": train_step})

            loss.backward()
            opt.step()

            if i % n_print == 0:
                desc = (
                    f"Epoch {i:03d}:  Step {train_step}  Train Loss:{loss.item():.3f}"
                )
                pbar.set_description(desc)

        if i % 10 == 0:
            torch.save(model.state_dict(), os.path.join(d, f"weights_{i:03d}.pt"))

        if wandb_log:
            flow_gt = flow_gt.reshape(-1, 2000, 3)
            flow_pred = flow_pred.reshape(-1, 2000, 3)
            flow_pred[0][500:] = 0

            task_sp = action.loc[0].item()
            obj_id = anchor.obj_id[0]
            wandb.log(
                {
                    "train_rand_plot": goalflow_plot(
                        action.pos.reshape((-1, 500, 3))[0],
                        pred_pose.reshape((-1, 500, 3))[0],
                        anchor.pos.reshape((-1, 2000, 3))[0],
                        flow_gt[0],
                        flow_pred[0],
                        task_sp,
                        obj_id,
                    )
                }
            )

        # Validation
        for (
            action_art,
            anchor_art,
            anchor_demo_art,
            action_ff,
            anchor_ff,
            anchor_demo_ff,
        ) in pbar_val:
            val_step += 1
            action = action_art
            anchor = anchor_art
            anchor_demo = anchor_demo_art

            action = action.to(device)
            anchor = anchor.to(device)
            anchor_demo = anchor_demo.to(device)
            R_gt = anchor.R_action_anchor.reshape(-1, 3, 3)
            t_gt = anchor.t_action_anchor
            mat = torch.zeros(action.num_graphs, 4, 4).to(device)
            mat[:, :3, :3] = R_gt
            mat[:, :3, 3] = t_gt
            mat[:, 3, 3] = 1
            gt_T_action = pytorch3d.transforms.Transform3d(
                device=device, matrix=mat.transpose(-1, -2)
            )
            flow_gt = anchor.flow
            flow_gt = torch.zeros_like(flow_gt).cuda().reshape(-1, 2000, 3)
            flow_gt[:, :500, :] = gt_T_action.transform_points(
                action.pos.reshape(-1, 500, 3)
            ) - action.pos.reshape(-1, 500, 3)
            flow_gt = flow_gt.reshape(-1, 3)

            with torch.no_grad():
                flow_pred = model(anchor)
                n_nodes = torch.as_tensor(
                    [d.num_nodes for d in anchor.to_data_list()]
                ).to(device)

                # Hybrid loss training
                hybrid_loss, idx = goalflow_hybrid_loss(flow_pred, flow_gt, n_nodes)
                R_pred, t_pred = model.svd(action.pos, flow_pred, flow_gt)
                pred_pose = (
                    (
                        torch.bmm(
                            action.pos.reshape(-1, 500, 3), R_pred.transpose(-1, -2)
                        )
                        + t_pred
                    )
                    .reshape(-1, 3)
                    .to(device)
                )

                n_nodes_action = torch.as_tensor(
                    [d.num_nodes for d in action.to_data_list()]
                ).to(device)
                loss = (
                    hybrid_loss
                    + artflownet_loss(
                        pred_pose, action.pos + flow_gt[idx], None, n_nodes_action
                    )
                    + artflownet_loss(
                        pred_pose, action.pos + flow_pred[idx], None, n_nodes_action
                    )
                )
                # loss = artflownet_loss(flow_pred, flow_gt, None, n_nodes)

                if wandb_log:
                    wandb.log({"val_loss": loss, "val-x-axis": val_step})

                if i % n_print == 0:
                    desc = f"Epoch {i:03d}: Step {val_step}  Val Loss:{loss.item():.3f}"
                    pbar_val.set_description(desc)
        if wandb_log:
            flow_gt = flow_gt.reshape(-1, 2000, 3)
            flow_pred = flow_pred.reshape(-1, 2000, 3)
            flow_pred[0][500:] = 0

            task_sp = action.loc[0].item()

            obj_id = anchor.obj_id[0]
            wandb.log(
                {
                    "val_rand_plot": goalflow_plot(
                        action.pos.reshape((-1, 500, 3))[0],
                        pred_pose.reshape((-1, 500, 3))[0],
                        anchor.pos.reshape((-1, 2000, 3))[0],
                        flow_gt[0],
                        flow_pred[0],
                        task_sp,
                        obj_id,
                    )
                }
            )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sem",
        type=str,
        default=None,
        help="Sem Label hinge/slider/None (both).",
    )
    parser.add_argument(
        "--cat",
        type=str,
        default="fridge",
        help="Generated dataset category name to pass in.",
    )
    parser.add_argument(
        "--num",
        type=str,
        default="100",
        help="Generated dataset nrepeat to pass in.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="if we want to wandb",
    )
    parser.add_argument(
        "--frac",
        type=str,
        default="1",
        help="Generated dataset category name to pass in.",
    )

    args = parser.parse_args()
    dset_cat = args.cat
    dset_num = args.num
    sem_label = args.sem
    wandb_log = args.wandb
    frac = args.frac

    main(
        dset_name=f"pretrain_goalflow",
        root=os.path.expanduser("~/partnet-mobility"),
        batch_size=16,
        use_bc_loss=True,
        n_epochs=75,
        lr=0.0003,
        n_repeat=1,
        embedding_dim=512,
        n_proc=60,
        sem_label=sem_label,
        wandb_log=wandb_log,
        frac=float(frac),
    )
"""
python -m part_embedding.flowtron.pretraining.pretrain_goalflow
"""
