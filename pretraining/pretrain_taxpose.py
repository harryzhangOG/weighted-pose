import os

import pytorch3d
import torch.optim
import torch_geometric.loader as tgl
import wandb
from tqdm import tqdm

from part_embedding.flowtron.models.loss import TAXPoseLoss
from part_embedding.flowtron.models.taxpose import DemoCondTAXPose
from part_embedding.goal_inference.model_sg import dcp_sg_plot
from part_embedding.taxpose4art.train_utils import create_segmenter_dataset

"""
This script pretrains the TAX-Pose component of weighted pose on FF objects only.
"""


def main(
    dset_name: str,
    root="/home/harry/partnet-mobility/raw",
    batch_size: int = 16,
    n_epochs: int = 50,
    lr: float = 0.001,
    n_repeat: int = 50,
    n_proc: int = 60,
    sem_label=None,
    wandb_log=False,
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
        run_name_log = f"pretrain-taxpose-{run_name}"
        wandb.run.name = run_name_log
    else:
        run_name_log = "debug"

    model = DemoCondTAXPose().to(device)

    # opt = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    crit = TAXPoseLoss()

    d = f"part_embedding/flowtron/pretraining/checkpoints/{run_name_log}"
    os.makedirs(d, exist_ok=True)

    # Train
    train_step = 0
    val_step = 0
    for i in range(1, n_epochs + 1):
        pbar = tqdm(train_loader)
        pbar_val = tqdm(test_loader)
        if unseen_dset:
            pbar_unseen = tqdm(unseen_loader)
        for (
            action_art,
            anchor_art,
            anchor_demo_art,
            action_ff,
            anchor_ff,
            anchor_demo_ff,
        ) in pbar:
            train_step += 1
            action = action_ff
            anchor = anchor_ff
            anchor_demo = anchor_demo_ff
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

            R_pred, t_pred, pred_T_action, Fx, Fy = model(action, anchor, anchor_demo)

            gt_T_action = pytorch3d.transforms.Transform3d(
                device=device, matrix=mat.transpose(-1, -2)
            )

            loss = crit(
                action.pos.reshape(action.num_graphs, -1, 3),
                pred_T_action,
                gt_T_action,
                Fx,
            )

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
            wandb.log(
                {
                    "train_rand_plot": dcp_sg_plot(
                        action.pos.reshape((-1, 500, 3))[0],
                        anchor.pos.reshape((anchor.num_graphs, -1, 3))[0],
                        t_gt[0],
                        t_pred[0],
                        R_gt[0],
                        R_pred[0],
                        None,
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
            action = action_ff
            anchor = anchor_ff
            anchor_demo = anchor_demo_ff
            action = action.to(device)
            anchor = anchor.to(device)
            anchor_demo = anchor_demo.to(device)

            R_gt = anchor.R_action_anchor.reshape(-1, 3, 3)
            t_gt = anchor.t_action_anchor
            mat = torch.zeros(action.num_graphs, 4, 4).to(device)
            mat[:, :3, :3] = R_gt
            mat[:, :3, 3] = t_gt
            mat[:, 3, 3] = 1

            with torch.no_grad():
                R_pred, t_pred, pred_T_action, Fx, Fy = model(
                    action, anchor, anchor_demo
                )

                gt_T_action = pytorch3d.transforms.Transform3d(
                    device=device, matrix=mat.transpose(-1, -2)
                )

                loss = crit(
                    action.pos.reshape(action.num_graphs, -1, 3),
                    pred_T_action,
                    gt_T_action,
                    Fx,
                )
                if wandb_log:
                    wandb.log({"val_loss": loss, "val-x-axis": val_step})

                if i % n_print == 0:
                    desc = f"Epoch {i:03d}: Step {val_step}  Val Loss:{loss.item():.3f}"
                    pbar_val.set_description(desc)
        if wandb_log:
            wandb.log(
                {
                    "val_rand_plot": dcp_sg_plot(
                        action.pos.reshape((-1, 500, 3))[0],
                        anchor.pos.reshape((anchor.num_graphs, -1, 3))[0],
                        t_gt[0],
                        t_pred[0],
                        R_gt[0],
                        R_pred[0],
                        None,
                    )
                }
            )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="if we want to wandb",
    )

    args = parser.parse_args()

    wandb_log = args.wandb

    main(
        dset_name=f"pretrain_taxpose",
        root=os.path.expanduser("~/partnet-mobility"),
        batch_size=16,
        n_epochs=100,
        lr=0.0003,
        n_repeat=1,
        n_proc=60,
        wandb_log=wandb_log,
    )
"""
python -m part_embedding.flowtron.pretraining.pretrain_taxpose
"""
