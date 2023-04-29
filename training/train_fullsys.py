import os

import numpy as np
import torch
import torch.optim
import torch_geometric.loader as tgl
import wandb
from tqdm import tqdm

from part_embedding.flowtron.models.classifiers import ArtFFDenseSeg
from part_embedding.goal_inference.model_sg import full_sys_plot
from part_embedding.taxpose4art.train_utils import create_fullsys_dataset


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
    if wandb_log:
        wandb.init(project="full-sys", entity="harryzhangog")
        run_name = wandb.run.name
        run_name_log = f"fullsys-{run_name}"
        wandb.run.name = run_name_log

    # Set up wandb config
    dict_config = {
        "learning_rate": lr,
        "n_repeat": n_repeat,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "dataset": dset_name,
    }

    train_dset, test_dset, unseen_dset = create_fullsys_dataset(
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
    train_loader = tgl.DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)
    test_loader = tgl.DataLoader(
        test_dset, batch_size=min(len(test_dset), 4), shuffle=True, num_workers=0
    )
    pbar = tqdm(train_loader)
    pbar_val = tqdm(test_loader)
    for (anchor_step1, anchor_step2, anchor_step3) in pbar:
        if wandb_log:
            wandb.log(
                {
                    "train_rand_plot": full_sys_plot(
                        anchor_step1.pos.reshape((anchor_step1.num_graphs, -1, 3))[0],
                        anchor_step2.pos.reshape((anchor_step1.num_graphs, -1, 3))[0],
                        anchor_step3.pos.reshape((anchor_step1.num_graphs, -1, 3))[0],
                    )
                }
            )
    for (anchor_step1, anchor_step2, anchor_step3) in pbar_val:
        continue


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ft",
        action="store_true",
        help="if we want to finetune",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="if we want to wandb",
    )

    args = parser.parse_args()

    wandb_log = args.wandb
    finetune = args.ft
    lr = 0.0003 if not finetune else 8e-5

    main(
        dset_name=f"classifier",
        root=os.path.expanduser("~/partnet-mobility"),
        batch_size=16,
        n_epochs=100,
        lr=0.001,
        n_repeat=1,
        n_proc=60,
        wandb_log=wandb_log,
    )
