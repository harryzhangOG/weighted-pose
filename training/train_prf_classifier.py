import os

import numpy as np
import torch
import torch.optim
import torch_geometric.loader as tgl
import wandb
from tqdm import tqdm

from part_embedding.flowtron.models.classifiers import PrisRevFF_Classifier
from part_embedding.taxpose4art.train_utils import create_classifier_dataset


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

    train_dset, test_dset, unseen_dset = create_classifier_dataset(
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
        wandb.init(project="3wayclassifier", entity="harryzhangog", config=dict_config)
        run_name = wandb.run.name
        run_name_log = f"3way-classifier-{run_name}"
        wandb.run.name = run_name_log
    else:
        run_name_log = "debug"

    model = PrisRevFF_Classifier().to(device)
    if finetune:
        ckpt = "taxpose-new-divine-snowball-9"
        model_path = f"part_embedding/flowtron/checkpoints/{ckpt}/weights_040.pt"
        model.load_state_dict(torch.load(model_path))

    # opt = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    crit = torch.nn.CrossEntropyLoss()

    d = f"part_embedding/flowtron/checkpoints/{run_name_log}"
    os.makedirs(d, exist_ok=True)

    # Train
    train_step = 0
    val_step = 0
    unseen_step = 0
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
            ff_or_art = np.random.uniform(0, 1)
            train_step += 1
            # if ff_or_art < 0.3:
            if train_step % 2 == 0:
                action = action_art
                anchor = anchor_art
                anchor_demo = anchor_demo_art
            else:
                action = action_ff
                anchor = anchor_ff
                anchor_demo = anchor_demo_ff
            action = action.to(device)
            anchor = anchor.to(device)

            opt.zero_grad()

            pred_logits = model(anchor)
            loss = crit(pred_logits, anchor.obj_cat)

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

        # Validation
        for (
            action_art,
            anchor_art,
            anchor_demo_art,
            action_ff,
            anchor_ff,
            anchor_demo_ff,
        ) in pbar_val:
            ff_or_art = np.random.uniform(0, 1)
            val_step += 1
            # if ff_or_art < 0.3:
            if val_step % 2 == 0:
                action = action_art
                anchor = anchor_art
                anchor_demo = anchor_demo_art
            else:
                action = action_ff
                anchor = anchor_ff
                anchor_demo = anchor_demo_ff
            action = action.to(device)
            anchor = anchor.to(device)
            with torch.no_grad():
                pred_logits = model(anchor)
                loss = crit(pred_logits, anchor.obj_cat)

                if wandb_log:
                    wandb.log({"val_loss": loss, "val-x-axis": val_step})

                if i % n_print == 0:
                    desc = f"Epoch {i:03d}: Step {val_step}  Val Loss:{loss.item():.3f}"
                    pbar_val.set_description(desc)


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
