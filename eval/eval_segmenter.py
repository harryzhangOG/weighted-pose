import os

import numpy as np
import torch.optim
import torch_geometric.loader as tgl
from tqdm import tqdm

from part_embedding.flowtron.models.classifiers import ArtFFDenseSeg
from part_embedding.goal_inference.dataset_v2 import CATEGORIES
from part_embedding.taxpose4art.train_utils import create_segmenter_dataset


def load_model(ckpt):
    model_path = f"part_embedding/flowtron/checkpoints/{ckpt}/weights_060.pt"
    model = ArtFFDenseSeg()
    model.load_state_dict(torch.load(model_path))

    return model.cuda()


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
    train_total = 0
    train_correct = 0
    train_meta = {0: [], 1: [], 2: [], 3: []}
    val_total = 0
    val_correct = 0
    val_meta = {0: [], 1: [], 2: [], 3: []}

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
        pred_logits = model(anchor)
        labels = anchor.seg_label.reshape(
            -1,
        ).long()
        _, predicted = torch.max(pred_logits.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        for label in [0, 1, 2, 3]:
            try:
                train_meta[label].append(
                    torch.logical_and(predicted == labels, labels == label).sum().item()
                    / ((labels == label).sum().item())
                )
            except:
                continue

        action = action_ff.to(device)
        anchor = anchor_ff.to(device)
        pred_logits = model(anchor)
        labels = anchor.seg_label.reshape(
            -1,
        ).long()
        _, predicted = torch.max(pred_logits.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        for label in [0, 1, 2, 3]:
            try:
                train_meta[label].append(
                    torch.logical_and(predicted == labels, labels == label).sum().item()
                    / ((labels == label).sum().item())
                )
            except:
                continue

    print("===================== TRAINING OBJECTS =====================")
    print("Free Floating:")

    for cat in train_meta:
        print(
            f"{cat:<20}\tAverage accuracy: {sum(train_meta[cat]) / len(train_meta[cat]):.2f}"
        )
    print(f"AVERAGE OVERALL: {train_correct / train_total}")

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
        pred_logits = model(anchor)
        labels = anchor.seg_label.reshape(
            -1,
        ).long()
        _, predicted = torch.max(pred_logits.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
        for label in [0, 1, 2, 3]:
            try:
                val_meta[label].append(
                    torch.logical_and(predicted == labels, labels == label).sum().item()
                    / ((labels == label).sum().item())
                )
            except:
                continue

        action = action_ff.to(device)
        anchor = anchor_ff.to(device)
        pred_logits = model(anchor)
        labels = anchor.seg_label.reshape(
            -1,
        ).long()
        _, predicted = torch.max(pred_logits.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
        for label in [0, 1, 2, 3]:
            try:
                val_meta[label].append(
                    torch.logical_and(predicted == labels, labels == label).sum().item()
                    / ((labels == label).sum().item())
                )
            except:
                continue

    print("===================== VAL OBJECTS =====================")
    print("Free Floating:")
    for cat in val_meta:
        print(
            f"{cat:<20}\tAverage accuracy: {sum(val_meta[cat]) / len(val_meta[cat]):.2f}"
        )
    print(f"AVERAGE OVERALL: {val_correct / val_total}")


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
