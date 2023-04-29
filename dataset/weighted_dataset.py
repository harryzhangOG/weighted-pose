import csv
import os
import pickle
from typing import Dict, List, Optional, Protocol

import numpy as np
import pybullet as p
import torch
import torch.utils.data as td
import torch_geometric.data as tgd

import part_embedding.goal_inference.create_pm_goal_dataset as pgc
from part_embedding.datasets.pm.utils import SingleObjDataset, parallel_sample
from part_embedding.envs.render_sim import PMRenderEnv
from part_embedding.goal_inference.dataset_v2 import (
    CATEGORIES,
    SEM_CLASS_DSET_PATH,
    base_from_bottom,
    downsample_pcd_fps,
    find_link_index_to_open,
    find_valid_action_initial_pose,
    load_action_obj_with_valid_scale,
    render_input_new,
)
from part_embedding.goal_inference.dset_utils import (
    ACTION_OBJS,
    SNAPPED_GOAL_FILE,
    render_input_articulated,
)
from part_embedding.taxpose4art.dataset import find_link_index_to_open as find_link
from part_embedding.taxpose4art.generate_art_training_data import transform_pcd
from part_embedding.taxpose4art.generate_art_training_data_flowbot import get_category


def get_sem(oid, move_joint):
    sem_file = csv.reader(
        open(os.path.expanduser(f"~/partnet-mobility/raw/{oid}/semantics.txt")),
        delimiter=" ",
    )
    for line in sem_file:
        if move_joint in line:
            return line[1]


class ArtData(Protocol):

    # Action Info
    action_pos: torch.FloatTensor
    t_action_anchor: Optional[torch.FloatTensor]
    R_action_anchor: Optional[torch.FloatTensor]
    flow: Optional[torch.FloatTensor]

    # Anchor Info
    obj_id: str
    anchor_pos: torch.FloatTensor

    # Task specification
    loc: Optional[float]


class WeightedDataset(tgd.Dataset):
    def __init__(
        self,
        root: str,
        obj_ids: List[str],
        dset_name: str = None,
        use_processed: bool = True,
        n_repeat: int = 50,
        randomize_camera: bool = False,
        n_proc: int = 60,
        even_downsample: bool = False,
        rotate_anchor: bool = False,
    ):

        self.obj_ids = obj_ids
        self.dset_name = dset_name
        self.generated_metadata = pickle.load(
            open(
                f"part_embedding/flowtron/dataset/training_data/all_100_obj_tf.pkl",
                "rb",
            )
        )

        # Cache the environments. Only cache at the object level though.
        self.envs: Dict[str, PMRenderEnv] = {}
        if "hinge" in dset_name or "slider" in dset_name:
            if "hinge" in dset_name:
                split_dset_name = dset_name[:-6]
            else:
                split_dset_name = dset_name[:-7]
        else:
            split_dset_name = dset_name
        self.full_sem_dset = pickle.load(open(SEM_CLASS_DSET_PATH, "rb"))
        self.ff_dset = pickle.load(open(SEM_CLASS_DSET_PATH, "rb"))
        self.use_processed = use_processed
        self.n_repeat = n_repeat
        self.randomize_camera = randomize_camera
        self.n_proc = n_proc
        self.even_downsample = even_downsample
        self.rotate_anchor = rotate_anchor
        with open(SNAPPED_GOAL_FILE, "rb") as f:
            self.snapped_goal_dict = pickle.load(f)

        super().__init__(root)
        if self.use_processed:
            # Map from scene_id to dataset. Very hacky way of getting scene id.
            self.inmem_map: Dict[str, td.Dataset[ArtData]] = {
                data_path: SingleObjDataset(data_path)
                for data_path in self.processed_paths
            }
            self.inmem: td.ConcatDataset = td.ConcatDataset(
                list(self.inmem_map.values())
            )

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{key}_{self.n_repeat}.pt" for key in self.obj_ids]

    @property
    def processed_dir(self) -> str:
        chunk = ""
        if self.randomize_camera:
            chunk += "_random"
        if self.even_downsample:
            chunk += "_even"
        return os.path.join(self.root, f"weighted_flow_snap_dset_wcat" + chunk)

    def process(self):
        if not self.use_processed:
            return

        else:
            # Run parallel sampling!
            get_data_args = [(f, "0") for f in self.obj_ids]
            parallel_sample(
                dset_cls=WeightedDataset,
                dset_args=(
                    self.root,
                    self.obj_ids,
                    self.dset_name,
                    False,
                    self.n_repeat,
                    self.randomize_camera,
                    self.n_proc,
                    self.even_downsample,
                    self.rotate_anchor,
                ),
                # Needs to be a tuple of arguments, so we can expand it when calling get_args.
                get_data_args=get_data_args,
                n_repeat=self.n_repeat,
                n_proc=self.n_proc,
            )

    def len(self) -> int:
        return len(self.obj_ids) * self.n_repeat

    def get(self, idx: int) -> ArtData:
        if self.use_processed:
            try:
                data = self.inmem[idx]  # type: ignore
            except:
                breakpoint()
        else:
            idx = idx // self.n_repeat
            obj_id = self.obj_ids[idx]
            data = self.get_data(obj_id, "0")

        return data  # type: ignore

    def get_data(self, obj_id: str, goal_id) -> ArtData:
        """Get a single observation sample.

        Args:
            obj_id: The anchor object ID from Partnet-Mobility.
        Returns:
            ObsActionData and AnchorData, both in the world frame, with a relative transform.
        """
        action_id = "block"

        tmp_id = obj_id.split("_")[0]

        env = PMRenderEnv(tmp_id, self.raw_dir, camera_pos=[-2.5, 0, 2.5], gui=False)
        if get_category(tmp_id) == "Refrigerator":
            sub_cat = "Fridge"
        elif get_category(tmp_id) == "WashingMachine":
            sub_cat = "Washingmachine"
        elif get_category(tmp_id) == "StorageFurniture":
            sub_cat = "Drawer"
        else:
            sub_cat = get_category(tmp_id)

        # Obtain entries from meta data
        curr_data_entry = self.generated_metadata[sub_cat][tmp_id][
            int(obj_id.split("_")[1]) % 100
        ]
        start_ang_obs = curr_data_entry["start"]
        end_ang = curr_data_entry["end"]
        transform = curr_data_entry["transformation"]

        # Next, check to see if the object needs to be opened in any way.
        object_dict = pgc.all_objs[CATEGORIES[tmp_id].lower()]

        partsem = object_dict[f"{tmp_id}_0"]["partsem"]

        links_tomove = find_link(self.full_sem_dset, partsem, obj_id, object_dict)
        currsem = get_sem(tmp_id, links_tomove)

        start_ang_goal = env.get_specific_joints_range(links_tomove)[1]
        if currsem == "hinge":
            start_ang_goal = start_ang_goal / np.pi * 180
            object_category = 0  # hinge: 0
        else:
            object_category = 1  # prismatic: 1
        return_data = []
        for curr_idx, start_ang in enumerate([start_ang_obs, start_ang_goal]):
            env.set_specific_joints_angle(links_tomove, start_ang, sem=currsem)

            # Render the scene.
            P_world, pc_seg, rgb, action_mask = render_input_articulated(
                env, links_tomove
            )

            # Separate out the action and anchor points.
            P_action_world = P_world[action_mask]
            P_anchor_world = P_world[~action_mask]

            P_action_world = torch.from_numpy(P_action_world)
            P_anchor_world = torch.from_numpy(P_anchor_world)

            action_pts_num = 500
            # Now, downsample
            if self.even_downsample:
                action_ixs = downsample_pcd_fps(P_action_world, n=action_pts_num)
                anchor_ixs = downsample_pcd_fps(P_anchor_world, n=2000 - action_pts_num)
            else:
                action_ixs = torch.randperm(len(P_action_world))[:action_pts_num]
                anchor_ixs = torch.randperm(len(P_anchor_world))[
                    : (2000 - action_pts_num)
                ]

            # Rebuild the world
            P_action_world = P_action_world[action_ixs]
            while len(P_action_world) < action_pts_num:
                temp = np.random.choice(np.arange(len(P_action_world)))
                P_action_world = torch.cat(
                    [
                        P_action_world,
                        P_action_world[temp : temp + 1],
                    ]
                )

            if len(P_action_world) != 500:
                print(len(P_action_world))
            P_anchor_world = P_anchor_world[anchor_ixs]
            while len(P_anchor_world) < 2000 - action_pts_num:
                temp = np.random.choice(np.arange(len(P_anchor_world)))
                P_anchor_world = torch.cat(
                    [
                        P_anchor_world,
                        P_anchor_world[temp : temp + 1],
                    ]
                )
            P_world = np.concatenate([P_action_world, P_anchor_world], axis=0)

            # Regenerate a mask.
            mask_act = torch.ones(len(P_action_world)).int()
            mask_anc = torch.zeros(len(P_anchor_world)).int()
            mask = torch.cat([mask_act, mask_anc])

            # Compute the transform from action object to goal.
            t_action_anchor = transform[:-1, -1]
            t_action_anchor = torch.from_numpy(t_action_anchor).float().unsqueeze(0)
            R_action_anchor = transform[:-1, :-1]
            R_action_anchor = torch.from_numpy(R_action_anchor).float().unsqueeze(0)

            # Compute the ground-truth flow.
            flow = np.zeros_like(P_world)
            flow2tf_res = transform_pcd(P_world[mask == 1], transform)
            flow[mask == 1] = flow2tf_res - P_world[mask == 1]
            flow = torch.from_numpy(flow).float()

            if curr_idx == 0:
                return_data.append(P_action_world)
                return_data.append(P_anchor_world)
            else:
                return_data.append(torch.from_numpy(P_world))

        # Assemble the data.
        action_data = tgd.Data(
            pos=return_data[0].float(), loc=end_ang - start_ang, action_id=None
        )
        anchor_data = tgd.Data(
            obj_id=obj_id,
            pos=return_data[1].float(),
            flow=flow.float(),
            x=mask.float().reshape(-1, 1),
            loc=0,
            t_action_anchor=t_action_anchor,
            R_action_anchor=R_action_anchor,
            obj_cat=object_category,
        )
        demo_data = tgd.Data(
            pos=return_data[2].float(),
            flow=flow.float(),
            x=mask.float().reshape(-1, 1),
        )
        p.disconnect()

        # First, create an environment which will generate our source observations.
        tmp_id = obj_id.split("_")[0]
        env = PMRenderEnv(tmp_id, self.raw_dir, camera_pos=[-3, 0, 1.2], gui=False)
        object_dict = pgc.all_objs[CATEGORIES[tmp_id].lower()]

        # Next, check to see if the object needs to be opened in any way.
        partsem = object_dict[f"{tmp_id}_{goal_id}"]["partsem"]
        if partsem != "none":
            links_tomove = find_link_index_to_open(
                self.ff_dset, partsem, tmp_id, object_dict, goal_id
            )
            env.articulate_specific_joints(links_tomove, 0.9)

        # Select the action object.
        action_obj = ACTION_OBJS[action_id]

        # Load the object at the original floating goal, with a size that is valid there.
        info = object_dict[f"{tmp_id}_{goal_id}"]
        floating_goal = np.array([info["x"], info["y"], info["z"]])
        action_body_id, scale = load_action_obj_with_valid_scale(
            action_obj, floating_goal, env
        )

        # Find the actual desired goal position.
        action_goal_pos_pre = self.snapped_goal_dict[CATEGORIES[tmp_id].lower()][
            f"{tmp_id}_{goal_id}"
        ]
        action_goal_pos = base_from_bottom(action_body_id, env, action_goal_pos_pre)

        # The start position depends on which mode we're in.
        action_obs_pos = find_valid_action_initial_pose(action_body_id, env)
        p.resetBasePositionAndOrientation(
            action_body_id,
            posObj=action_obs_pos,
            ornObj=[0, 0, 0, 1],
            physicsClientId=env.client_id,
        )
        P_world, pc_seg, rgb, action_mask = render_input_new(action_body_id, env)

        # We need enough visible points.
        if sum(action_mask) < 1:
            # If we don't find them in obs mode, it's because the random object position has been occluded.
            # In this case, we just need to resample.
            MAX_ATTEMPTS = 20
            i = 0
            while sum(action_mask) < 1:
                action_obs_pos = find_valid_action_initial_pose(action_body_id, env)
                p.resetBasePositionAndOrientation(
                    action_body_id,
                    posObj=action_obs_pos,
                    ornObj=[0, 0, 0, 1],
                    physicsClientId=env.client_id,
                )

                P_world, pc_seg, rgb, action_mask = render_input_new(
                    action_body_id, env
                )

                i += 1
                if i >= MAX_ATTEMPTS:
                    raise ValueError("couldn't find a valid goal :(")

        return_data = []
        for curr_idx, action_pos in enumerate([action_obs_pos, action_goal_pos]):
            # Place the object at the desired start position.
            p.resetBasePositionAndOrientation(
                action_body_id,
                posObj=action_pos,
                ornObj=[0, 0, 0, 1],
                physicsClientId=env.client_id,
            )

            P_world, pc_seg, rgb, action_mask = render_input_new(action_body_id, env)

            # Separate out the action and anchor points.
            P_action_world = P_world[action_mask]
            P_anchor_world = P_world[~action_mask]

            P_action_world = torch.from_numpy(P_action_world)
            P_anchor_world = torch.from_numpy(P_anchor_world)

            action_pts_num = 500
            # Now, downsample
            if self.even_downsample:
                action_ixs = downsample_pcd_fps(P_action_world, n=action_pts_num)
                anchor_ixs = downsample_pcd_fps(P_anchor_world, n=2000 - action_pts_num)
            else:
                action_ixs = torch.randperm(len(P_action_world))[:action_pts_num]
                anchor_ixs = torch.randperm(len(P_anchor_world))[
                    : (2000 - action_pts_num)
                ]

            # Rebuild the world
            P_action_world = P_action_world[action_ixs]
            while len(P_action_world) < action_pts_num:
                temp = np.random.choice(np.arange(len(P_action_world)))
                P_action_world = torch.cat(
                    [
                        P_action_world,
                        P_action_world[temp : temp + 1],
                    ]
                )

            if len(P_action_world) != 500:
                print(len(P_action_world))
            P_anchor_world = P_anchor_world[anchor_ixs]
            while len(P_anchor_world) < 2000 - action_pts_num:
                temp = np.random.choice(np.arange(len(P_anchor_world)))
                P_anchor_world = torch.cat(
                    [
                        P_anchor_world,
                        P_anchor_world[temp : temp + 1],
                    ]
                )
            P_world = np.concatenate([P_action_world, P_anchor_world], axis=0)
            # Regenerate a mask.
            mask_act = torch.ones(len(P_action_world)).int()
            mask_anc = torch.zeros(len(P_anchor_world)).int()
            mask = torch.cat([mask_act, mask_anc])

            # Depending on what mode we're in, create the ground truth displacement data or not.
            # Compute the transform from action object to goal.
            t_action_anchor = action_goal_pos - action_obs_pos
            t_action_anchor = torch.from_numpy(t_action_anchor).float().unsqueeze(0)

            # Compute the ground-truth flow.
            flow = np.tile(t_action_anchor, (P_world.shape[0], 1))
            flow[~mask] = 0
            flow = torch.from_numpy(flow[mask == 1]).float()
            if len(flow) != len(P_action_world):
                breakpoint()
            if curr_idx == 0:
                return_data.append(P_action_world)
                return_data.append(P_anchor_world)
            else:
                return_data.append(torch.from_numpy(P_world))

        # Assemble the data.
        action_data_ff = tgd.Data(
            pos=return_data[0].float(),
            loc=None,
            action_id=action_id,
        )
        anchor_data_ff = tgd.Data(
            obj_id=obj_id,
            pos=return_data[1].float(),
            flow=flow.float(),
            # x=mask.float().reshape(-1, 1),
            loc=0,
            t_action_anchor=t_action_anchor,
            R_action_anchor=torch.eye(3).unsqueeze(0),
            obj_cat=2,
        )
        demo_data_ff = tgd.Data(
            pos=return_data[2].float(),
            flow=flow.float(),
            x=mask.float().reshape(-1, 1),
        )
        p.disconnect()

        return action_data, anchor_data, demo_data, action_data_ff, anchor_data_ff, demo_data_ff  # type: ignore
