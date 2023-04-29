import torch.nn as nn

from part_embedding.flowtron.models.demo_embedding import DemoEncoder
from part_embedding.goal_inference_brian.brian_chuer_model import (
    ResidualFlow_DiffEmbTransformer,
    extract_flow_and_weight,
)
from part_embedding.goal_inference_brian.se3 import dualflow2pose


class DemoCondTAXPose(nn.Module):
    def __init__(self):
        super(DemoCondTAXPose, self).__init__()
        self.demo_emb = DemoEncoder()
        self.taxpose_emb = ResidualFlow_DiffEmbTransformer(emb_dims=512, gc=False)
        self.weight_normalize = "l1"
        self.softmax_temperature = None

    def forward(self, action, anchor, demo_pcd):
        demo_emb = self.demo_emb(demo_pcd)
        anchor.x = None
        X = action.pos
        Y = anchor.pos
        Xs = X.view(action.num_graphs, -1, 3)
        Ys = Y.view(anchor.num_graphs, -1, 3)

        Fx, Fy = self.taxpose_emb(Xs, Ys)

        Fx, pred_w_action = extract_flow_and_weight(Fx, True)
        Fy, pred_w_anchor = extract_flow_and_weight(Fy, True)

        pred_T_action = dualflow2pose(
            xyz_src=Xs,
            xyz_tgt=Ys,
            flow_src=Fx,
            flow_tgt=Fy,
            weights_src=pred_w_action,  # Taxpose alpha
            weights_tgt=pred_w_anchor,
            return_transform3d=True,
            normalization_scehme=self.weight_normalize,
            temperature=self.softmax_temperature,
        )

        # It's weirdly structured...
        mat = pred_T_action.get_matrix().transpose(-1, -2)

        R_pred = mat[:, :3, :3]
        t_pred = mat[:, :3, 3]

        return R_pred, t_pred, pred_T_action, Fx, Fy
