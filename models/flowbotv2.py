import numpy as np
import torch.nn as nn
import torch.optim

from part_embedding.flow_prediction.artflownet import ArtFlowNetParams, create_flownet
from part_embedding.taxpose4art.gc_goalflow_net import GCGoalFlowNet


class FlowbotV2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = create_flownet(0, 6)

    def forward(self, action, anchor):
        flowscrew = self.net(anchor)
        return flowscrew


class SVD(nn.Module):
    def __init__(self):
        super(SVD, self).__init__()
        self.emb_dims = 16

    def forward(self, src, pred_flow, gt_flow):
        nonzero_gt_flowixs = torch.where(gt_flow.norm(dim=1) != 0.0)
        pred_flow_nz = pred_flow[nonzero_gt_flowixs].reshape(-1, 500, 3)
        src = (src).reshape(-1, 500, 3)
        # R, t = flow2pose(src, pred_flow_nz)

        X = src
        Y = src + pred_flow_nz
        X_bar = X.mean(axis=1).reshape(-1, 1, 3)
        Y_bar = Y.mean(axis=1).reshape(-1, 1, 3)
        H = torch.bmm((X - X_bar).transpose(1, 2), (Y - Y_bar))
        U, _, Vh = torch.linalg.svd(H)
        R = torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2))
        t = -torch.bmm(X_bar, R.transpose(1, 2)) + Y_bar

        return R, t


class GoalFlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        p = ArtFlowNetParams()
        # self.net = create_flownet(in_channels=1, out_channels=3, p=p.net)
        self.net = GCGoalFlowNet()
        self.svd = SVD()

    def forward(self, action, anchor):
        if isinstance(self.net, GCGoalFlowNet):
            # anchor.task_sp = action.loc.float().reshape(-1, 1)
            task_sps = []
            for sp in action.loc.float().reshape(-1, 1):
                if abs(sp.item()) < 1:
                    task_sps.append([0, sp.item()])
                else:
                    task_sps.append([1, sp.item()])
            task_sps = np.array(task_sps)
            anchor.task_sp = torch.from_numpy(task_sps).cuda().float().reshape(-1, 2)
            anchor.x = None
        flow = self.net(anchor)
        return flow


class ArtClassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = create_flownet(1, 2)

    def forward(self, action, anchor):
        out = self.net(anchor)
        return out
