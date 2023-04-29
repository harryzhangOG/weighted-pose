import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as tgd
from torch_geometric.data.batch import Batch

import part_embedding.nets.pointnet2 as pnp
from part_embedding.flowtron.models.classifiers import ArtFFDenseSeg
from part_embedding.flowtron.models.demo_embedding import SingleFrameDemoEncoder
from part_embedding.flowtron.models.flowbotv2 import SVD
from part_embedding.goal_inference_brian.brian_chuer_model import (
    ResidualFlow_DiffEmbTransformer,
    extract_flow_and_weight,
)
from part_embedding.goal_inference_brian.dgcnn import get_graph_feature
from part_embedding.goal_inference_brian.se3 import dualflow2pose_weighted_v2  # noqa
from part_embedding.nets.pointnet2 import PN2DenseParams


class DGCNN_Mod(nn.Module):
    def __init__(self, emb_dims=512, input_dims=3, gc=False):
        super(DGCNN_Mod, self).__init__()
        self.conv1 = nn.Conv2d(input_dims * 2, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        final_conv = 512 if not gc else 512 + 128
        self.conv5 = nn.Conv2d(final_conv, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x, demo_emb=None):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        if demo_emb is not None:
            x_cat = demo_emb.reshape(-1, 128, 1, 1).repeat(
                1, 1, x4.shape[2], x4.shape[3]
            )
            x4 = torch.cat([x4, x_cat], dim=1)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x


class GCGoalFlowNet(nn.Module):
    def __init__(self, p: PN2DenseParams = PN2DenseParams()):
        super().__init__()
        p.in_dim = 0
        p.final_outdim = 3

        # The Set Aggregation modules.
        self.sa1 = pnp.SAModule(3 + p.in_dim, p.sa1_outdim, p=p.sa1)
        self.sa2 = pnp.SAModule(3 + p.sa1_outdim, p.sa2_outdim, p=p.sa1)
        self.sa3 = pnp.GlobalSAModule(3 + p.sa2_outdim, p.gsa_outdim, p=p.gsa)

        # The Feature Propagation modules.
        self.fp3 = pnp.FPModule(p.gsa_outdim + p.sa2_outdim, p.sa2_outdim, p.fp3)
        self.fp2 = pnp.FPModule(p.sa2_outdim + p.sa1_outdim, p.sa1_outdim, p.fp2)
        self.fp1 = pnp.FPModule(p.sa1_outdim + p.in_dim, p.fp1_outdim, p.fp1)

        # Final linear layers at the output.
        self.lin1 = torch.nn.Linear(p.fp1_outdim, p.lin1_dim)
        self.lin2 = torch.nn.Linear(p.lin1_dim, p.lin2_dim)
        self.lin3 = torch.nn.Linear(p.lin2_dim, p.final_outdim)  # Flow output.
        self.proj1 = torch.nn.Linear(p.sa2_outdim // 2, p.sa2_outdim)
        self.proj2 = torch.nn.Linear(p.sa1_outdim, p.sa1_outdim)
        self.svd = SVD()

    def forward(self, anchor, demo_emb=None):
        sa0_out = (anchor.x, anchor.pos, anchor.batch)

        if demo_emb is not None:
            task_sp = demo_emb

        sa1_out = self.sa1(*sa0_out)
        sa2_out = self.sa2(*sa1_out)
        x3, pos3, batch3 = self.sa3(*sa2_out)

        if demo_emb is not None:
            x3 = torch.mul(x3, task_sp.tile(1, 8))

        sa3_out = x3, pos3, batch3

        fp3_x, fp3_pos, fp3_batch = self.fp3(*sa3_out, *sa2_out)

        if demo_emb is not None:
            temp = torch.cat(
                [
                    task_sp[i].tile((fp3_batch == i).sum(), 2)
                    for i in range(task_sp.shape[0])
                ]
            )
            fp3_x = torch.mul(fp3_x, temp)
        fp3_out = fp3_x, fp3_pos, fp3_batch
        fp2_x, fp2_pos, fp2_batch = self.fp2(*fp3_out, *sa1_out)

        if demo_emb is not None:
            temp = torch.cat(
                [
                    task_sp[i].tile((fp2_batch == i).sum(), 1)
                    for i in range(task_sp.shape[0])
                ]
            )
            fp2_x = torch.mul(fp2_x, temp)
        fp2_out = fp2_x, fp2_pos, fp2_batch
        x, _, _ = self.fp1(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x


class WeightedPose(nn.Module):
    def __init__(self, seg_ckpt="4way-segmenter-volcanic-universe-7", pretrained=True):
        super(WeightedPose, self).__init__()
        self.demo_emb = SingleFrameDemoEncoder()
        emb_dims = 512
        self.taxpose_emb = ResidualFlow_DiffEmbTransformer(emb_dims=emb_dims, gc=False)
        # self.taxpose_emb.emb_nn_action = DGCNN_Mod(emb_dims=emb_dims, gc=False)
        # self.taxpose_emb.emb_nn_anchor = DGCNN_Mod(emb_dims=emb_dims, gc=False)
        self.weight_normalize = "l1"
        self.softmax_temperature = None
        self.goal_flow = GCGoalFlowNet()
        if pretrained:
            print("Loading pretrained models")
            gf_ckpt = "pretrain_goalflow-faithful-jazz-9"
            self.goal_flow.load_state_dict(
                torch.load(
                    f"part_embedding/flowtron/pretraining/checkpoints/{gf_ckpt}/weights_060.pt"
                )
            )
            tp_ckpt = "pretrain-taxpose-ruby-totem-3"
            state_dict = {}
            loaded_dict = torch.load(
                f"part_embedding/flowtron/pretraining/checkpoints/{tp_ckpt}/weights_060.pt"
            )
            for k in loaded_dict:
                if not "demo" in k:
                    state_dict[k[12:]] = loaded_dict[k]
            self.taxpose_emb.load_state_dict(state_dict)
            print("Pretrained models loaded")
        self.seg_net = ArtFFDenseSeg()
        model_path = f"part_embedding/flowtron/checkpoints/{seg_ckpt}/weights_060.pt"
        self.seg_net.load_state_dict(torch.load(model_path))

    def forward(self, action, anchor):
        anchor.x = None
        goal_flow = self.goal_flow(anchor)
        Y = anchor.pos

        # For now, assume we know which points are action
        Xs = Y.view(anchor.num_graphs, -1, 3)[:, :500, :]
        Ys = Y.view(anchor.num_graphs, -1, 3)[:, 500:, :]

        Fx, Fy = self.taxpose_emb(Xs, Ys)

        Fx, pred_w_action = extract_flow_and_weight(Fx, True)
        Fy, pred_w_anchor = extract_flow_and_weight(Fy, True)

        # Don't want to update learned_w
        learned_w = self.seg_net(anchor).detach()
        temp = learned_w.view(anchor.num_graphs, -1, 4)[:, :500, :]
        temp2 = torch.concat(
            [temp[:, :, 1:3].max(axis=2)[0].unsqueeze(-1), temp[:, :, 3:]], axis=-1
        )
        # learned_w_ff = nn.Sigmoid()(temp2)[:, :, 0]
        learned_w_ff = nn.Softmax(dim=-1)(temp2)[:, :, 0]

        pred_T_action = dualflow2pose_weighted_v2(
            xyz_src=Xs,
            xyz_tgt=Ys,
            flow_src=Fx,
            flow_tgt=Fy,
            goal_flow=goal_flow,
            learned_w=learned_w_ff.reshape(-1, 500, 1),
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

        return R_pred, t_pred, pred_T_action, Fx, Fy, goal_flow, learned_w


if __name__ == "__main__":
    model = WeightedPose().cuda()
    X = torch.rand(16, 500, 3).float().cuda()
    action_data = []
    for x in X:
        action_data.append(
            tgd.Data(
                pos=x,
            )
        )
    action_data = Batch.from_data_list(action_data).cuda()
    Y = torch.rand(16, 2000, 3).float().cuda()
    anchor_data = []
    for y in Y:
        anchor_data.append(tgd.Data(pos=y, x=torch.ones((2000, 1)).cuda()))
    anchor_data = Batch.from_data_list(anchor_data).cuda()
    demo = torch.rand(16, 2000, 3).float().cuda()
    demo_data = []
    for d in demo:
        demo_data.append(tgd.Data(pos=d, x=torch.ones((2000, 1)).cuda()))
    demo_data = Batch.from_data_list(demo_data).cuda()

    model(action_data, anchor_data, demo_data)
