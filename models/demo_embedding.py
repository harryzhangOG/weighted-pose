import torch
import torch.nn as nn

import part_embedding.nets.pointnet2 as pnp


class SingleFrameDemoEncoder(nn.Module):
    """
    Encode a single-frame demonstration

    e.g. A oven with door open, a block in the oven.

    Input: A point cloud of demo frame with labels, where 0 means action and 1 means anchor
    Output: an emb_dims + 1 dimensional tensor where the first emb_dims entries represent the emb
            and the last entry represents the weight.
    """

    def __init__(self, emb_dims=128):
        super(SingleFrameDemoEncoder, self).__init__()

        self.emb_dims = emb_dims + 1
        self.emb_net = pnp.PN2Encoder(in_dim=1, out_dim=self.emb_dims)

    def forward(self, demo_pcd):
        demo_embedding_w = self.emb_net(demo_pcd)  # (NxB) x emb_dims + 1
        demo_embedding = demo_embedding_w[:, :-1].reshape((-1, self.emb_dims - 1))
        w = torch.sigmoid(demo_embedding_w[:, -1].reshape((-1, 1)))
        return demo_embedding, w


class DemoEncoder(nn.Module):
    """
    Encode a single-frame demonstration

    e.g. A oven with door open, a block in the oven.

    Input: A point cloud of demo frame with labels, where 0 means action and 1 means anchor
    Output: an emb_dims + 1 dimensional tensor where the first emb_dims entries represent the emb
            and the last entry represents the weight.
    """

    def __init__(self, emb_dims=128):
        super(DemoEncoder, self).__init__()

        self.emb_dims = emb_dims
        self.emb_net = pnp.PN2Encoder(in_dim=1, out_dim=self.emb_dims)

    def forward(self, demo_pcd):
        demo_embedding_w = self.emb_net(demo_pcd)  # (NxB) x emb_dims
        demo_embedding = demo_embedding_w.reshape((-1, self.emb_dims))
        return demo_embedding
