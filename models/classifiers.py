import torch.nn as nn

from part_embedding.nets.pointnet2 import (
    GlobalSAModule,
    PN2Dense,
    PN2EncoderParams,
    SAModule,
)


class PrisRevFF_Classifier(nn.Module):
    """
    A 3-way classifier that classify if the given point cloud is:
        - Prismatic
        - Revolute, or
        - Free-floating
    This is a global classifier. That is, it only outputs a single latent vector.
    """

    def __init__(
        self,
        in_dim: int = 0,
        out_dim: int = 3,
        p: PN2EncoderParams = PN2EncoderParams(),
    ):

        super().__init__()

        # The Set Aggregation modules.
        self.sa1_module = SAModule(in_chan=3 + in_dim, out_chan=p.sa1_outdim, p=p.sa1)
        self.sa2_module = SAModule(
            in_chan=3 + p.sa1_outdim, out_chan=p.sa2_outdim, p=p.sa2
        )
        self.sa3_module = GlobalSAModule(
            in_chan=3 + p.sa2_outdim, out_chan=out_dim, p=p.gsa
        )

    def forward(self, data):
        sa0_out = (None, data.pos, data.batch)

        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return x


class ArtFFDenseSeg(nn.Module):
    """
    A dense, perpoint segmentation model that segments:
        - Prismatic oart
        - Revolute part
        - FF object
        - Furniture body
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = PN2Dense(out_channels=4)

    def forward(self, data):
        return self.net(data)
