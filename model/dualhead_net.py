

import sys
sys.path.append('model/basic_modules')
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params


from ms_gcn import MultiScale_GraphConv as MS_GCN
from ms_tcn import MultiScale_TemporalConv as MS_TCN
from ms_g3d_basic_module import MultiWindow_MS_G3D


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3):
        super(Model, self).__init__()

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # STGC Block: original channels is 96
        c1 = 96

        # Fine Branch: channel dimension should be 1/2 of Coarse Branch
        c1_fine = 48
        c2_fine = c1_fine * 2  # 96
        c3_fine = c2_fine * 2  # 192

        # Coarse Branch: channel dimension remains 96
        c1_coarse = 96
        c2_coarse = c1_coarse * 2  # 192
        c3_coarse = c2_coarse * 2  # 384


        # STGC Block
        self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        # Fine Block 1: Embedding
        self.embed_fine = nn.Conv2d(c1, c1_fine, 1) # decrease the channels to 1/2

        # Fine Block 2
        self.gcn3d2_fine = MultiWindow_MS_G3D(c1_fine, c2_fine, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2_fine = nn.Sequential(
            MS_GCN(num_gcn_scales, c1_fine, c1_fine, A_binary, disentangled_agg=True),
            MS_TCN(c1_fine, c2_fine, stride=2),
            MS_TCN(c2_fine, c2_fine))
        self.sgcn2_fine[-1].act = nn.Identity()
        self.tcn2_fine = MS_TCN(c2_fine, c2_fine)

        # Fine Block 3
        self.gcn3d3_fine = MultiWindow_MS_G3D(c2_fine, c3_fine, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3_fine = nn.Sequential(
            MS_GCN(num_gcn_scales, c2_fine, c2_fine, A_binary, disentangled_agg=True),
            MS_TCN(c2_fine, c3_fine, stride=2),
            MS_TCN(c3_fine, c3_fine))
        self.sgcn3_fine[-1].act = nn.Identity()
        self.tcn3_fine = MS_TCN(c3_fine, c3_fine)

        # Fine Head
        self.fc_fine = nn.Linear(c3_fine, num_class)



        # Coarse Block 1: Temporal Subsampling

        # Coarse Block 2
        self.gcn3d2_coarse = MultiWindow_MS_G3D(c1_coarse, c2_coarse, A_binary, num_g3d_scales, window_sizes=[3],
                                                window_dilations=[1], window_stride=2)
        self.sgcn2_coarse = nn.Sequential(
            MS_GCN(num_gcn_scales, c1_coarse, c1_coarse, A_binary, disentangled_agg=True),
            MS_TCN(c1_coarse, c2_coarse, stride=2, dilations=[1, 2]),
            MS_TCN(c2_coarse, c2_coarse, dilations=[1, 2]))
        self.sgcn2_coarse[-1].act = nn.Identity()
        self.tcn2_coarse = MS_TCN(c2_coarse, c2_coarse, dilations=[1, 2])

        # Coarse Block 3
        self.gcn3d3_coarse = MultiWindow_MS_G3D(c2_coarse, c3_coarse, A_binary, num_g3d_scales, window_sizes=[3],
                                                window_dilations=[1], window_stride=2)
        self.sgcn3_coarse = nn.Sequential(
            MS_GCN(num_gcn_scales, c2_coarse, c2_coarse, A_binary, disentangled_agg=True),
            MS_TCN(c2_coarse, c3_coarse, stride=2, dilations=[1, 2]),
            MS_TCN(c3_coarse, c3_coarse, dilations=[1, 2]))
        self.sgcn3_coarse[-1].act = nn.Identity()
        self.tcn3_coarse = MS_TCN(c3_coarse, c3_coarse, dilations=[1, 2])

        # Coarse Head
        self.fc_coarse = nn.Linear(c3_coarse, num_class)

        # Temporal Attention 1
        self.conv_ta_fine_to_coarse_block_2_to_1 = nn.Conv1d(c2_fine, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta_fine_to_coarse_block_2_to_1.weight, 0)
        nn.init.constant_(self.conv_ta_fine_to_coarse_block_2_to_1.bias, 0)

        # Temporal Attention 2
        self.conv_ta_fine_to_coarse_block_3_to_2 = nn.Conv1d(c3_fine, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta_fine_to_coarse_block_3_to_2.weight, 0)
        nn.init.constant_(self.conv_ta_fine_to_coarse_block_3_to_2.bias, 0)

        # Spatial Attention 1
        num_jpts = A_binary.shape[-1]
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        pad = (ker_jpt - 1) // 2

        self.conv_sa_coarse_to_fine_block_2_to_2 = nn.Conv1d(c2_coarse, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa_coarse_to_fine_block_2_to_2.weight)  #
        nn.init.constant_(self.conv_sa_coarse_to_fine_block_2_to_2.bias, 0)

        # Spatial Attention 2
        self.conv_sa_coarse_to_fine_block_3_to_3 = nn.Conv1d(c3_coarse, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa_coarse_to_fine_block_3_to_3.weight)  #
        nn.init.constant_(self.conv_sa_coarse_to_fine_block_3_to_3.bias, 0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        N, C, T1, V, M = x.size()
        x_in = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T1)
        x_in = self.data_bn(x_in)
        x_in = x_in.view(N * M, V, C, T1).permute(0, 2, 3, 1).contiguous()

        # STGC Block
        x_stgc_block = F.relu(self.sgcn1(x_in) + self.gcn3d1(x_in), inplace=True)
        x_stgc_block = self.tcn1(x_stgc_block)

        # Fine Block 1: Embedding
        x_fine_block_1 = self.embed_fine(x_stgc_block)

        # Coarse Block 1: Temporal Subsampling
        x_coarse_block_1 = x_stgc_block[:, :, ::2]

        # Fine Block 2
        x_fine_block_2 = F.relu(self.sgcn2_fine(x_fine_block_1) + self.gcn3d2_fine(x_fine_block_1), inplace=True)
        x_fine_block_2 = self.tcn2_fine(x_fine_block_2)

        # Temporal Attention 1
        # Fine-2-Coarse: Fine Block 2 output -> Coarse Block 1 output, SE-Net like Attention
        se_temporal = x_fine_block_2.mean(-1)
        se1_temporal = self.sigmoid(self.conv_ta_fine_to_coarse_block_2_to_1(se_temporal))
        x_coarse_block_1 = x_coarse_block_1 * se1_temporal.unsqueeze(-1) + x_coarse_block_1

        # Coarse Block 2
        x_coarse_block_2 = F.relu(self.sgcn2_coarse(x_coarse_block_1) + self.gcn3d2_coarse(x_coarse_block_1), inplace=True)
        x_coarse_block_2 = self.tcn2_coarse(x_coarse_block_2)  # N' C' T' V'

        # Spatial Attention 1
        # Coarse-2-Fine: Coarse Block 2 output -> Fine Block 2 output, SE-Net like Attention
        se_spatial = x_coarse_block_2.mean(-2)  # N' C' V'
        se1_spatial = self.sigmoid(self.conv_sa_coarse_to_fine_block_2_to_2(se_spatial))
        x_fine_block_2 = x_fine_block_2 * se1_spatial.unsqueeze(-2) + x_fine_block_2

        # Fine Block 3
        x_fine_block_3 = F.relu(self.sgcn3_fine(x_fine_block_2) + self.gcn3d3_fine(x_fine_block_2), inplace=True)
        x_fine_block_3 = self.tcn3_fine(x_fine_block_3)


        # Temporal Attention 2
        # Fine-2-Coarse: Fine Block 3 output -> Coarse Block 2 output, SE-Net like Attention
        se_temporal = x_fine_block_3.mean(-1)
        se1_temporal = self.sigmoid(self.conv_ta_fine_to_coarse_block_3_to_2(se_temporal))
        x_coarse_block_2 = x_coarse_block_2 * se1_temporal.unsqueeze(-1) + x_coarse_block_2


        # Coarse Block 3
        x_coarse_block_3 = F.relu(self.sgcn3_coarse(x_coarse_block_2) + self.gcn3d3_coarse(x_coarse_block_2), inplace=True)
        x_coarse_block_3 = self.tcn3_coarse(x_coarse_block_3)

        # Spatial Attention 2
        # Coarse-2-Fine: Coarse Block 3 output -> Fine Block 3 output, SE-Net like Attention
        se_spatial = x_coarse_block_3.mean(-2)  # N' C' V'
        se1_spatial = self.sigmoid(self.conv_sa_coarse_to_fine_block_3_to_3(se_spatial))
        x_fine_block_3 = x_fine_block_3 * se1_spatial.unsqueeze(-2) + x_fine_block_3 # N, C, T, V * N, C, 1, V

        # Fine Head Output
        out_fine = x_fine_block_3
        out_fine_channels = out_fine.size(1)
        out_fine = out_fine.view(N, M, out_fine_channels, -1)
        out_fine = out_fine.mean(3)  # Global Average Pooling (Spatial+Temporal)
        out_fine = out_fine.mean(1)  # Average pool number of bodies in the sequence

        out_fine = self.fc_fine(out_fine)

        # Coarse Head Output
        out_coarse = x_coarse_block_3
        out_coarse_channels = out_coarse.size(1)
        out_coarse = out_coarse.view(N, M, out_coarse_channels, -1)
        out_coarse = out_coarse.mean(3)  # Global Average Pooling (Spatial+Temporal)
        out_coarse = out_coarse.mean(1)  # Average pool number of bodies in the sequence

        out_coarse = self.fc_coarse(out_coarse)

        return out_fine, out_coarse


if __name__ == "__main__":
    # For debugging purposes
    import sys

    sys.path.append('')

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=9,
        num_g3d_scales=6,
        graph='graph_msg3d.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 1, 3, 300, 25, 2
    x = torch.randn(N, C, T, V, M)
    model.forward(x)

    print('Model total # params:', count_params(model))

