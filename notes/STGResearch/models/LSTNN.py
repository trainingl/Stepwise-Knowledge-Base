import torch
import torch.nn as nn
from einops import rearrange
import math


# 2025_TITS_A Lightweight Spatio-Temporal Neural Network with Sampling-based Time Series Decomposition for Traffic Forecasting
# Note: this paper utilizes traffic data from the past 48 time steps to predict traffic flow for the next 12 time steps
class BasicUnitBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BasicUnitBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)
        self.layerNorm = nn.LayerNorm(hidden_dim)
    
    def forward(self, input_data):
        # Multi-Layer Perceptron with Residual Links,
        # x shape: (B, D, N, T)
        hidden = self.fc2(self.dropout(self.activate(self.fc1(input_data))))
        hidden = self.layerNorm((input_data + hidden).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return hidden
    

class SamplingBranchEncoder(nn.Module):
    def __init__(self, tod_embeddings, dow_embeddings, spatial_embeddings, if_time_of_day, if_day_of_week, if_spatial,
                 time_of_day_size, input_dim, patch_len, stride, embed_dim, tod_dim, dow_dim, spatial_dim, output_len, num_layers):
        super(SamplingBranchEncoder, self).__init__()
        self.tod_embeddings = tod_embeddings
        self.dow_embeddings = dow_embeddings
        self.spatial_embeddings = spatial_embeddings
        self.if_time_of_day = if_time_of_day
        self.if_day_of_week = if_day_of_week
        self.if_spatial = if_spatial
        self.time_of_day_size = time_of_day_size
        self.stride = stride
        self.output_len = output_len
        
        self.data_embedding_layer = nn.Conv2d(
            in_channels=input_dim * patch_len, out_channels=embed_dim, kernel_size=(1, 1), bias=True
        )
        self.data_encoder = nn.Sequential(
            *[BasicUnitBlock(embed_dim, embed_dim) for _ in range(num_layers)]
        )
        self.hidden_dim = embed_dim + int(self.if_spatial) * spatial_dim
        self.spatial_encoder = nn.Sequential(                              # 1. add spatial emb
            *[BasicUnitBlock(self.hidden_dim, self.hidden_dim) for _ in range(num_layers)]
        )
        self.fusion_dim = self.hidden_dim + 2 * int(self.if_time_of_day) * tod_dim + 2 * int(self.if_day_of_week) * dow_dim 
        self.temporal_encoder = nn.Sequential(                              # 2. add temporal emb (e.g., time of day, day of week) 
            *[BasicUnitBlock(self.fusion_dim, self.fusion_dim) for _ in range(num_layers)]
        )
        self.proj = nn.Conv2d(                                              # 3. add future temporal emb
            in_channels=self.stride * self.fusion_dim + tod_dim + dow_dim, 
            out_channels=output_len,
            kernel_size=(1, 1),
            bias=True
        )

    def forward(self, patch_data):
        """
            patch_data shape: (B, P, L, N, C)
            P: the number of patch data, P = stride
            L: the length of patch data, L = patch_len
        """
        batch_size, num_patch, _, _, _ = patch_data.shape

        # Temporal Embedding
        if self.if_time_of_day:
            tod_data = patch_data[..., 1]    # (B, P, L, N)
            tod_start_data = self.tod_embeddings[(tod_data[:, :, 0, :] * self.time_of_day_size).type(torch.LongTensor)]  # (B, P, N, tod_dim)
            tod_end_data = self.tod_embeddings[(tod_data[:, :, -1, :] * self.time_of_day_size).type(torch.LongTensor)]   # (B, P, N, tod_dim)
            # (B, N) -> (B, N, tod_dim)
            future_tod_emb = self.tod_embeddings[((tod_data[:, -1, -1, :] * self.time_of_day_size + self.output_len) % self.time_of_day_size).type(torch.LongTensor)]
            future_tod_emb = future_tod_emb.permute(0, 2, 1).unsqueeze(-1)              # (B, tod_dim, N, 1)
        else:
            tod_start_data, tod_end_data, future_tod_emb = None, None, None

        if self.if_day_of_week:
            dow_data = patch_data[..., 2]    # (B, P, L, N)
            dow_start_emb = self.dow_embeddings[(dow_data[:, :, 0, :]).type(torch.LongTensor)]  # (B, P, N, dow_dim)
            dow_end_emb = self.dow_embeddings[((dow_data[:, :, -1, :]).type(torch.LongTensor))] # (B, P, N, dow_dim)
            future_dow_data = dow_end_emb[:, -1, :, :].permute(0, 2, 1).unsqueeze(-1)   # (B, dow_dim, N, 1)
        else:
            dow_start_emb, dow_end_emb, future_dow_data = None, None, None

        # Spatial Embedding
        if self.if_spatial:
            # (B, P, N, spatial_dim)
            spatial_emb = self.spatial_embeddings.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).expand(-1, num_patch, -1, -1)
        else:
            spatial_emb = None
        
        # 1.data encoding
        input_data = torch.cat((patch_data[..., 0], patch_data[..., 1], patch_data[..., 2]), dim=2)   # (B, P, input_dim * patch_len, N)
        data_emb = self.data_embedding_layer(input_data.transpose(1, 2))   # (B, D, P, N)
        data_emb = self.data_encoder(data_emb).permute(0, 2, 3, 1)   # (B, P, N, D)

        # 2.spatial encoding
        hidden = torch.cat((data_emb, spatial_emb), dim=-1).permute(0, 3, 1, 2)
        hidden = self.spatial_encoder(hidden).permute(0, 2, 3, 1)    # (B, P, N, D)

        # 3.temporal encoding
        hidden = torch.cat((tod_start_data, dow_start_emb, hidden, tod_end_data, dow_end_emb), dim=-1).permute(0, 3, 1, 2)
        hidden = self.temporal_encoder(hidden)   # (B, D, P, N)

        hidden = rearrange(hidden, 'B D P N -> B (D P) N').unsqueeze(-1)  # (B, D * P, N, 1), P = stride
        hidden = torch.cat((hidden, future_tod_emb, future_dow_data), dim=1)
        out = self.proj(hidden)
        return out
    

class LSTNN(nn.Module):
    def __init__(self, num_nodes, input_dim, embed_dim, tod_dim, dow_dim, spatial_dim, input_len, output_len, patch_len,
                 stride, time_of_day_size, day_of_week_size, if_time_of_day, if_day_of_week, if_spatial, num_layers):
        super(LSTNN, self).__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_layers = num_layers

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.tod_dim = tod_dim
        self.dow_dim = dow_dim
        self.spatial_dim = spatial_dim
        self.tod_size = time_of_day_size
        self.dow_size = day_of_week_size
        self.if_time_of_day = if_time_of_day
        self.if_day_of_week = if_day_of_week
        self.if_spatial = if_spatial

        # 1. temporal embeddings
        if self.if_time_of_day:
            self.tod_embeddings = nn.Parameter(torch.empty(self.tod_size, self.tod_dim))
            nn.init.xavier_uniform_(self.tod_embeddings)
        if self.if_day_of_week:
            self.dow_embeddings = nn.Parameter(torch.empty(self.dow_size, self.dow_dim))
            nn.init.xavier_uniform_(self.dow_embeddings)
        
        # 2. spatial embeddings
        if self.if_spatial:
            self.node_embeddings = nn.Parameter(torch.empty(self.num_nodes, self.spatial_dim))
            nn.init.xavier_uniform_(self.node_embeddings)

        # 3. encoder
        self.patch_encoder = SamplingBranchEncoder(
            self.tod_embeddings, self.dow_embeddings, self.node_embeddings, if_time_of_day, if_day_of_week, if_spatial,
            time_of_day_size, input_dim, patch_len, stride, embed_dim, tod_dim, dow_dim, spatial_dim, output_len, num_layers
        )
        self.downsample_encoder = SamplingBranchEncoder(
            self.tod_embeddings, self.dow_embeddings, self.node_embeddings, if_time_of_day, if_day_of_week, if_spatial,
            time_of_day_size, input_dim, patch_len, stride, embed_dim, tod_dim, dow_dim, spatial_dim, output_len, num_layers
        )

        # 4. residual
        self.residual = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        # x shape: (B, T, N, C)
        # prepare data
        input_data = x[..., range(self.input_dim)]

        # 1. padding
        input_len_pad = math.ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if input_len_pad:
            input_data = torch.cat((input_data, input_data[:, -1:, :, :].expand(-1, input_len_pad, -1, -1)), dim=1)

        # 2. downsampling 下采样
        downsample_input = [input_data[:, i::self.stride, :, :] for i in range(self.stride)]
        downsample_input = torch.stack(downsample_input, dim=1)

        # 3. patchsampling 分段采样
        patch_input = input_data.unfold(dimension=1, size=self.patch_len, step=self.patch_len).permute(0, 1, 4, 2, 3)
        # print(patch_input.shape, downsample_input.shape)
        patch_out = self.patch_encoder(patch_input)
        downsample_out = self.downsample_encoder(downsample_input)

        # 4. residual & add
        output = patch_out + downsample_out + self.residual(input_data)
        return output


if __name__ == '__main__':
    model = LSTNN(
        num_nodes=170, 
        input_dim=3, 
        embed_dim=32, 
        tod_dim=32, 
        dow_dim=32, 
        spatial_dim=32, 
        input_len=48, 
        output_len=12, 
        patch_len=12,
        stride=4, 
        time_of_day_size=288, 
        day_of_week_size=7, 
        if_time_of_day=True, 
        if_day_of_week=True, 
        if_spatial=True, 
        num_layers=4
    )
    x = torch.randn(32, 48, 170, 1)
    tod = torch.rand(32, 48, 170, 1)
    dow = torch.randint(0, 6, size=(32, 48, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)
    print("Output shape: ", model(x).shape)