import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from utils.utils import *


# 2025_INS_STEMLP: A spatial-temporal embedding multi-layer perceptron for traffic flow prediction
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=(1, 1), bias=True)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)
        
    def forward(self, input_data: torch.Tensor):
        # input_data shape: (B, D, N, 1)
        """
        --------------------------------------------
        |                                          |
        x --- fc1 --- relu --- dropout --- fc2 --- + --- 
        """
        hidden = self.fc2(self.dropout(self.activate(self.fc1(input_data))))
        hidden = input_data + hidden
        return hidden
    
    
class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super(LaplacianPE, self).__init__()
        self.lape_dim = lape_dim
        # self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)
    
    def forward(self, lap_pos_enc):
        # lap_pe shape: (N, lape_dim)
        # lap_pos_enc = self.embedding_lap_pos_enc(lap_pos_enc)
        # (N, d) -> (1, N, d, 1) -> (1, d, N, 1)
        lap_pos_enc = lap_pos_enc.unsqueeze(0).unsqueeze(-1).transpose(1, 2)
        return lap_pos_enc


class STEMLP(nn.Module):
    def __init__(self, num_nodes, input_dim, embed_dim, input_len, output_len, node_dim, lape_dim, adp_dim,
                 num_layerA, num_layerB, num_layerC, adj_mx, time_of_day_size, day_of_week_size,
                 if_time_of_day=True, if_day_of_week=True, if_pre_spatial=True, if_adp_spatial=True):
        super(STEMLP, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_len = input_len
        self.output_len = output_len
        self.node_dim = node_dim
        self.lape_dim = lape_dim
        self.adp_dim = adp_dim
        self.num_layerA = num_layerA
        self.num_layerB = num_layerB
        self.num_layerC = num_layerC
        self.lap_mx = self._calculate_laplacian_pe(adj_mx)
        
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.if_time_of_day = if_time_of_day
        self.if_day_of_week = if_day_of_week
        self.if_pre_spatial = if_pre_spatial
        self.if_adp_spatial = if_adp_spatial
        
        # 1.data embeddings
        # 1.1 spatial embeddings
        if self.if_adp_spatial:  # adaptive spatial embeddings
            self.node_embeddings1 = nn.Parameter(torch.randn(self.num_nodes, self.adp_dim), requires_grad=True)
            self.node_embeddings2 = nn.Parameter(torch.randn(self.adp_dim, self.num_nodes), requires_grad=True)
            self.adp_spatial_embedding = LaplacianPE(self.lape_dim, self.lape_dim)
        if self.if_pre_spatial:  # predefine spatial embeddings
            self.pre_spatial_embedding = LaplacianPE(self.lape_dim, self.lape_dim)
        # 1.2 temporal embeddings
        if self.if_time_of_day:
            self.time_of_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.node_dim))
            nn.init.xavier_uniform_(self.time_of_day_emb)
        if self.if_day_of_week:
            self.day_of_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.node_dim))
            nn.init.xavier_uniform_(self.day_of_week_emb)
        # 1.3 feature embeddings
        self.feature_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True
        )
        
        # 2.MLP encoder
        self.hidden_dim = 0
        if self.if_time_of_day or self.if_day_of_week:
            self.data_time_hidden_dim = self.embed_dim + self.node_dim * int(self.if_time_of_day) + self.node_dim * int(self.if_day_of_week)
            self.data_time_encoder = nn.Sequential(
                *[MultiLayerPerceptron(self.data_time_hidden_dim, int(self.data_time_hidden_dim * 1.5)) for _ in range(self.num_layerA)]
            )
            self.hidden_dim += self.data_time_hidden_dim
        if self.if_adp_spatial or self.if_pre_spatial:
            self.data_spatial_hidden_dim = self.embed_dim + self.lape_dim * int(self.if_pre_spatial) + self.lape_dim * int(self.if_adp_spatial)
            self.data_spatial_encoder = nn.Sequential(
                *[MultiLayerPerceptron(self.data_spatial_hidden_dim, int(self.data_spatial_hidden_dim * 1.5)) for _ in range(self.num_layerB)]
            )
            self.hidden_dim += self.data_spatial_hidden_dim
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, int(self.hidden_dim * 1.5)) for _ in range(self.num_layerC)]
        )
        
        # 3.regression
        self.regression_layer = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True) 
    
    def _calculate_normalized_laplacian(self, adj_mx):
        adj_mx = sp.coo_matrix(adj_mx)
        d = np.array(adj_mx.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj_mx.shape[0]) - adj_mx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian, isolated_point_num
    
    def _calculate_laplacian_pe(self, adj_mx):
        L, isolated_point_num = self._calculate_normalized_laplacian(adj_mx)
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

        laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: self.lape_dim + isolated_point_num + 1]).float()
        laplacian_pe.require_grad = False
        return laplacian_pe
    
    def _calculate_sym_normalized_laplacian(self, adj):
        adj = adj.masked_fill(torch.isinf(adj), 0)
        adj = adj.masked_fill(torch.isnan(adj), 0)
        degree_matrix = torch.sum(adj, dim=1, keepdim=False)
        isolated_point_num = torch.sum(degree_matrix == 0).item()
        if isolated_point_num == self.num_nodes:
            isolated_point_num = isolated_point_num - self.lape_dim - 1
        EigVal, EigVec = torch.linalg.eigh(adj)
        idx = torch.argsort(torch.abs(EigVal), dim=0, descending=True)
        EigVal, EigVec = EigVal[idx], EigVec[:, idx]
        laplacian_pe = EigVec[:, isolated_point_num + 1: self.lape_dim + isolated_point_num + 1]
        return laplacian_pe
    
    def forward(self, input_data: torch.Tensor):
        # input_data shape: (B, T, N, D)
        # prepare data
        x = input_data[..., range(self.input_dim)]
        batch_size, _, num_nodes, _ = input_data.shape
        
        # 1. temporal embeddings
        temp_emb = []
        if self.if_time_of_day:
            t_i_d_data = input_data[..., 1]  # (B, T, N)
            time_of_day_emb = self.time_of_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]  # (B, N, d)
            temp_tod_emb = time_of_day_emb.transpose(1, 2).unsqueeze(-1)  # (B, d, N, 1)
            temp_emb.append(temp_tod_emb)
        if self.if_day_of_week:
            d_i_w_data = input_data[..., 2]  # (B, T, N)
            day_of_week_emb = self.day_of_week_emb[
                (d_i_w_data[:, -1, :]).type(torch.LongTensor)
            ] # (B, N, d)
            temp_dow_emb = day_of_week_emb.transpose(1, 2).unsqueeze(-1)  # (B, d, N, 1)
            temp_emb.append(temp_dow_emb)
        # 2. spatial embeddings
        node_emb = []
        if self.if_adp_spatial:
            adp_mx = F.softmax(F.relu(torch.mm(self.node_embeddings1, self.node_embeddings2)), dim=1)
            adp_lap_pe = self._calculate_sym_normalized_laplacian(adp_mx)  #  (N, d) -> (1, d, N, 1)
            adp_spatial_emb = self.adp_spatial_embedding(adp_lap_pe).expand(batch_size, -1, -1, -1)
            node_emb.append(adp_spatial_emb)
        if self.if_pre_spatial:
            self.lap_mx = self.lap_mx.to(input_data.device)
            #  (N, d) -> (1, d, N, 1)
            pre_spatial_emb = self.pre_spatial_embedding(self.lap_mx).expand(batch_size, -1, -1, -1)
            node_emb.append(pre_spatial_emb)
        # 3. feature embeddings
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.feature_emb_layer(x)
        
        # 4. spatial encoder and temporal encoder
        if self.if_adp_spatial or self.if_pre_spatial:
            s_emb = torch.cat([time_series_emb] + node_emb, dim=1)
            ds_emb = self.data_spatial_encoder(s_emb)
        if self.if_time_of_day or self.if_day_of_week:
            t_emb = torch.cat([time_series_emb] + temp_emb, dim=1)
            dt_emb = self.data_time_encoder(t_emb)
        hidden = torch.cat([ds_emb, dt_emb], dim=1)
        hidden = self.encoder(hidden)
        
        # 5. regression
        output = self.regression_layer(hidden)
        return output
    
    
if __name__ == '__main__':
    adj_mx = load_pickle("./data/adj_PEMS08.pkl")
    model = STEMLP(
        num_nodes=170, 
        input_dim=3, 
        embed_dim=96, 
        input_len=12, 
        output_len=12, 
        node_dim=32, 
        lape_dim=32, 
        adp_dim=8,
        num_layerA=3, 
        num_layerB=3, 
        num_layerC=1, 
        adj_mx=adj_mx,
        time_of_day_size=288, day_of_week_size=7,
        if_time_of_day=True, if_day_of_week=True, 
        if_pre_spatial=True, if_adp_spatial=True
    )
    x = torch.randn(32, 12, 170, 1)
    tod = torch.rand(32, 12, 170, 1)
    dow = torch.randint(0, 6, size=(32, 12, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)
    print("Output shape: ", model(x).shape)