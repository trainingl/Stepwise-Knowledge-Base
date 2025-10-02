import torch
import torch.nn as nn


# 2022_CIKM_Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)
        
    def forward(self, input_data: torch.Tensor):
        # input_data shape: (B, D, N, 1)
        """
        ---------------residual--------------------
        |                                         |
        x --- fc1 --- relu --- dropout --- fc2 -- + ---
        """
        hidden = self.fc2(self.dropout(self.activate(self.fc1(input_data))))
        hidden = input_data + hidden
        return hidden
    

class STID(nn.Module):
    def __init__(self, num_nodes, input_dim, embed_dim, input_len, output_len, num_layer, 
                 node_dim, temp_dim_tod, temp_dim_dow, time_of_day_size, day_of_week_size,
                 if_time_of_day, if_day_of_week, if_spatial):
        super(STID, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_len = input_len
        self.output_len = output_len
        self.num_layer = num_layer
        
        self.node_dim = node_dim
        self.temp_dim_tod = temp_dim_tod
        self.temp_dim_dow = temp_dim_dow
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.if_time_of_day = if_time_of_day
        self.if_day_of_week = if_day_of_week
        self.if_spatial = if_spatial
        
        # 1.data embeddings
        # 1.1 spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # 1.2 temporal embeddings
        if self.if_time_of_day:
            self.time_of_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_dim_tod))
            nn.init.xavier_uniform_(self.time_of_day_emb)
        if self.if_day_of_week:
            self.day_of_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_dim_dow))
            nn.init.xavier_uniform_(self.day_of_week_emb)
        # 1.3 feature embeddings
        self.feature_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True
        )
        # 2.encoding
        self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
            self.temp_dim_tod * int(self.if_time_of_day) + self.temp_dim_dow * int(self.if_day_of_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )
        # 3.regression
        self.regressor = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        
    def forward(self, input_data: torch.Tensor):
        # input_data shape: (B, T, N, C)
        x = input_data[..., range(self.input_dim)]
        batch_size, _, num_nodes, _ = x.shape
        
        # 1. temporal embeddings
        temp_emb = []
        if self.if_time_of_day:
            t_i_d_data = input_data[..., 1]  # (B, T, N)
            time_of_data_emb = self.time_of_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]  # (B, N, d)
            temp_emb.append(time_of_data_emb.transpose(1, 2).unsqueeze(-1))  # (B, d, N, 1)
        else:
            time_of_data_emb = None
        if self.if_day_of_week:
            d_i_w_data = input_data[..., 2]  # (B, T, N)
            day_of_week_emb = self.day_of_week_emb[d_i_w_data[:, -1, :].type(torch.LongTensor)]  # (B, N, d) 
            temp_emb.append(day_of_week_emb.transpose(1, 2).unsqueeze(-1))   # (B, d, N, 1)
        else:
            day_of_week_emb = None            
            
        # 2. spatial embeddings
        node_emb = []
        if self.if_spatial:
            # (N, d) -> (1, N, d) -> (B, N, d) -> (B, d, N) -> (B, d, N, 1)
            node_emb.append(
                self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            )
            
        # 3. feature embeddings
        x = x.transpose(1, 2).contiguous()
        # (B, N, T, d) -> (B, N, T*d) -> (B, T*d, N) -> (B, T*d, N, 1)
        x = x.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_serise_emb = self.feature_emb_layer(x)
        
        # 4.concate all embeddings along feature dimemsion
        hidden = torch.cat([time_serise_emb] + node_emb + temp_emb, dim=1)
        # encoding
        hidden = self.encoder(hidden)
        
        # 5.regression
        output = self.regressor(hidden)
        return output
    
    
if __name__ == "__main__":
    model = STID(
        num_nodes=170, 
        input_dim=3, 
        embed_dim=32, 
        input_len=12, 
        output_len=12, 
        num_layer=3, 
        node_dim=32, 
        temp_dim_tod=32, 
        temp_dim_dow=32, 
        time_of_day_size=288, 
        day_of_week_size=7,
        if_time_of_day=True, 
        if_day_of_week=True, 
        if_spatial=True
    )
    x = torch.randn(32, 12, 170, 1)
    tod = torch.rand(32, 12, 170, 1)
    dow = torch.randint(0, 6, size=(32, 12, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)
    print("Output shape: ", model(x).shape)