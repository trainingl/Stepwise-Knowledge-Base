import torch
import torch.nn as nn


# 2023_ICASSP_Embedding Enhanced MLP Enables Simple and Extensible Spatiotemporal Forecasting
class TimeMLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.15):
        super(TimeMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.BatchNorm2d(num_features=input_dim),
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=(1, 1), bias=True),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
    def forward(self, input_data):
        # input_data shape: (B, D, N, 1)
        """
        --------------------residual---------------------
        |                                               |
        x --- batchNorm --- fc --- relu --- dropout --- + ---
        """
        hidden = self.mlp(input_data)
        hidden = input_data + hidden
        return hidden
    

class TimeBlock(nn.Module):
    def __init__(self, hidden_dim, dropout, num_layer):
        super(TimeBlock, self).__init__()
        self.time_layers = nn.Sequential(
            *[TimeMLP(input_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout) for _ in range(num_layer)]
        )
        
    def forward(self, input_data):
        # input_data shape: (B, D, N, 1)
        hidden = self.time_layers(input_data)
        return hidden
    
    
class E2MLP(nn.Module):
    def __init__(self, num_nodes, input_dim, input_len, output_len, dropout, num_layer, num_block,
                 embed_dim, node_dim, temp_embed_dim, hidden_dim, time_of_day_size, day_of_week_size):
        super(E2MLP, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_len = input_len
        self.output_len = output_len
        self.dropout = dropout
        self.num_layer = num_layer
        self.num_block = num_block
        
        self.node_dim = node_dim
        self.temp_embed_dim = temp_embed_dim
        self.hidden_dim = hidden_dim
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.if_spatial = True
        self.if_time_of_day = True
        self.if_day_of_week = True
        
        # 1.data embeddings
        # 1.1 spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # 1.2 temporal embeddings
        if self.if_time_of_day:
            self.time_of_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.temp_embed_dim))
            nn.init.xavier_uniform_(self.time_of_day_emb)
        if self.if_day_of_week:
            self.day_of_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.temp_embed_dim))
            nn.init.xavier_uniform_(self.day_of_week_emb)
        # 1.3 feature embeddings
        self.feature_emb_layer = nn.Conv2d(in_channels=input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        
        # 2.encoder
        self.mix_layers = nn.ModuleList()
        emb_dim = self.node_dim * int(self.if_spatial) + 2 * self.temp_embed_dim * int(self.if_time_of_day)
        self.mix_layers.append(nn.Conv2d(self.embed_dim + emb_dim, self.hidden_dim, kernel_size=(1, 1), bias=True))
        for _ in range(1, self.num_block):
            self.mix_layers.append(nn.Conv2d(self.hidden_dim + emb_dim, self.hidden_dim, kernel_size=(1, 1), bias=True))
        
        self.encoders = nn.ModuleList()
        self.forecast_layers = nn.ModuleList()
        for _ in range(0, self.num_block):
            self.encoders.append(TimeBlock(hidden_dim=self.hidden_dim, dropout=self.dropout, num_layer=self.num_layer))
            self.forecast_layers.append(
                nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
            )
            
    def forward(self, input_data: torch.Tensor):
        # input_data shape: (B, T, N, D)
        # prepare data
        x = input_data[..., range(self.input_dim)]
        batch_size, _, num_nodes, _ = input_data.shape
        
        # 1. temporal embeddings
        if self.if_time_of_day:
            t_i_d_data = input_data[..., 1]   # (B, T, N)
            time_of_day_emb = self.time_of_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]  # (B, N, d)
            temp_tod_emb = time_of_day_emb.transpose(1, 2).unsqueeze(-1)  # (B, d, N, 1)
        else:
            temp_tod_emb = None
        if self.if_day_of_week:
            d_i_w_data = input_data[..., 2]   # (B, T, N)
            day_of_week_emb = self.day_of_week_emb[
                (d_i_w_data[:, -1, :]).type(torch.LongTensor)
            ] # (B, N, d)
            temp_dow_emb = day_of_week_emb.transpose(1, 2).unsqueeze(-1)  # (B, d, N, 1)
        else:
            temp_dow_emb = None
        # 2. spatial embeddings
        if self.if_spatial:
            # (N, d) -> (1, N, d) -> (B, N, d) -> (B, d, N) -> (B, d, N, 1)
            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
        else:
            node_emb = None
        # 3. feature embeddings
        # (B, T, N, d) -> (B, N, T, d) -> (B, N, T*d) ->(B, T*d, N) -> (B, T*d, N, 1)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        hidden = self.feature_emb_layer(x)
        
        # 4.encoding (skip connection)
        output = torch.zeros(batch_size, self.output_len, num_nodes, 1).to(input_data.device)
        for i in range(0, self.num_block):
            # 4.1 embedding mixed layer
            if self.if_spatial:
                hidden = torch.cat((hidden, node_emb), dim=1)
            if self.if_time_of_day:
                hidden = torch.cat((hidden, temp_tod_emb, temp_dow_emb), dim=1)
            hidden = self.mix_layers[i](hidden)
            # 4.2 multi-layer perceptron with residual links
            hidden = self.encoders[i](hidden)
            output += self.forecast_layers[i](hidden)
            
        return output
    
    
if __name__ == '__main__':
    model = E2MLP(
        num_nodes=170, 
        input_dim=3, 
        input_len=12, 
        output_len=12, 
        dropout=0.15, 
        num_layer=4, 
        num_block=3,
        embed_dim=128, 
        node_dim=32, 
        temp_embed_dim=32, 
        hidden_dim=256, 
        time_of_day_size=288, 
        day_of_week_size=7
    )
    x = torch.randn(32, 12, 170, 1)
    tod = torch.rand(32, 12, 170, 1)
    dow = torch.randint(0, 6, size=(32, 12, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)
    print("Output shape: ", model(x).shape)