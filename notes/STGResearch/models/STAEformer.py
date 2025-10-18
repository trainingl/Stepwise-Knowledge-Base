import torch
import torch.nn as nn


# 2023_CIKM_STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads, mask=False):
        super(AttentionLayer, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        assert model_dim % num_heads == 0, 'model_dim must be divisible by num_heads'
        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        """
            query shape: (batch_size, ..., tgt_length, model_dim)
            key & value shape: (batch_size, ..., src_length, model_dim)
            Attention(Q, K, V) = softmax(QK^(T) / sqrt(d_k))V
        """
        batch_size, tgt_length, src_length = query.shape[0], query.shape[-2], key.shape[-2]
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Multi-Heads: (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)

        attn_score = (query @ key) / self.head_dim ** 0.5
        # apply causal mask
        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular matrix
            attn_score.masked_fill_(~mask, -torch.inf)
        attn_score = torch.softmax(attn_score, dim=-1)  # (num_heads * batch_size, ..., tgt_length, src_length)
        
        out = attn_score @ value   # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        return out

        
class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0., mask=False):
        super(SelfAttentionLayer, self).__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim)
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        # x shape: (batch_size, ..., seq_length, model_dim)
        """
        ----------- residual ---------------                  ----------- residual -------------
        |                                  |                  |                                |
        x ----- attention ----- dropout -- + ---- layernorm ----- feedforward ----- dropout -- + ----- layernorm ---
        """
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        out = out.transpose(dim, -2)
        return out
    

class STAEformer(nn.Module):
    def __init__(self, num_nodes, in_steps=12, out_steps=12, steps_per_day=288, input_dim=3, output_dim=1,
                 input_embedding_dim=24, tod_embedding_dim=24, dow_embedding_dim=24, spatial_embedding_dim=0,
                 adaptive_embedding_dim=80, feed_forward_dim=256, num_heads=4, num_layers=3, dropout=0.1, use_mixed_proj=True):
        super(STAEformer, self).__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        
        # 1. spatial-temporal embeddings
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if self.tod_embedding_dim > 0:
            self.tod_embeddings = nn.Embedding(steps_per_day, tod_embedding_dim)
        if self.dow_embedding_dim > 0:
            self.dow_embeddings = nn.Embedding(7, dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.node_embeddings = nn.Parameter(torch.empty(self.num_nodes, self.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_embeddings)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embeddings = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        
        # 2. spatial-temporal attention layers
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # 3. regressor to predict
        if use_mixed_proj:
            self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * self.output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

    def forward(self, x):
        # x shape: (batch_size, in_steps, num_nodes, 3)
        batch_size = x.shape[0]
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., :self.input_dim]

        x = self.input_proj(x)   # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features_list = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embeddings((tod * self.steps_per_day).long())
            # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features_list.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embeddings(dow.type(torch.LongTensor))    
            # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features_list.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_embeddings.expand(
                batch_size, self.in_steps, *self.node_embeddings.shape
            )  # (batch_size, in_steps, num_nodes, spatial_embedding_dim)
            features_list.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embeddings.expand(
                batch_size, *self.adaptive_embeddings.shape
            )  # (batch_size, in_steps, num_nodes, adaptive_embedding_dim)
            features_list.append(adp_emb)
        x = torch.cat(features_list, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(out) 
            out = out.transpose(1, 3)
            out = self.output_proj(out)
        # out shape: (batch_size, out_steps, num_nodes, output_dim)
        return out
    

if __name__ == '__main__':
    model = STAEformer(
        num_nodes=170, 
        in_steps=12, 
        out_steps=12, 
        steps_per_day=288, 
        input_dim=3, 
        output_dim=1,
        input_embedding_dim=24, 
        tod_embedding_dim=24, 
        dow_embedding_dim=24, 
        spatial_embedding_dim=0,         
        adaptive_embedding_dim=80, 
        feed_forward_dim=256, 
        num_heads=4, 
        num_layers=3, 
        dropout=0.1, use_mixed_proj=True
    )
    x = torch.randn(32, 12, 170, 1)
    tod = torch.rand(32, 12, 170, 1)
    dow = torch.randint(0, 6, size=(32, 12, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)
    print("Output shape: ", model(x).shape)