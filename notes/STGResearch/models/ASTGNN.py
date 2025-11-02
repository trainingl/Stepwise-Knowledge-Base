import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 2022_TKDE_Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for Traffic Forecasting
class GCN(nn.Module):
    def __init__(self, norm_adj, input_dim, output_dim):
        super(GCN, self).__init__()
        self.norm_adj = norm_adj
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Theta = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        # spatial graph convolution operation
        # x shape: (B, N, D)
        return F.relu(self.Theta(torch.matmul(self.norm_adj, x)))


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_nodes, norm_adj, dropout, gcn_num_layers=0):
        super(SpatialPositionalEncoding, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.gcn_layers = nn.ModuleList(
            [
                GCN(norm_adj, d_model, d_model)
                for _ in range(gcn_num_layers)
            ]
        )

    def forward(self, x):
        # x shape: (B, N, T, d_model)
        num_nodes = x.shape[1]
        idxs = torch.LongTensor(torch.arange(num_nodes)).to(x.device)
        embed = self.node_embeddings(idxs).unsqueeze(0)   # (N, d_model) -> (1, N, d_model)
        for gcn in self.gcn_layers:
            embed = gcn(embed)
        x = x + embed.unsqueeze(2)  # (B, N, T, d_model) + (1, N, 1, d_model)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_head == 0
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.FC_Q = nn.Linear(d_model, d_model)
        self.FC_K = nn.Linear(d_model, d_model)
        self.FC_V = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        """
            query/key/value shape: (B, N, T, d_model)
            mask shape: (B, T, T)
        """
        batch_size, num_nodes, _, _ = query.shape
        query = self.FC_Q(query).view(batch_size, num_nodes, -1, self.num_head, self.head_dim).transpose(2, 3)  # (B, N, H, T, dk)
        key = self.FC_K(key).view(batch_size, num_nodes, -1, self.num_head, self.head_dim).transpose(2, 3)      # (B, N, H, T, dk)
        value = self.FC_V(value).view(batch_size, num_nodes, -1, self.num_head, self.head_dim).transpose(2, 3)  # (B, N, H, T, dk)

        # calcuate attention scores
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, N, H, T, T)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T, T)
            scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores equal to 0
        attn_score = F.softmax(scores, dim=-1)     # (B, N, H, T, T)
        if self.dropout is not None:
            attn_score = self.dropout(attn_score)
        
        output = torch.matmul(attn_score, value)      #  (B, N, H, T, dk)
        output = output.transpose(2, 3).contiguous()  #  (B, N, T, H, dk)
        output = output.view(batch_size, num_nodes, -1, self.num_head * self.head_dim)  # (B, N, T, d_model)
        return self.proj(output)
    

class MultiHeadAttentionAwareTemporalContext_q1d_k1d(nn.Module):
    def __init__(self, num_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContext_q1d_k1d, self).__init__()
        assert d_model % num_head == 0
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.fc = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        
        self.padding = (kernel_size - 1) // 2
        self.temporal_conv2D_key = nn.Conv2d(d_model, d_model, kernel_size=(1, kernel_size), padding=(0, self.padding))
        self.temporal_conv2D_query = nn.Conv2d(d_model, d_model, kernel_size=(1, kernel_size), padding=(0, self.padding))
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        """
            query/key/value shape: (B, N, T, d_model)
            mask shape: (B, T, T)
            query_multi_segment: whether query has mutiple time segments
            key_multi_segment: whether key has mutiple time segments
            if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        """
        batch_size, num_nodes, _, _ = query.shape

        if self.w_length > 0:
            # w_query: (B, d_model, N, T), w_key: (B, d_model, N, T)
            w_query, w_key = query[:, :, :self.w_length, :].permute(0, 3, 1, 2), key[:, :, :self.w_length, :].permute(0, 3, 1, 2)
        if self.d_length > 0:
            d_query = query[:, :, self.w_length : self.w_length + self.d_length, :].permute(0, 3, 1, 2)
            d_key = key[:, :, self.w_length : self.w_length + self.d_length, :].permute(0, 3, 1, 2)
        if self.h_length > 0:
            h_query = query[:, :, self.w_length + self.d_length : self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)
            h_key = key[:, :, self.w_length + self.d_length : self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                # (B, d_model, N, T) -> (B, num_head, head_dim, N, T) -> (B, N, num_head, T, head_dim)
                w_query = self.temporal_conv2D_query(w_query).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                w_key = self.temporal_conv2D_key(w_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                query_list.append(w_query)
                key_list.append(w_key)

            if self.d_length > 0:
                # (B, d_model, N, T) -> (B, num_head, head_dim, N, T) -> (B, N, num_head, T, head_dim)
                d_query = self.temporal_conv2D_query(d_query).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                d_key = self.temporal_conv2D_key(d_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                query_list.append(d_query)
                key_list.append(d_key)

            if self.h_length > 0:
                # (B, d_model, N, T) -> (B, num_head, head_dim, N, T) -> (B, N, num_head, T, head_dim)
                h_query = self.temporal_conv2D_query(h_query).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                h_key = self.temporal_conv2D_key(h_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                query_list.append(h_query)
                key_list.append(h_key)
            
            query = torch.cat(query_list, dim=-2)
            key = torch.cat(key_list, dim=-2)

        elif (not query_multi_segment) and (not key_multi_segment):
            query = self.temporal_conv2D_query(query.permute(0, 3, 1, 2)).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
            key = self.temporal_conv2D_key(key.permute(0, 3, 1, 2)).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)

        elif (not query_multi_segment) and (key_multi_segment):
            query = self.temporal_conv2D_query(query.permute(0, 3, 1, 2)).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
            key_list = []
            if self.w_length > 0:
                w_key = self.temporal_conv2D_key(w_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                key_list.append(w_key)
            
            if self.d_length > 0:
                d_key = self.temporal_conv2D_key(d_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                key_list.append(d_key)

            if self.h_length > 0:
                h_key = self.temporal_conv2D_key(h_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                key_list.append(h_key)

            key = torch.cat(key_list, dim=-2)

        # (B, N, num_head, T, head_dim)
        value = self.fc(value).view(batch_size, num_nodes, -1, self.num_head, self.head_dim).transpose(2, 3)
        attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, N, H, T, T)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)   # (B, 1, 1, T, T)
            attn_score = attn_score.masked_fill_(mask == 0, -1e9)
        attn_score = F.softmax(attn_score, dim=-1)  # (B, N, H, T, T)
        if self.dropout is not None:
            attn_score = self.dropout(attn_score)
        
        output = torch.matmul(attn_score, value)
        output = output.transpose(2, 3).contiguous()  #  (B, N, T, H, dk)
        output = output.view(batch_size, num_nodes, -1, self.num_head * self.head_dim)  # (B, N, T, d_model)
        return self.proj(output)
    

class MultiHeadAttentionAwareTemporalContext_qc_k1d(nn.Module):
    def __init__(self, num_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContext_qc_k1d, self).__init__()
        assert d_model % num_head == 0
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.fc = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1) // 2
        self.temporal_conv2D_query = nn.Conv2d(d_model, d_model, kernel_size=(1, kernel_size), padding=(0, self.causal_padding))
        self.temporal_conv2D_key = nn.Conv2d(d_model, d_model, kernel_size=(1, kernel_size), padding=(0, self.padding_1D))
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        """
            query/key/value shape: (B, N, T, d_model)
            mask shape: (B, T, T)
            query_multi_segment: whether query has mutiple time segments
            key_multi_segment: whether key has mutiple time segments
            if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        """
        batch_size, num_nodes, _, _ = query.shape

        if self.w_length > 0:
            # w_query: (B, d_model, N, T), w_key: (B, d_model, N, T)
            w_query, w_key = query[:, :, :self.w_length, :].permute(0, 3, 1, 2), key[:, :, :self.w_length, :].permute(0, 3, 1, 2)
        if self.d_length > 0:
            d_query = query[:, :, self.w_length : self.w_length + self.d_length, :].permute(0, 3, 1, 2)
            d_key = key[:, :, self.w_length : self.w_length + self.d_length, :].permute(0, 3, 1, 2)
        if self.h_length > 0:
            h_query = query[:, :, self.w_length + self.d_length : self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)
            h_key = key[:, :, self.w_length + self.d_length : self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                # (B, d_model, N, T) -> (B, num_head, head_dim, N, T) -> (B, N, num_head, T, head_dim)
                w_query = self.temporal_conv2D_query(w_query)[:, :, :, :-self.causal_padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                w_key = self.temporal_conv2D_key(w_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                query_list.append(w_query)
                key_list.append(w_key)

            if self.d_length > 0:
                # (B, d_model, N, T) -> (B, num_head, head_dim, N, T) -> (B, N, num_head, T, head_dim)
                d_query = self.temporal_conv2D_query(d_query)[:, :, :, :-self.causal_padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                d_key = self.temporal_conv2D_key(d_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                query_list.append(d_query)
                key_list.append(d_key)

            if self.h_length > 0:
                # (B, d_model, N, T) -> (B, num_head, head_dim, N, T) -> (B, N, num_head, T, head_dim)
                h_query = self.temporal_conv2D_query(h_query)[:, :, :, :-self.causal_padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                h_key = self.temporal_conv2D_key(h_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                query_list.append(h_query)
                key_list.append(h_key)
            
            query = torch.cat(query_list, dim=-2)
            key = torch.cat(key_list, dim=-2)

        elif (not query_multi_segment) and (not key_multi_segment):
            query = self.temporal_conv2D_query(query.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
            key = self.temporal_conv2D_key(key.permute(0, 3, 1, 2)).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)

        elif (not query_multi_segment) and (key_multi_segment):
            query = self.temporal_conv2D_query(query.permute(0, 3, 1, 2))[:, :, :, :-self.causal_padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
            key_list = []
            if self.w_length > 0:
                w_key = self.temporal_conv2D_key(w_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                key_list.append(w_key)
            
            if self.d_length > 0:
                d_key = self.temporal_conv2D_key(d_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                key_list.append(d_key)

            if self.h_length > 0:
                h_key = self.temporal_conv2D_key(h_key).contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                key_list.append(h_key)

            key = torch.cat(key_list, dim=-2)

        # (B, N, num_head, T, head_dim)
        value = self.fc(value).view(batch_size, num_nodes, -1, self.num_head, self.head_dim).transpose(2, 3)
        attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, N, H, T, T)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)   # (B, 1, 1, T, T)
            attn_score = attn_score.masked_fill_(mask == 0, -1e9)
        attn_score = F.softmax(attn_score, dim=-1)  # (B, N, H, T, T)
        if self.dropout is not None:
            attn_score = self.dropout(attn_score)
        
        output = torch.matmul(attn_score, value)
        output = output.transpose(2, 3).contiguous()  #  (B, N, T, H, dk)
        output = output.view(batch_size, num_nodes, -1, self.num_head * self.head_dim)  # (B, N, T, d_model)
        return self.proj(output)
    

class MultiHeadAttentionAwareTemporalContext_qc_kc(nn.Module):
    def __init__(self, num_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContext_qc_kc, self).__init__()
        assert d_model % num_head == 0
        self.num_head = num_head
        self.head_dim = d_model // num_head
        self.fc = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        
        self.padding = kernel_size - 1
        self.temporal_conv2D_query = nn.Conv2d(d_model, d_model, kernel_size=(1, kernel_size), padding=(0, self.padding))
        self.temporal_conv2D_key = nn.Conv2d(d_model, d_model, kernel_size=(1, kernel_size), padding=(0, self.padding))
        self.dropout = nn.Dropout(p=dropout)
        self.w_length = num_of_weeks * points_per_hour
        self.d_length = num_of_days * points_per_hour
        self.h_length = num_of_hours * points_per_hour

    def forward(self, query, key, value, mask=None, query_multi_segment=False, key_multi_segment=False):
        """
            query/key/value shape: (B, N, T, d_model)
            mask shape: (B, T, T)
            query_multi_segment: whether query has mutiple time segments
            key_multi_segment: whether key has mutiple time segments
            if query/key has multiple time segments, causal convolution should be applied separately for each time segment.
        """
        batch_size, num_nodes, _, _ = query.shape

        if self.w_length > 0:
            # w_query: (B, d_model, N, T), w_key: (B, d_model, N, T)
            w_query, w_key = query[:, :, :self.w_length, :].permute(0, 3, 1, 2), key[:, :, :self.w_length, :].permute(0, 3, 1, 2)
        if self.d_length > 0:
            d_query = query[:, :, self.w_length : self.w_length + self.d_length, :].permute(0, 3, 1, 2)
            d_key = key[:, :, self.w_length : self.w_length + self.d_length, :].permute(0, 3, 1, 2)
        if self.h_length > 0:
            h_query = query[:, :, self.w_length + self.d_length : self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)
            h_key = key[:, :, self.w_length + self.d_length : self.w_length + self.d_length + self.h_length, :].permute(0, 3, 1, 2)

        if query_multi_segment and key_multi_segment:
            query_list = []
            key_list = []
            if self.w_length > 0:
                # (B, d_model, N, T) -> (B, num_head, head_dim, N, T) -> (B, N, num_head, T, head_dim)
                w_query = self.temporal_conv2D_query(w_query)[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                w_key = self.temporal_conv2D_key(w_key)[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                query_list.append(w_query)
                key_list.append(w_key)

            if self.d_length > 0:
                # (B, d_model, N, T) -> (B, num_head, head_dim, N, T) -> (B, N, num_head, T, head_dim)
                d_query = self.temporal_conv2D_query(d_query)[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                d_key = self.temporal_conv2D_key(d_key)[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                query_list.append(d_query)
                key_list.append(d_key)

            if self.h_length > 0:
                # (B, d_model, N, T) -> (B, num_head, head_dim, N, T) -> (B, N, num_head, T, head_dim)
                h_query = self.temporal_conv2D_query(h_query)[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                h_key = self.temporal_conv2D_key(h_key)[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                query_list.append(h_query)
                key_list.append(h_key)
            
            query = torch.cat(query_list, dim=-2)
            key = torch.cat(key_list, dim=-2)

        elif (not query_multi_segment) and (not key_multi_segment):
            query = self.temporal_conv2D_query(query.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
            key = self.temporal_conv2D_key(key.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)

        elif (not query_multi_segment) and (key_multi_segment):
            query = self.temporal_conv2D_query(query.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
            key_list = []
            if self.w_length > 0:
                w_key = self.temporal_conv2D_key(w_key)[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                key_list.append(w_key)
            
            if self.d_length > 0:
                d_key = self.temporal_conv2D_key(d_key)[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                key_list.append(d_key)

            if self.h_length > 0:
                h_key = self.temporal_conv2D_key(h_key)[:, :, :, :-self.padding].contiguous().view(batch_size, self.num_head, self.head_dim, num_nodes, -1).permute(0, 3, 1, 4, 2)
                key_list.append(h_key)

            key = torch.cat(key_list, dim=-2)

        # (B, N, num_head, T, head_dim)
        value = self.fc(value).view(batch_size, num_nodes, -1, self.num_head, self.head_dim).transpose(2, 3)
        attn_score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, N, H, T, T)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)   # (B, 1, 1, T, T)
            attn_score = attn_score.masked_fill_(mask == 0, -1e9)
        attn_score = F.softmax(attn_score, dim=-1)  # (B, N, H, T, T)
        if self.dropout is not None:
            attn_score = self.dropout(attn_score)
        
        output = torch.matmul(attn_score, value)
        output = output.transpose(2, 3).contiguous()  #  (B, N, T, H, dk)
        output = output.view(batch_size, num_nodes, -1, self.num_head * self.head_dim)  # (B, N, T, d_model)
        return self.proj(output)


class PositionWiseSpatialGCN(nn.Module):
    def __init__(self, norm_adj, input_dim, output_dim, dropout=.0):
        super(PositionWiseSpatialGCN, self).__init__()
        self.norm_adj = norm_adj
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Theta = nn.Linear(input_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (B, N, T, input_dim)
        batch_size, num_nodes, timesteps, _ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, num_nodes, self.input_dim)  # (B * T, N, input_dim)
        out = self.Theta(torch.matmul(self.norm_adj, x))
        out = F.relu(out.reshape(batch_size, timesteps, num_nodes, self.output_dim).transpose(1, 2))
        out = self.dropout(out)
        return out
    

class PositionWiseSpatialAttention(nn.Module):
    def __init__(self, norm_adj, input_dim, output_dim, dropout=.0):
        super(PositionWiseSpatialAttention, self).__init__()
        self.norm_adj = norm_adj
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Theta = nn.Linear(input_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # x shape: (B, N, T, input_dim)
        batch_size, num_nodes, timesteps, _ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(-1, num_nodes, self.input_dim)  # (B * T, N, input_dim)
        spatial_scores = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(self.input_dim)
        spatial_scores = F.softmax(spatial_scores, dim=-1)   # (B * T, N, N)
        out = self.Theta(torch.matmul(self.norm_adj.mul(spatial_scores), x))
        out = F.relu(out.reshape(batch_size, timesteps, num_nodes, self.output_dim).transpose(1, 2))
        out = self.dropout(out)
        return out


class SubLayerConnection(nn.Module):
    def __init__(self, d_model, dropout, sublayer, residual_connection, use_LayerNorm, is_Attn=True):
        super(SubLayerConnection, self).__init__()
        self.is_Attn = is_Attn
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm

        if self.use_LayerNorm:
            self.layerNorm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x shape: (B, N, T, d_model)
        residual = x
        if self.use_LayerNorm:
            x = self.layerNorm(x)

        if self.is_Attn: # Multi-Head Temporal Attention
            out = self.dropout(self.sublayer(x, x, x, query_multi_segment=True, key_multi_segment=True))
        else:            # Feed Forward GCN
            out = self.dropout(self.sublayer(x))

        if self.residual_connection:
            return residual + out
        else:
            return out
        

class EncoderLayer(nn.Module):
    def __init__(self, num_head, d_model, norm_adj, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, 
                 dropout=.0, aware_temporal_context=True, scaledSAt=True, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if aware_temporal_context:  # employ temporal trend-aware attention
            self.temporal_attn = MultiHeadAttentionAwareTemporalContext_q1d_k1d(num_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size, dropout)
        else:         # employ traditional attention
            self.temporal_attn = MultiHeadAttention(num_head, d_model, dropout)
        
        if scaledSAt: # employ spatial attention
            self.position_wise_gcn = PositionWiseSpatialAttention(norm_adj, d_model, d_model, dropout)
        else:
            self.position_wise_gcn = PositionWiseSpatialGCN(norm_adj, d_model, d_model, dropout)

        if residual_connection or use_LayerNorm:
            self.attn_sublayer = SubLayerConnection(d_model, dropout, self.temporal_attn, residual_connection, use_LayerNorm, is_Attn=True)
            self.ffn_sublayer = SubLayerConnection(d_model, dropout, self.position_wise_gcn, residual_connection, use_LayerNorm, is_Attn=False)

    def forward(self, x):
        # x shape: (B, N, T, d_model)
        if self.residual_connection or self.use_LayerNorm:
            x = self.attn_sublayer(x)
            return self.ffn_sublayer(x)
        else:
            x = self.temporal_attn(x, x, x, query_multi_segment=True, key_multi_segment=True)
            return self.position_wise_gcn(x)


class DecoderLayer(nn.Module):
    def __init__(self, num_head, d_model, norm_adj, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size=3, 
                 dropout=.0, aware_temporal_context=True, scaledSAt=True, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if aware_temporal_context:  # employ temporal trend-aware attention
            self.temporal_attn1 = MultiHeadAttentionAwareTemporalContext_qc_kc(num_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size, dropout)
            self.temporal_attn2 = MultiHeadAttentionAwareTemporalContext_qc_k1d(num_head, d_model, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size, dropout)
        else:         # employ traditional attention
            self.temporal_attn1 = MultiHeadAttention(num_head, d_model, dropout)
            self.temporal_attn2 = MultiHeadAttention(num_head, d_model, dropout)
        
        if scaledSAt: # employ spatial attention
            self.position_wise_gcn = PositionWiseSpatialAttention(norm_adj, d_model, d_model, dropout)
        else:
            self.position_wise_gcn = PositionWiseSpatialGCN(norm_adj, d_model, d_model, dropout)

        if residual_connection or use_LayerNorm:
            self.attn_sublayer1 = SubLayerConnection(d_model, dropout, self.temporal_attn1, residual_connection, use_LayerNorm, is_Attn=True)
            self.attn_sublayer2 = SubLayerConnection(d_model, dropout, self.temporal_attn2, residual_connection, use_LayerNorm, is_Attn=True)
            self.ffn_sublayer = SubLayerConnection(d_model, dropout, self.position_wise_gcn, residual_connection, use_LayerNorm, is_Attn=False)

    def forward(self, x, memory):
        """
            x shape: (B, N, T', d_model)
            memory shape: (B, N, T, d_model)
        """
        m = memory
        attn_shape = (1, x.size(-2), x.size(-2))
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        tgt_mask = (subsequent_mask == 0).to(memory.device)  # (1, T', T')
        if self.residual_connection or self.use_LayerNorm:
            x = self.attn_sublayer1(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False)
            x = self.attn_sublayer2(x, m, m, query_multi_segment=False, key_multi_segment=True)
            return self.ffn_sublayer(x)
        else:
            x = self.temporal_attn1(x, x, x, tgt_mask, query_multi_segment=False, key_multi_segment=False)
            x = self.temporal_attn2(x, m, m, query_multi_segment=False, key_multi_segment=True)
            return self.position_wise_gcn(x)


class ASTGNN(nn.Module):
    def __init__(self, 
                 num_nodes,
                 input_dim, 
                 output_dim, 
                 d_model, 
                 adj_mx, 
                 num_head, 
                 num_layers, 
                 num_of_weeks,
                 num_of_days, 
                 num_of_hours, 
                 points_per_hour, 
                 num_for_predict, 
                 dropout=.0, 
                 aware_temporal_context=True,
                 scaledSAt=True, 
                 SE=True, 
                 TE=False, 
                 kernel_size=3, 
                 gcn_num_layers=0, 
                 residual_connection=True, 
                 use_LayerNorm=True):
        super(ASTGNN, self).__init__()
        self.num_nodes = num_nodes
        self.norm_adj = self.norm_Adj(adj_mx)
        # encoder temporal position embedding
        max_len = max(num_of_weeks * 7 * 24 * num_for_predict, num_of_days * 24 * num_for_predict, num_of_hours * num_for_predict)

        if SE and TE:
            pass
        elif SE and (not TE):
            spatial_position = SpatialPositionalEncoding(d_model, num_nodes, self.norm_adj, dropout, gcn_num_layers)
            self.encoder_embedding = nn.Sequential(nn.Linear(input_dim, d_model), copy.deepcopy(spatial_position))
            self.decoder_embedding = nn.Sequential(nn.Linear(output_dim, d_model), copy.deepcopy(spatial_position))
        elif (not SE) and TE:
            pass
        else:
            self.encoder_embedding = nn.Sequential(nn.Linear(input_dim, d_model))
            self.decoder_embedding = nn.Sequential(nn.Linear(output_dim, d_model))

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(num_head, d_model, self.norm_adj, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size, 
                dropout, aware_temporal_context, scaledSAt, residual_connection, use_LayerNorm)
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(num_head, d_model, self.norm_adj, num_of_weeks, num_of_days, num_of_hours, points_per_hour, kernel_size, 
                dropout, aware_temporal_context, scaledSAt, residual_connection, use_LayerNorm)
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def norm_Adj(self, W):
        """
        W: torch.Tensor, shape is (N, N), N is the num of nodes
        normalized Adj matrix: (D^hat)^{-1} A^hat; torch.Tensor, shape (N, N)
        """
        assert W.dim() == 2 and W.shape[0] == W.shape[1]
        N = W.shape[0]
        W = W + torch.eye(N, device=W.device, dtype=W.dtype)
        D = torch.diag(1.0 / torch.sum(W, dim=1))
        return torch.mm(D, W)

    def forward(self, src, trg):
        """
        src shape: (B, N, T, input_dim)
        trg shape: (B, N, T, output_dim)
        """
        # 1. encoder
        h = self.encoder_embedding(src)
        encoder_output = self.encoder_norm(self.encoder(h))
        # 2. decoder
        out = self.decoder(self.decoder_embedding(trg), encoder_output)
        out = self.decoder_norm(out)
        # 3. predict
        return self.output_proj(out)