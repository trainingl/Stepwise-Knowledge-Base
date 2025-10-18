import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# 2025_Neural Networks_PDG2Seq: Periodic Dynamic Graph to Sequence Model for Traffic Flow Prediction
class PDGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim):
        super(PDGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, 2 * cheb_k + 1, input_dim, output_dim)
        )
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
    
    def forward(self, x, supports, node_embeddings):
        """
            x shape: (B, N, input_dim)
            supports shape: [(B, N, N), (B, N, N)]
            node_embeddings shape: (N, embed_dim)
        """
        out = [x]
        for a in supports:
            x1 = torch.einsum("bnm,bmc->bnc", a, x).contiguous()
            out.append(x1)
            for _ in range(2, self.cheb_k + 1):
                x2 = torch.einsum("bnm,bmc->bnc", a, x1).contiguous()
                out.append(x2)
                x1 = x2
        x_g = torch.stack(out, dim=2)  # (B, N, k, input_dim) 
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv


class PDG2SeqCell(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, time_dim):
        super(PDG2SeqCell, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim + self.hidden_dim, self.hyperGNN_dim),
            nn.Sigmoid(),
            nn.Linear(self.hyperGNN_dim, self.middle_dim),
            nn.Sigmoid(),
            nn.Linear(self.middle_dim, time_dim)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(input_dim + self.hidden_dim, self.hyperGNN_dim),
            nn.Sigmoid(),
            nn.Linear(self.hyperGNN_dim, self.middle_dim),
            nn.Sigmoid(),
            nn.Linear(self.middle_dim, time_dim)
        )
        self.gate = PDGCN(input_dim + self.hidden_dim, 2 * self.hidden_dim, cheb_k, embed_dim)
        self.update = PDGCN(input_dim + self.hidden_dim, self.hidden_dim, cheb_k, embed_dim)

    @staticmethod
    def preprocessing(adj):
        num_nodes = adj.shape[-1]
        adj = adj + torch.eye(num_nodes).to(adj.device)
        d = torch.unsqueeze(torch.sum(adj, dim=-1), dim=-1)  # Degree Matrix: (N, 1)
        adj = adj / d   # D^(-1)A
        return adj

    def forward(self, x, state, st_embeddings):
        """
            x shape: (B, N, input_dim)
            state shape: (B, N, hidden_dim)
            st_embeddings shape: [(B, time_dim), (B, time_dim), (N, embed_dim)]
        """
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        # dynamic graph generation
        filter1 = self.fc1(input_and_state)  # F(t) = V(t) || H(t-1)
        filter2 = self.fc2(input_and_state)  # F(t) = V(t) || H(t-1)
        nodevec1 = torch.tanh(torch.einsum('bd,bnd->bnd', st_embeddings[0], filter1))
        nodevec2 = torch.tanh(torch.einsum('bd,bnd->bnd', st_embeddings[1], filter2))
        adj = torch.matmul(nodevec1, nodevec2.transpose(1, 2)) - torch.matmul(
            nodevec2, nodevec1.transpose(1, 2))
        supports = [self.preprocessing(F.relu(adj)), self.preprocessing(F.relu(-adj.transpose(-2, -1)))]

        z_r = torch.sigmoid(self.gate(input_and_state, supports, st_embeddings[2]))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports, st_embeddings[2]))
        h = r * state + (1 - r) * hc
        return h
    
    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)
        

class PDG2Seq_Encoder(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, time_dim, num_layers=1):
        super(PDG2Seq_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.PDG2Seq_cells = nn.ModuleList()
        self.PDG2Seq_cells.append(PDG2SeqCell(num_nodes, input_dim, self.hidden_dim, cheb_k, embed_dim, time_dim))
        for _ in range(1, num_layers):
            self.PDG2Seq_cells.append(PDG2SeqCell(num_nodes, self.hidden_dim, self.hidden_dim, cheb_k, embed_dim, time_dim))

    def forward(self, x, init_state, st_embeddings):
        """
            x shape: (B, T, N, input_dim)
            init_state shape: (num_layers, B, N, hidden_dim)
            st_embeddings shape: [(B, T, time_dim), (B, T, time_dim), (N, embed_dim)]
        """
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.PDG2Seq_cells[i](current_inputs[:, t, :, :], state, [st_embeddings[0][:, t, :], st_embeddings[1][:, t, :], st_embeddings[2]])
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.PDG2Seq_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)
    

class PDG2Seq_Decoder(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, time_dim, num_layers=1):
        super(PDG2Seq_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.PDG2Seq_cells = nn.ModuleList()
        self.PDG2Seq_cells.append(PDG2SeqCell(num_nodes, input_dim, self.hidden_dim, cheb_k, embed_dim, time_dim))
        for _ in range(1, self.num_layers):
            self.PDG2Seq_cells.append(PDG2SeqCell(num_nodes, self.hidden_dim, self.hidden_dim, cheb_k, embed_dim, time_dim))

    def forward(self, xt, init_state, st_embeddings):
        """
            xt shape: (B, N, input_dim)
            init_state shape: (num_layers, B, N, hidden_dim)
            st_embeddings shape: [(B, time_dim), (B, time_dim), (N, embed_dim)]
        """
        assert xt.shape[1] == self.num_nodes and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.PDG2Seq_cells[i](current_inputs, init_state[i], st_embeddings)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden
    

class PDG2Seq(nn.Module):
    def __init__(self, num_nodes, input_dim, rnn_units, output_dim, embed_dim, time_dim, cheb_k, horizon, num_layers, use_day, use_week, cl_decay_steps, use_curriculum_learning):
        super(PDG2Seq, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.use_day = use_day
        self.use_week = use_week
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning

        # 1. spatial-temporal embeddings
        self.node_embeddings = nn.Parameter(torch.empty(self.num_nodes, embed_dim))
        self.tod_embedding1 = nn.Parameter(torch.empty(288, time_dim))
        self.tod_embedding2 = nn.Parameter(torch.empty(288, time_dim))
        self.dow_embedding1 = nn.Parameter(torch.empty(7, time_dim))
        self.dow_embedding2 = nn.Parameter(torch.empty(7, time_dim))

        # 2. encoder & decoder
        self.encoder = PDG2Seq_Encoder(num_nodes, input_dim, rnn_units, cheb_k, embed_dim, time_dim, num_layers)
        self.decoder = PDG2Seq_Decoder(num_nodes, output_dim, rnn_units, cheb_k, embed_dim, time_dim, num_layers)

        # 3. predictor
        self.proj = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    def forward(self, x, labels=None, batches_seen=None):
        """
            x shape: (B, T, N, 3)
            labels shape: (B, T, N, 3)
        """
        # 1. prepare data
        batch_size = x.shape[0]
        x_tod = x[..., 0, 1]
        y_tod = labels[..., 0, 1]
        tod_emb1_en = self.tod_embedding1[(x_tod * 288).type(torch.LongTensor)]  # (B, T, time_dim)
        tod_emb2_en = self.tod_embedding2[(x_tod * 288).type(torch.LongTensor)]
        tod_emb1_de = self.tod_embedding1[(y_tod * 288).type(torch.LongTensor)]  # (B, T, time_dim)
        tod_emb2_de = self.tod_embedding2[(y_tod * 288).type(torch.LongTensor)]
        
        if self.use_week:
            x_dow = x[..., 0, 2]
            y_dow = labels[..., 0, 2]
            dow_emb1_en = self.dow_embedding1[(x_dow).type(torch.LongTensor)]    # (B, T, time_dim)
            dow_emb2_en = self.dow_embedding2[(x_dow).type(torch.LongTensor)]
            dow_emb1_de = self.dow_embedding1[(y_dow).type(torch.LongTensor)]    # (B, T, time_dim)
            dow_emb2_de = self.dow_embedding2[(y_dow).type(torch.LongTensor)]
            temporal_embedding_en1 = torch.mul(tod_emb1_en, dow_emb1_en)
            temporal_embedding_en2 = torch.mul(tod_emb2_en, dow_emb2_en)
            temporal_embedding_de1 = torch.mul(tod_emb1_de, dow_emb1_de)
            temporal_embedding_de2 = torch.mul(tod_emb2_de, dow_emb2_de)
        else:
            temporal_embedding_en1 = tod_emb1_en
            temporal_embedding_en2 = tod_emb2_en
            temporal_embedding_de1 = tod_emb1_de
            temporal_embedding_de2 = tod_emb2_de

        # 2. encoder
        st_embeddings = [temporal_embedding_en1, temporal_embedding_en2, self.node_embeddings]
        x = x[..., 0:1]
        init_state = self.encoder.init_hidden(batch_size).to(x.device)
        h_en, _ = self.encoder(x, init_state, st_embeddings)
        h_last = h_en[:, -1, :, :]

        # 3. decoder
        ht_list = [h_last] * self.num_layers
        go = torch.zeros((batch_size, self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(go, ht_list, [temporal_embedding_de1[:, t, :], temporal_embedding_de2[:, t, :], self.node_embeddings])
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    go = labels[:, t, :, 0:1]
        
        output = torch.stack(out, dim=1)
        return output
    

if __name__ == '__main__':
    model = PDG2Seq(
        num_nodes=170, 
        input_dim=1, 
        rnn_units=64, 
        output_dim=1, 
        embed_dim=8, 
        time_dim=16, 
        cheb_k=2, 
        horizon=12, 
        num_layers=1,
        use_day=True, 
        use_week=True, 
        cl_decay_steps=2000, 
        use_curriculum_learning=True
    )
    
    x = torch.randn(1, 12, 170, 1)
    x_tod = torch.rand(1, 12, 170, 1)
    x_dow = torch.randint(0, 6, size=(1, 12, 170, 1))
    x = torch.cat([x, x_tod, x_dow], dim=-1)

    y = torch.randn(1, 12, 170, 1)
    y_tod = torch.rand(1, 12, 170, 1)
    y_dow = torch.randint(0, 6, size=(1, 12, 170, 1))
    y = torch.cat([y, y_tod, y_dow], dim=-1)
    output = model(x, labels=y, batches_seen=1)
    print("Output shape: ", output.shape)