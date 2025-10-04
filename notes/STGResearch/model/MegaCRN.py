import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 2023_AAAI_Spatial-temporal meta-graph learning for traffic forecasting
class AGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        """
            x shape: (B, N, input_dim)
            supports: type(list), [(N, N), (N, N)]
        """
        x_g = []
        support_set = []
        for support in supports:
            # support: the Laplace matrix, [I, L]
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for _ in range(2, self.cheb_k):
                # L is calculated recursively using Chebyshev polynomials
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # (B, N, 2 * cheb_k * input_dim)
        x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias
        return x_gconv
    

class AGCRNCell(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k):
        super(AGCRNCell, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        self.gate = AGCN(input_dim + self.hidden_dim, 2 * self.hidden_dim, cheb_k)
        self.update = AGCN(input_dim + self.hidden_dim, self.hidden_dim, cheb_k)

    def forward(self, x, state, supports):
        """
            x shape: (B, N, input_dim)
            state shape: (B, N, hidden_dim)
            supports: type(list), [(N, N), (N, N)]
        """
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h
    
    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)
    

class ADCRNN_Encoder(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, num_layers):
        super(ADCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(num_nodes, input_dim, self.hidden_dim, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(num_nodes, self.hidden_dim, self.hidden_dim, cheb_k))

    def forward(self, x, init_state, supports):
        """
            x shape: (B, T, N, input_dim)
            init_state shape: (num_layers, B, N, hidden_dim)
            supports: type(list), [(N, N), (N, N)]
        """
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            init_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                init_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(init_states, dim=1)
        # output_hidden: the outputs of each layer, shape of (num_layers, B, N, hidden_dim)
        # current_inputs: the last state for each layer, shape of (B, T, N, hidden_dim)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class ADCRNN_Decoder(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, num_layers):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(num_nodes, input_dim, self.hidden_dim, cheb_k))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(num_nodes, self.hidden_dim, self.hidden_dim, cheb_k))
        
    def forward(self, xt, init_state, supports):
        """
            xt shape: (B, N, input_dim)
            init_state shape: (num_layers, B, N, hidden_dim)
            supports: type(list), [(N, N), (N, N)]
        """
        assert xt.shape[1] == self.num_nodes and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        # current_inputs: the last state for layers, (B, N, hidden_dim) 
        # output_hidden: the outputs of each layer, (num_layers, B, N, hidden_dim)
        return current_inputs, output_hidden


class MegaCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, cheb_k=3,
                 ycov_dim=1, mem_num=20, mem_dim=64, cl_decay_steps=2000, use_curriculum_learning=True):
        super(MegaCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning

        # 1. memory-node bank
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        # 2. encoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.hidden_dim, self.cheb_k, self.num_layers)

        # 3. decoder
        self.decoder_dim = self.hidden_dim + self.mem_dim
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        # 4. output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.hidden_dim, self.mem_dim), requires_grad=True)   # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)   # project memory to embeddings1
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True)   # project memory to embeddings2
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t:torch.Tensor):
        # h_t shape: (B, N, hidden_dim)
        query = torch.matmul(h_t, self.memory['Wq'])  # (B, N, mem_dim)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1) # ind shape: (B, N, 2)
        # construct positive and negative samples
        pos = self.memory['Memory'][ind[:, :, 0]]   # (B, N, mem_dim)
        neg = self.memory['Memory'][ind[:, :, 1]]   # (B, N, mem_dim)
        return value, query, pos, neg

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    def forward(self, x, y_cov, labels=None, batch_seen=None):
        """
            x shape: (B, T, N, input_dim)
            y_cov shape: (B, T, N, ycov_dim)
            labels shape: (B, T, N, output_dim)
        """
        batch_size, _, num_nodes, _ = x.shape
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]
        init_state = self.encoder.init_hidden(batch_size)
        ht_en, _ = self.encoder(x, init_state, supports)
        h_t = ht_en[:, -1, :, :]   # last state: (B, N, hidden_dim)

        h_att, query, pos, neg = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)  # (B, N, hidden_dim + mem_dim)
        ht_list = [h_t] * self.num_layers

        go = torch.zeros((batch_size, num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            ht_de, _ = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports)
            go = self.proj(ht_de)
            out.append(go)
            # Using curriculum learning to solve autoregressive error accumulation
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batch_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)  # (B, T, N, output_dim)
        return output, h_att, query, pos, neg


if __name__ == '__main__':
    model = MegaCRN(
        num_nodes=170, 
        input_dim=1, 
        output_dim=1, 
        horizon=12, 
        rnn_units=64, 
        num_layers=1, 
        cheb_k=2,
        ycov_dim=1, 
        mem_num=20, 
        mem_dim=64, 
        cl_decay_steps=2000, 
        use_curriculum_learning=True
    )
    x = torch.randn(64, 12, 170, 1)
    y_cov = torch.randn(64, 12, 170, 1)
    labels = torch.randn(64, 12, 170, 1)
    output, h_att, query, pos, neg = model(x, y_cov, labels, batch_seen=1)
    print("Output shape: ", output.shape)