import torch
import torch.nn as nn
import numpy as np


# 2024_KDD_Heterogeneity-informed meta-parameter learning for spatiotemporal time series forecasting
class HimGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super(HimGCN, self).__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None
        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim))
            )
            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
            )
        else:
            self.weights = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(cheb_k * input_dim, output_dim))
            )
            self.bias = nn.init.constant_(
                nn.Parameter(torch.FloatTensor(output_dim)), val=0
            )
    
    def forward(self, x, support, embeddings):
        """
            x shape: (B, N, input_dim)
            support shape: (N, N) or (B, N, N)
            embeddings shape:
                1.  T: (B, embed_dim)
                2.  S: (N, embed_dim)
                3. ST: (B, N, embed_dim)
        """
        x_g = []
        if support.dim() == 2:
            # 1.1 support shape: (N, N)
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]
            for _ in range(2, self.cheb_k):
                graph_list.append(torch.matmul(2 * support, graph_list[-1]) - graph_list[-2])
            for graph in graph_list:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
        if support.dim() == 3:
            # 1.2 support shape: (B, N, N)
            graph_list = [torch.eye(support.shape[1]).repeat(support.shape[0], 1, 1).to(support.device), support]
            for _ in range(2, self.cheb_k):
                graph_list.append(torch.matmul(2 * support, graph_list[-1]) - graph_list[-2])
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)  # (B, N, cheb_k * input_dim)

        if self.meta_axis:
            if self.meta_axis == 'T':
                # 2.1 time_embeddings shape: (B, embed_dim)
                weights = torch.einsum("bd,dio->bio", embeddings, self.weights_pool)   # (B, cheb_k * input_dim, output_dim)
                bias = torch.matmul(embeddings, self.bias_pool)   # (B, output_dim)
                x_gconv = torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
            elif self.meta_axis == 'S':
                # 2.2 node_embedddings shape: (N, embed_dim)
                weights = torch.einsum("nd,dio->nio", embeddings, self.weights_pool)   # (N, cheb_k * input_dim, output_dim)
                bias = torch.matmul(embeddings, self.bias_pool)   # (N, output_dim)
                x_gconv = torch.einsum("bni,nio->bno", x_g, weights) + bias
            elif self.meta_axis == 'ST':
                # 2.3 st_embeddings shape: (B, N, embed_dim)
                weights = torch.einsum("bnd,dio->bnio", embeddings, self.weights_pool) # (B, N, cheb_k * input_dim, output_dim)
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)  # (B, N, output_dim)
                x_gconv = torch.einsum("bni,bnio->bno", x_g, weights) + bias
        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias
        return x_gconv
        

class HimGCRU(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis='S'):
        super(HimGCRU, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        self.gate = HimGCN(input_dim + self.hidden_dim, 2 * self.hidden_dim, cheb_k, embed_dim, meta_axis)
        self.update = HimGCN(input_dim + self.hidden_dim, self.hidden_dim, cheb_k, embed_dim, meta_axis)
    
    def forward(self, x, state, support, embeddings):
        """
            x shape: (B, N, input_dim)
            state shape: (B, N, hidden_dim)
            support shape: (N, N) or (B, N, N)
            embeddings shape:
                1.  T: (B, embed_dim)
                2.  S: (N, embed_dim)
                3. ST: (B, N, embed_dim)
        """
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, support, embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, support, embeddings))
        h = r * state + (1 - r) * hc
        return h
    
    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)
    

class HimEncoder(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, num_layers, meta_axis='S'):
        super(HimEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        self.cells.append(HimGCRU(num_nodes, input_dim, self.hidden_dim, cheb_k, embed_dim, meta_axis))
        for _ in range(1, self.num_layers):
            self.cells.append(HimGCRU(num_nodes, self.hidden_dim, self.hidden_dim, cheb_k, embed_dim, meta_axis))

    def forward(self, x, support, embeddings):
        """
            x shape: (B, T, N, input_dim)
            support shape: (N, N) or (B, N, N)
            embeddings shape: (B, embed_dim) or (N, embed_dim)
        """
        batch_size, seq_length, _, _ = x.shape
        current_input = x
        output_hidden = []
        for cell in self.cells:
            state = cell.init_hidden_state(batch_size).to(x.device)
            inner_states = []
            for t in range(seq_length):
                state = cell(current_input[:, t, :, :], state, support, embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_input = torch.stack(inner_states, dim=1)
        # current_input: the outputs of the last layer, shape of (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer, shape of (num_layers, B, N, hidden_dim)
        return current_input, output_hidden


class HimDecoder(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, num_layers, meta_axis="ST"):
        super(HimDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        self.cells.append(HimGCRU(num_nodes, input_dim, self.hidden_dim, cheb_k, embed_dim, meta_axis))
        for _ in range(1, self.num_layers):
            self.cells.append(HimGCRU(num_nodes, self.hidden_dim, self.hidden_dim, cheb_k, embed_dim, meta_axis))

    def forward(self, xt, init_state, support, embeddings):
        """
            xt shape: (B, N, input_dim)
            init_state shape: (num_layer, B, N, hidden_dim)
            support shape: (B, N, N)
            embeddings shape: (B, N, embed_dim)
        """
        current_input = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.cells[i](current_input, init_state[i], support, embeddings)
            output_hidden.append(state)
            current_input = state
        return current_input, output_hidden
        

class HimNet(nn.Module):
    def __init__(self, 
                 num_nodes, 
                 input_dim=3, 
                 output_dim=1,
                 hidden_dim=64, 
                 num_layers=1,
                 horizon=12, 
                 cheb_k=2, 
                 ycov_dim=2, 
                 tod_embedding_dim=8,
                 dow_embedding_dim=8, 
                 node_embedding_dim=16,
                 st_embedding_dim=16,
                 tf_decay_steps=4000,
                 use_teacher_forcing=True):
        super(HimNet, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.node_embedding_dim = node_embedding_dim
        self.st_embedding_dim = st_embedding_dim
        self.tf_decay_steps = tf_decay_steps
        self.use_teacher_forcing = use_teacher_forcing

        # 1. st_embeddings
        self.tod_embedding = nn.Embedding(288, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.node_embedding = nn.init.xavier_normal_(
            nn.Parameter(torch.empty(num_nodes, node_embedding_dim))
        )
        self.st_proj = nn.Linear(hidden_dim, st_embedding_dim)

        # 2. encoders & decoder
        self.encoder_s = HimEncoder(
            num_nodes, 
            input_dim, 
            hidden_dim, 
            cheb_k, 
            node_embedding_dim, 
            num_layers, 
            meta_axis='S'
        )
        self.encoder_t = HimEncoder(
            num_nodes, 
            input_dim, 
            hidden_dim, 
            cheb_k, 
            tod_embedding_dim + dow_embedding_dim, 
            num_layers, 
            meta_axis='T'
        )
        self.decoder = HimDecoder(
            num_nodes, 
            output_dim + ycov_dim, 
            hidden_dim, 
            cheb_k, 
            st_embedding_dim, 
            num_layers, 
            meta_axis="ST"
        )

        # 3.regressor
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def compute_sampling_threshold(self, batches_seen):
        return self.tf_decay_steps / (
            self.tf_decay_steps + np.exp(batches_seen / self.tf_decay_steps)
        )

    def forward(self, x, y_cov, label=None, batches_seen=None):
        """
            x shape: (B, T, N, 3)
            y_cov shape(B, T, N, ycov_dim)
            label shape: (B, T, N, 1)
        """
        tod = x[:, -1, 0, 1]
        dow = x[:, -1, 0, 2]
        tod_emb = self.tod_embedding((tod * 288).long())  # (B, tod_embedding_dim)
        dow_emb = self.dow_embedding(dow.long())          # (B, dow_embedding_dim)
        time_embedding = torch.cat([tod_emb, dow_emb], dim=-1)
        support = torch.softmax(
            torch.relu(self.node_embedding @ self.node_embedding.T), dim=-1
        )  # adaptive graph

        h_s, _ = self.encoder_s(x, support, self.node_embedding)
        h_t, _ = self.encoder_t(x, support, time_embedding)
        h_last = (h_s + h_t)[:, -1, :, :]   # (B, N, hidden_dim)

        st_embedding = self.st_proj(h_last) # (B, N, st_embedding_dim)
        support = torch.softmax(
            torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding, st_embedding)), dim=-1
        )  # dynamic graph
        
        ht_list = [h_last] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(
                torch.cat([go, y_cov[:, t, ...]], dim=-1),
                ht_list,
                support,
                st_embedding
            )
            go = self.out_proj(h_de)
            out.append(go)
            if self.training and self.use_teacher_forcing:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = label[:, t, :, :]
        output = torch.stack(out, dim=1)
        return output
    

if __name__ == '__main__':
    model = HimNet(
        num_nodes=170, 
        input_dim=3, 
        output_dim=1,
        hidden_dim=96, 
        num_layers=1,
        horizon=12, 
        cheb_k=2, 
        ycov_dim=2, 
        tod_embedding_dim=10,
        dow_embedding_dim=2, 
        node_embedding_dim=14,
        st_embedding_dim=10,
        tf_decay_steps=6000,
        use_teacher_forcing=True
    )
    x = torch.randn(1, 12, 170, 1)
    tod = torch.rand(1, 12, 170, 1)
    dow = torch.randint(0, 6, size=(1, 12, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)

    y_tod = torch.rand(1, 12, 170, 1)
    y_dow = torch.randint(0, 6, size=(1, 12, 170, 1))
    y_cov = torch.cat([y_tod, y_dow], dim=-1)
    labels = torch.randn(1, 12, 170, 1)
    output = model(x, y_cov, labels, batches_seen=1)
    print("Output shape: ", output.shape)