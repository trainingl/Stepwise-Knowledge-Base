import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 2023_PR_A Decomposition Dynamic graph convolutional recurrent network for traffic forecasting
class DGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim):
        super(DGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, input_dim, output_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
        self.hyperGNN_dim = 16
        self.middle_dim = 2
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, self.hyperGNN_dim),
            nn.Sigmoid(),
            nn.Linear(self.hyperGNN_dim, self.middle_dim),
            nn.Sigmoid(),
            nn.Linear(self.middle_dim, self.embed_dim)
        )

    def forward(self, x, st_embeddings):
        """
            x shape: (B, N, input_dim)
            st_embeddings shape: type(list), [(B, N, embed_dim), (N, embed_dim)]
        """
        num_nodes = st_embeddings[0].shape[1]
        supports1 = torch.eye(num_nodes).to(x.device)      # I: (N, N)
        filter = self.fc(x)
        nodevec = torch.tanh(torch.mul(st_embeddings[0], filter))  # (B, N, embed_dim)
        adj_mx = F.relu(torch.matmul(nodevec, nodevec.transpose(2, 1)))
        supports2 = self.get_laplacian(adj_mx, supports1)  # L: (B, N, N)

        x_g1 = torch.einsum("nm,bmc->bnc", supports1, x)
        x_g2 = torch.einsum("bnm,bmc->bnc", supports2, x)
        x_g = torch.stack([x_g1, x_g2], dim=1)   
        x_g = x_g.permute(0, 2, 1, 3)     # (B, 2, N, input_dim) ->  (B, N, 2, input_dim)
        weights = torch.einsum("nd,dkio->nkio", st_embeddings[1], self.weights_pool)
        bias = torch.matmul(st_embeddings[1], self.bias_pool)
        x_gconv = torch.einsum("bnki,nkio->bno", x_g, weights) + bias
        return x_gconv

    def get_laplacian(self, graph, I, normalize=True):
        """
            :param graph: the graph structure without self loop, [N, N].
            :param normalize: whether to used the normalized laplacian.
            :return: graph laplacian matrix.
        """
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        else:
            graph = graph + I
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.matmul(torch.matmul(D, graph), D)
        return L


class DDGCRNCell(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, embed_dim):
        super(DDGCRNCell, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        self.gate = DGCN(input_dim + self.hidden_dim, 2 * self.hidden_dim, cheb_k, embed_dim)
        self.update = DGCN(input_dim + self.hidden_dim, self.hidden_dim, cheb_k, embed_dim)

    def forward(self, x, state, st_embeddings):
        """
            x shape: (B, N, input_dim)
            state shape: (B, N, hidden_dim)
            st_embeddings: type(list), [(B, N, embed_dim), (N, embed_dim)]
        """
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, st_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, st_embeddings))
        h = r * state + (1 - r) * hc
        return h
    
    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)


class DGCRM(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, num_layers=1):
        super(DGCRM, self).__init__()
        assert num_layers >= 1, "At least one DCRNN layer in the Encoder."
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.num_layers = num_layers
        self.DGCRM_cells = nn.ModuleList()
        self.DGCRM_cells.append(DDGCRNCell(num_nodes, self.input_dim, self.hidden_dim, cheb_k, embed_dim))
        for _ in range(2, self.num_layers):
            self.DGCRM_cells.append(DDGCRNCell(num_nodes, self.hidden_dim, self.hidden_dim, cheb_k, embed_dim))

    def forward(self, x, init_state, st_embeddings):
        """
            x shape: (B, T, N, input_dim)
            state shape: (num_layers, B, N, hidden_dim)
            st_embeddings: type(list), [(B, T, N, embed_dim), (N, embed_dim)]
        """
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.DGCRM_cells[i](current_inputs[:, t, :, :], state, [st_embeddings[0][:, t, :, :], st_embeddings[1]])
                inner_states.append(state)  # (B, N, hidden_dim)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer, shape of (B, T, N, hidden_dim)
        # output_hidden: the last state of each layer, shape of (num_layers, B, N, hidden_dim)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.DGCRM_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)
    

class DDGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, rnn_units, output_dim, embed_dim, cheb_k, 
                 horizon, num_layers, use_day, use_week):
        super(DDGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.use_D = use_day
        self.use_W = use_week

        # 1. st_embeddings
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.T_i_D_emb = nn.Parameter(torch.empty(288, self.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, self.embed_dim))
        # 2. encoder
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.encoder1 = DGCRM(num_nodes, input_dim, rnn_units, cheb_k, embed_dim, num_layers)
        self.encoder2 = DGCRM(num_nodes, input_dim, rnn_units, cheb_k, embed_dim, num_layers)
        # 3. predictor
        self.end_conv1 = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv2 = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv3 = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, i=2):
        # source shape: (B, T, N, 3)
        node_embedings = self.node_embeddings   # (N, embed_dim)
        if self.use_D:
            t_i_d_data = source[..., 1]
            T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]  # (B, T, N, embed_dim)
            node_embedings = torch.mul(node_embedings, T_i_D_emb)
        if self.use_W:
            d_i_w_data = source[..., 2]
            D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]        # (B, T, N, embed_dim)
            node_embedings = torch.mul(node_embedings, D_i_W_emb)
        st_embeddings = [node_embedings, self.node_embeddings]
        
        source = source[..., 0].unsqueeze(-1)  # (B, T, N, 1)
        if i == 1:
            init_state1 = self.encoder1.init_hidden(source.shape[0])
            output, _ = self.encoder1(source, init_state1, st_embeddings)
            output = self.dropout1(output[:, -1:, :, :])  # (B, 1, N, D)
            # CNN based predictor
            output1 = self.end_conv1(output)  # (B, T*C, N, 1)
            return output1
        else:
            init_state1 = self.encoder1.init_hidden(source.shape[0])
            output, _ = self.encoder1(source, init_state1, st_embeddings)
            output = self.dropout1(output[:, -1:, :, :])  # (B, 1, N, D)

            # CNN based predictor
            output1 = self.end_conv1(output)
            source1 = self.end_conv2(output)
            source2 = source - source1

            init_state2 = self.encoder2.init_hidden(source2.shape[0])
            output2, _ = self.encoder2(source2, init_state2, st_embeddings)
            output2 = self.dropout2(output2[:, -1:, :, :])
            output2 = self.end_conv3(output2)
            return output1 + output2


if __name__ == '__main__':
    model = DDGCRN(
        num_nodes=170, 
        input_dim=1, 
        rnn_units=64, 
        output_dim=1, 
        embed_dim=10, 
        cheb_k=2, 
        horizon=12, 
        num_layers=1, 
        use_day=True, 
        use_week=True
    )
    x = torch.randn(32, 12, 170, 1)
    tod = torch.rand(32, 12, 170, 1)
    dow = torch.randint(0, 6, size=(32, 12, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)
    print("Output shape: ", model(x).shape)