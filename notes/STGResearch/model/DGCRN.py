import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys


# 2023_TKDD_Dynamic graph convolutional recurrent network for traffic prediction: Benchmark and solution
class GConv_Hyper(nn.Module):
    def __init__(self, dims, gdep, alpha, beta, gamma):
        super(GConv_Hyper, self).__init__()
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mlp = nn.Sequential(
            nn.Linear((gdep + 1) * dims[0], dims[1]),
            nn.Sigmoid(),
            nn.Linear(dims[1], dims[2]),
            nn.Sigmoid(),
            nn.Linear(dims[2], dims[3])
        )

    def forward(self, x, adj):
        """
            x shape: (B, N, D)
            adj shape: (N, N)
        """
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + self.gamma * torch.einsum("bnc,nm->bmc", h, adj)
            out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        return ho


class GConv_RNN(nn.Module):
    def __init__(self, dims, gdep, alpha, beta, gamma):
        super(GConv_RNN, self).__init__()
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])
    
    def forward(self, x, adj):
        """
            x shape: (B, N, dims[0])
            adj shape: [(B, N, N), (N, N)]
            The first one is a dynamic graph, the second one is a predefined graph.
        """
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = (self.alpha * x 
                + self.beta * torch.einsum("bnc,bnm->bmc", h, adj[0]) 
                + self.gamma * torch.einsum("bnc,nm->bmc", h, adj[1]))
            out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        return ho


class GraphConstructor(nn.Module):
    def __init__(self, dims_hyper, tanhalpha, gcn_depth, predefined_A, list_weight):
        super(GraphConstructor, self).__init__()
        self.alpha = tanhalpha
        self.predefined_A = predefined_A
        # dims_hyper = [input_dim + hidden_dim, hyperGNN_dim, middle_dim, node_dim]
        self.gcn1_hyper_1 = GConv_Hyper(dims_hyper, gcn_depth, *list_weight)
        self.gcn1_hyper_2 = GConv_Hyper(dims_hyper, gcn_depth, *list_weight)
        self.gcn2_hyper_1 = GConv_Hyper(dims_hyper, gcn_depth, *list_weight)
        self.gcn2_hyper_2 = GConv_Hyper(dims_hyper, gcn_depth, *list_weight)

    def preprocessing(self, adj, predefined_A):
        # D^(-1)A: add self-loops and divide by the degree matrix
        adj = adj + torch.eye(adj.size(1)).to(adj.device)
        adj = adj / torch.unsqueeze(adj.sum(-1), -1)
        return [adj, predefined_A]

    def forward(self, x, state, nodevec1, nodevec2):
        """
            x shape: (B, N, input_dim)
            state shape: (B, N, hidden_dim)
            nodevec1 & nodevec2 shape: (N, node_dim)
        """
        hyper_input = torch.cat((x, state), dim=-1)  # I(t) = V(t) || T(t) || H(t-1)
        filter1 = self.gcn1_hyper_1(hyper_input, self.predefined_A[0]) + self.gcn1_hyper_2(hyper_input, self.predefined_A[1])
        filter2 = self.gcn2_hyper_1(hyper_input, self.predefined_A[0]) + self.gcn2_hyper_2(hyper_input, self.predefined_A[1])
        nodevec1 = torch.tanh(self.alpha * torch.mul(nodevec1, filter1))
        nodevec2 = torch.tanh(self.alpha * torch.mul(nodevec2, filter2))  # (B, N, node_dim)
        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - \
            torch.matmul(nodevec2, nodevec1.transpose(2, 1))
        adj = F.relu(torch.tanh(self.alpha * a))  # (B, N, N)
        adp = self.preprocessing(adj, self.predefined_A[0])
        adpT = self.preprocessing(adj.transpose(2, 1), self.predefined_A[1])
        return adp, adpT


class DGCRM(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, hyperGNN_dim, middle_dim, node_dim, tanhalpha, gcn_depth, predefined_A, list_weight):
        super(DGCRM, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        dims_hyper = [input_dim + self.hidden_dim, hyperGNN_dim, middle_dim, node_dim]
        dims = [input_dim + self.hidden_dim, self.hidden_dim]
        self.dgcn = GraphConstructor(dims_hyper, tanhalpha, gcn_depth, predefined_A, list_weight)
        self.gz1 = GConv_RNN(dims, gcn_depth, *list_weight)
        self.gz2 = GConv_RNN(dims, gcn_depth, *list_weight)
        self.gr1 = GConv_RNN(dims, gcn_depth, *list_weight)
        self.gr2 = GConv_RNN(dims, gcn_depth, *list_weight)
        self.gc1 = GConv_RNN(dims, gcn_depth, *list_weight)
        self.gc2 = GConv_RNN(dims, gcn_depth, *list_weight)

    def forward(self, x, state, nodevec1, nodevec2):
        """
            x shape: (B, N, input_dim)
            state shape: (B, N, hidden_dim)
            nodevec1 & nodevec2 shape: (N, node_dim)
        """
        adp, adpT = self.dgcn(x, state, nodevec1, nodevec2)
        combined = torch.cat((x, state), dim=-1)
        z = torch.sigmoid(self.gz1(combined, adp) + self.gz2(combined, adpT))
        r = torch.sigmoid(self.gr1(combined, adp) + self.gr2(combined, adpT))
        candidate = torch.cat((x, r * state), dim=-1)
        hc = torch.tanh(self.gc1(candidate, adp) + self.gc2(candidate, adpT))
        h = z * state + (1 - z) * hc
        return h

    def init_hidden_state(self, batch_size):
        hidden_state = Variable(torch.zeros(batch_size, self.num_nodes, self.hidden_dim))
        nn.init.orthogonal_(hidden_state)
        return hidden_state


class DGCRN(nn.Module):
    def __init__(self, 
                 gcn_depth, 
                 num_nodes, 
                 predefined_A=None,
                 node_dim=40, 
                 middle_dim=2,
                 seq_length=12,
                 input_dim=2, 
                 list_weight=[0.05, 0.95, 0.95], 
                 tanhalpha=3, 
                 cl_decay_steps=4000, 
                 rnn_size=64, 
                 hyperGNN_dim=16):
        super(DGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_size
        self.output_dim = 1
        self.ycov_dim = 1
        self.seq_length = seq_length
        self.predefined_A = predefined_A
        self.gcn_depth = gcn_depth
        self.use_curriculum_learning = True
        self.cl_decay_steps = cl_decay_steps

        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)
        self.idx = torch.arange(self.num_nodes)

        # The encoder and decoder in DGCRN all has only one layer.
        self.encoder = DGCRM(num_nodes, self.input_dim, self.hidden_dim, hyperGNN_dim, middle_dim, 
                             node_dim, tanhalpha, gcn_depth, predefined_A, list_weight)
        self.decoder = DGCRM(num_nodes, self.output_dim + self.ycov_dim, self.hidden_dim, hyperGNN_dim, middle_dim, 
                             node_dim, tanhalpha, gcn_depth, predefined_A, list_weight)
        self.fc_final = nn.Linear(self.hidden_dim, self.output_dim)
    
    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    def forward(self, x, ycl, batches_seen=None, task_level=12):
        """
            x shape: (B, T, N, 2)
            ycl shape: (B, T, N, 2)
            The first feature is spreed V(t), and the second feature is time of day T(t)
        """
        batch_size = x.size(0)
        self.idx = self.idx.to(x.device)
        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)
        # 1. DGCRN_Encoder
        output_hidden = []
        hidden_state = self.encoder.init_hidden_state(batch_size).to(x.device)
        for t in range(self.seq_length):
            hidden_state = self.encoder(x[:, t, :, :], hidden_state, nodevec1, nodevec2)
            output_hidden.append(hidden_state)
        # 2. DGCRN_Decoder
        go_symbol = torch.zeros(batch_size, self.num_nodes, self.output_dim).to(x.device)
        timeofday = ycl[:, :, :, 1:]
        decoder_input = go_symbol
        hidden_state = output_hidden[-1]
        outputs_final = []
        for i in range(task_level):  # task_level gradually converges from 1 to seq_length
            try:
                decoder_input = torch.cat((decoder_input, timeofday[:, i, :, :]), dim=-1)
            except:
                print(decoder_input.shape, timeofday.shape)
                sys.exit(0)
            hidden_state = self.decoder(decoder_input, hidden_state, nodevec1, nodevec2)
            decoder_output = self.fc_final(hidden_state)
            outputs_final.append(decoder_output)
            # curriculum learning
            decoder_input = decoder_output
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = ycl[:, i, :, :1]
        
        # outputs_final shape: (B, task_level, num_nodes, output_dim)
        outputs_final = torch.stack(outputs_final, dim=1)
        random_predict = torch.zeros(batch_size, self.seq_length - task_level, self.num_nodes, self.output_dim).to(outputs_final.device)
        outputs = torch.cat((outputs_final, random_predict), dim=1)   # (B, seq_length, num_nodes, output_dim)
        return outputs


if __name__ == "__main__":
    adj_mx = torch.randn(170, 170)
    predefined_A = [adj_mx, adj_mx.T]  # doubletransition
    model = DGCRN(
        gcn_depth=2, 
        num_nodes=170, 
        predefined_A=predefined_A,
        node_dim=40, 
        middle_dim=2,
        seq_length=12,
        input_dim=2, 
        list_weight=[0.05, 0.95, 0.95], 
        tanhalpha=3, 
        cl_decay_steps=4000, 
        rnn_size=64, 
        hyperGNN_dim=16
    )
    x = torch.randn(1, 12, 170, 2)
    ycl = torch.randn(1, 12, 170, 2) 
    y = model(x, ycl, batches_seen=1)
    print(y.size())