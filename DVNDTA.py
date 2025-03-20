# %%
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from DVNHIL import DVNHIL
from torch_geometric.utils import degree

class DVNDTA(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.lin_node = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())

        self.gconv1 = DVNHIL(hidden_dim, hidden_dim)
        self.gconv2 = DVNHIL(hidden_dim, hidden_dim)
        self.gconv3 = DVNHIL(hidden_dim, hidden_dim)
        self.gconv4 = DVNHIL(hidden_dim, hidden_dim)

        self.fc = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

    def forward(self, data):
        x, edge_index_intra, edge_index_inter, pos, edge_attr = \
            data.x, data.edge_index_intra, data.edge_index_inter, data.pos, data.edge_attr

        data.x = self.lin_node(x)

        degree_intra = degree(edge_index_intra[1], num_nodes=x.size(0), dtype=torch.float).unsqueeze(-1)
        degree_inter = degree(edge_index_inter[1], num_nodes=x.size(0), dtype=torch.float).unsqueeze(-1)
        degree_inter = torch.log(degree_inter + 1)

        x, v_emb_p, v_emb_l = self.gconv1(data.x, edge_index_intra, edge_index_inter, pos=pos, data=data,degree_intra=degree_intra,degree_inter=degree_inter)
        x, v_emb_p, v_emb_l = self.gconv2(x, edge_index_intra, edge_index_inter, pos=pos, data=data, v_emb_p=v_emb_p, v_emb_l=v_emb_l, degree_intra=degree_intra,degree_inter=degree_inter)
        x, v_emb_p, v_emb_l = self.gconv3(x, edge_index_intra, edge_index_inter, pos=pos, data=data, v_emb_p=v_emb_p, v_emb_l=v_emb_l, degree_intra=degree_intra,degree_inter=degree_inter)
        x, v_emb_p, v_emb_l = self.gconv4(x, edge_index_intra, edge_index_inter, pos=pos, data=data, v_emb_p=v_emb_p, v_emb_l=v_emb_l, degree_intra=degree_intra,degree_inter=degree_inter)

        data.x = x

        x = global_add_pool(data.x, data.batch)
        x = self.fc(x)
        return x.view(-1)


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h

# %%