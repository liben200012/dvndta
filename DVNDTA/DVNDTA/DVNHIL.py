import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool
import torch.nn as nn

class DVNHIL(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int, 
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(DVNHIL, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())

        self.virtualnode_embedding_protein = torch.nn.Embedding(1, self.in_channels)
        self.virtualnode_embedding_ligand = torch.nn.Embedding(1, self.in_channels)
        torch.nn.init.constant_(self.virtualnode_embedding_protein.weight.data, 0)
        torch.nn.init.constant_(self.virtualnode_embedding_ligand.weight.data, 0)

        self.mlp_virtualnode_protein = torch.nn.Sequential(
                        nn.Linear(self.in_channels, 2 * self.out_channels),
                        nn.LeakyReLU(negative_slope = 0.1),
                        nn.Linear(2 * self.out_channels, self.in_channels),
                        nn.LeakyReLU(negative_slope = 0.1),
                    )


        self.mlp_virtualnode_ligand = torch.nn.Sequential(
                        nn.Linear(self.in_channels, 2 * self.out_channels),
                        nn.LeakyReLU(negative_slope=0.1),
                        nn.Linear(2 * self.out_channels, self.in_channels),
                        nn.LeakyReLU(negative_slope = 0.1),
                    )

        self.gate_protein_inter = nn.Sequential(
            nn.Linear(self.in_channels, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        self.gate_ligand_inter = nn.Sequential(
            nn.Linear(self.in_channels, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.deg_coef = nn.Parameter(torch.zeros(1, self.in_channels, 2))
        nn.init.xavier_normal_(self.deg_coef)

        self.vp_update_scale = nn.Parameter(torch.tensor(0.5))
        self.vl_update_scale = nn.Parameter(torch.tensor(0.5))


    def forward(self, x, edge_index_intra, edge_index_inter, pos=None,
                size=None, data=None, v_emb_p=None, v_emb_l=None, degree_intra=None, degree_inter=None):
        batch = data.batch
        ligand_nodes = data.split == 0
        protein_nodes = data.split == 1

        if v_emb_p is None:
            v_emb_p = self.virtualnode_embedding_protein(
                torch.zeros(batch[-1].item() + 1, device='cuda:0').to(edge_index_intra.dtype)
            )
            v_emb_l = self.virtualnode_embedding_ligand(
                torch.zeros(batch[-1].item() + 1, device='cuda:0').to(edge_index_intra.dtype)
            )
            v_emb_p_temp = global_mean_pool(x[protein_nodes], batch[protein_nodes]) + v_emb_p
            v_emb_l_temp = global_mean_pool(x[ligand_nodes], batch[ligand_nodes]) + v_emb_l
            v_emb_p = self.mlp_virtualnode_protein(v_emb_p_temp)
            v_emb_l = self.mlp_virtualnode_ligand(v_emb_l_temp)

        row_cov, col_cov = edge_index_intra
        coord_diff_cov = pos[row_cov] - pos[col_cov]
        radial_cov = self.mlp_coord_cov(_rbf(torch.norm(coord_diff_cov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_intra = self.propagate(edge_index=edge_index_intra, x=x, radial=radial_cov, size=size)

        x_1 = x
        x_2 = x
        x_2 = torch.stack([x_2, x_2 * degree_inter], dim=-1)
        x_2 = (x_2 * self.deg_coef).sum(dim=-1)
        gate_protein_weights = self.gate_protein_inter(x_2[protein_nodes])
        gate_ligand_weights = self.gate_ligand_inter(x_2[ligand_nodes])

        x_1[protein_nodes] = ((1 - gate_protein_weights) * x_1[protein_nodes]) + (gate_protein_weights * v_emb_l[batch[protein_nodes]])
        x_1[ligand_nodes] = ((1 - gate_ligand_weights) * x_1[ligand_nodes]) + (gate_ligand_weights * v_emb_p[batch[ligand_nodes]])

        row_ncov, col_ncov = edge_index_inter
        coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
        radial_ncov = self.mlp_coord_ncov(_rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_inter = self.propagate(edge_index=edge_index_inter, x=x_1, radial=radial_ncov, size=size)

        out_node = self.mlp_node_cov(x + out_node_intra) + self.mlp_node_ncov(x_1 + out_node_inter) + x

        v_emb_p_temp = global_mean_pool(out_node[protein_nodes], batch[protein_nodes]) + v_emb_p
        v_emb_l_temp = global_mean_pool(out_node[ligand_nodes], batch[ligand_nodes]) + v_emb_l
        v_emb_p = v_emb_p + self.mlp_virtualnode_protein(v_emb_p_temp) * self.vp_update_scale
        v_emb_l = v_emb_l + self.mlp_virtualnode_ligand(v_emb_l_temp) * self.vl_update_scale

        return out_node, v_emb_p, v_emb_l

    def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
        x = x_j * radial

        return x


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

# %%