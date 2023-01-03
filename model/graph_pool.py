import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_scatter import scatter_add, scatter_max, scatter_mean

from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from le_conv import LEConv
from torch_geometric.data import Batch, Data
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils.loop import add_self_loops
from torch_geometric.nn.pool.topk_pool import topk

from torch_sparse import coalesce
from torch_sparse import transpose
from torch_sparse import spspmm

from asap_pool import ASAP_Pooling

# torch.set_num_threads(1)

def readout(x, batch):
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0) 
    return torch.cat((x_mean, x_max), dim=-1)

class ASAP_Pool(nn.Module):
    def __init__(self, config):
        super(ASAP_Pool, self).__init__()
        self.ratio = 0.8
        if type(ratio)!=list:
            ratio = [ratio for i in range(config.graph_layers)]

        self.num_features = config.slot_size
        self.node_features = self.num_features
        self.hidden = config.graph_hidden	
        self.num_layers = config.graph_layers
        self.undirected = config.undirected
        self.self_loops = config.self_loops
        self.num_heads  = config.graph_heads
        self.dropout_att = config.graph_drop	
        #self.dropout = dropout

        self.embeddings         = nn.Embedding(self.num_features, self.node_features, padding_idx=-1) # Embeddings for the strategies (num_features is num_strategies)
        #self.embeddings.weight  = nn.Parameter(torch.FloatTensor(np.diag(np.diag(np.ones((self.num_features, self.node_features))))))  # diag matrix of 1 hot
        self.conv1 = GATConv(self.node_features, self.hidden, heads=self.num_heads)
        self.pool1 = ASAP_Pooling(in_channels=self.hidden *self.num_heads, ratio=self.ratio[0], dropout_att=self.dropout_att)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GATConv(self.hidden, self.hidden, heads=self.num_heads))
            self.pools.append(ASAP_Pooling(in_channels=self.hidden, ratio=self.ratio[i], dropout_att=self.dropout_att))
        self.lin1 = nn.Linear(2*self.hidden * self.num_heads, self.hidden) # 2*hidden due to readout layer
        self.lin2 = nn.Linear(self.hidden, self.num_features-1)
        self.reset_parameters()

    def reset_parameters(self):
        self.embeddings.reset_parameters()
        self.conv1.reset_parameters()
        self.pool1.reset_parameters()
        for conv, pool in zip(self.convs, self.pools):
            conv.reset_parameters()
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, graph_prompt_ids, batch_graph):
        #data_list = self.convert_ontology_to_graph(ds_ids, ds_mask, slot_connect)
        #data = Batch.from_data_list(data_list)#.to(feats.device)
        x, edge_index, batch = batch_graph.x, batch_graph.edge_index, batch_graph.batch
        x = self.embeddings(x.squeeze(1))  # added  # x is num_graph x node_feats / 22
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_weight, batch, perm, S_index, S_weight, att_wts = self.pool1(x=x, edge_index=edge_index, edge_weight=None, batch=batch)
        #save_perm = perm.clone()
        xs = readout(x, batch)
        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x=x, edge_index=edge_index))
            x, edge_index, edge_weight, batch, perm, _, _, _ = pool(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
            xs += readout(x, batch)
        x = F.relu(self.lin1(xs))
        x = F.dropout(x, p=0.5, training=self.training)
        ####logits = self.lin2(x)
        #out = F.log_softmax(x, dim=-1)

        #if return_extra:
        #    gat_attn_wts = self.conv1.attention_score
        #    return logits, batch_graph.y, (S_index, S_weight, att_wts, save_perm, batch_graph, gat_attn_wts)
        
        return graph_prompt_embeds #logits, data.y, None #out


    def __repr__(self):
        return self.__class__.__name__
