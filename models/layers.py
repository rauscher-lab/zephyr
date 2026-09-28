import copy
import typing
import warnings

import numpy as np
import torch
from torch import nn

import config
import utils
from utils import symmetrize, remove_centroid, index_softmax, RigidBodyMotion, Positive


class LambdaMod(nn.Module):
    def __init__(self, Lambda: typing.Callable, name=''):
        super().__init__()
        self.Lambda = utils.Lambda(Lambda)
        self.name = name

    def forward(self, x):
        return self.Lambda(x)

    def extra_repr(self) -> str:
        return f"{self.Lambda}"


class EquiCrossProdLayer(nn.Module):
    def __init__(self, node_feat_d: int, hidden_d: int, agg=config.pos_msg_agg, epsilon_x=config.epsilon_x,
                 res_con=True, bias=config.model_bias, act_fn=config.act_fn,
                 norm_layer=config.norm_layer, norm_layer_bias=config.norm_layer_bias,
                 phi_weight=config.phi_x_weight, dropout=config.dropout,
                 use_edge_feat=config.cross_prod_use_edge_feat, edge_feat_d=0,
                 phi_weight_act=config.cross_prod_layer_phi_weight_act):
        super().__init__()
        self.hidden_d = hidden_d
        self.node_feat_d = node_feat_d
        self.agg = agg
        self.res_con = res_con
        self.use_edge_feat = use_edge_feat

        # Define the module that scales the cross products in the position update
        # x'_i = x_i + sum_jk x_ij x x_ik/(d_ij d_ik + epsilon_x) phi(h_i,h_j,h_k,d_ij,d_ik,d_jk,e_ij,e_ik)
        # e_ij and e_ik are optional edge features
        phi_input_dim = 3 * node_feat_d + 3
        if self.use_edge_feat:
            phi_input_dim += 2 * edge_feat_d
        self.phi = nn.Sequential(nn.Linear(phi_input_dim, hidden_d, bias=bias),
                                 act_fn,
                                 nn.Linear(hidden_d, hidden_d, bias=bias),
                                 act_fn,
                                 nn.Linear(hidden_d, 1, bias=bias))

        # Add normalization layers after the activation function
        if norm_layer == 'layernorm':
            self.phi.insert(2, nn.LayerNorm(hidden_d, bias=norm_layer_bias))
            self.phi.insert(5, nn.LayerNorm(hidden_d, bias=norm_layer_bias))

        # Add module to weight each phi_x output
        if phi_weight:
            if phi_weight_act == 'sigmoid':
                weight_act = nn.Sigmoid()
            elif phi_weight_act == 'exp':
                weight_act = LambdaMod(Lambda=lambda x: torch.exp(x))
            elif phi_weight_act == 'exp_clamped':
                weight_act = LambdaMod(Lambda=lambda x: torch.exp(x.clamp(min=-10, max=10)))
            elif phi_weight_act in [None, 'local_softmax']:
                weight_act = LambdaMod(Lambda=lambda x: x)
            else:
                raise ValueError(f"weight activation {config.phi_weight_act!r} is not supported.")

            self.phi_weight = nn.Sequential(nn.Linear(phi_input_dim, 1, bias=bias), weight_act)
            self.phi_weight_is_softmax = phi_weight_act == 'local_softmax'
        else:
            self.phi_weight = None

        # Add dropout layers before all nn.Linear in all phi modules.
        if dropout > 0:
            phi_mod = [self.phi, self.phi_weight]
            phi_mod = [p for p in phi_mod if p is not None]
            for phi_temp in phi_mod:
                Lin_mod_ind = [i for i, n in enumerate(phi_temp) if isinstance(n, nn.Linear)]
                for ind in reversed(Lin_mod_ind):
                    phi_temp.insert(ind, nn.Dropout(dropout))

        # Buffers
        self.epsilon_x = nn.parameter.Buffer(torch.as_tensor(epsilon_x))
        self.epsilon = nn.parameter.Buffer(torch.as_tensor(1e-6))

    def forward(self, node_pos: torch.Tensor, node_feat: torch.Tensor, triplet_ind: torch.Tensor,
                edge_feat: torch.Tensor = None, triplet_edges_ind: torch.Tensor = None):
        """
        Performs an update of the node position using vector representations of triplets of node positions (x_i,x_j,x_k)
        Args:
            node_pos: node positions. shape=(B,N,3)
            node_feat: node features. shape=(B,N,d)
            triplet_ind: sets of node indices defining the terms in the position update sum.
                         For each [i,j,k] in triplet_ind, w_ijk*phi(input_ijk)*(x_ij x x_ik)/(d_ij * d_ik) is added to x_i.
                         w_ijk = phi_weight(input_ijk)
                              or exp(phi_weight(input_ijk))/sum_(l=i) exp(phi_weight(input_ljk)) if using 'local_softmax'
            edge_feat: (optional) edge features. shape=(B,E,d)
            triplet_edges_ind: (optional) edges index of edges involved in each triplet.
        Returns:
            updated node position of shape=(B,N,3)
        """

        # Flatten the batch and node dimension
        B, N, _ = node_feat.shape
        BN = B * N
        node_pos = node_pos.flatten(0, 1)
        node_feat = node_feat.flatten(0, 1)
        triplet_ind = torch.flatten(torch.arange(B, device=node_feat.device).reshape(-1, 1, 1) * N + triplet_ind, 0, 1)

        # Compute edge vectors [x_ij, x_ik, x_jk] and their norm
        edges_vec = node_pos[triplet_ind[:, [1, 2, 2]]] - node_pos[triplet_ind[:, [0, 0, 1]]]  # shape=(B*N,3,3)
        edges_norm = torch.linalg.norm(edges_vec, dim=-1).unsqueeze(-1)
        edges_unit_vec = edges_vec / (edges_norm + self.epsilon_x)

        # Compute the position update for each node
        vec_msg = self.compute_vec_msg(edges_unit_vec)
        phi_input = [node_feat[triplet_ind].flatten(-2, -1), edges_norm.squeeze(-1)]

        # Add edge features, if given
        if self.use_edge_feat:
            phi_input.append(edge_feat[triplet_edges_ind.flatten(0, 1)].flatten(-2, -1))
        phi_input = torch.cat(phi_input, dim=-1)
        phi_output = self.phi(phi_input)

        if self.phi_weight is not None:
            phi_weight = self.phi_weight(phi_input)
            if self.phi_weight_is_softmax:
                phi_weight = index_softmax(src=phi_weight, dim=0, index=triplet_ind[:, 0], size=(BN, 1))
            phi_output *= phi_weight
        node_pos_delta = vec_msg * phi_output

        # Accumulate the node_pos delta onto the first node of each triplet.
        node_pos_new = torch.zeros_like(node_pos)
        node_pos_new.index_add_(dim=0, index=triplet_ind[:, 0], source=node_pos_delta)

        # Normalize update
        if self.agg.startswith('sum'):
            pass
        elif self.agg.startswith('weightN'):
            phi_weight_sum = self.epsilon * node_pos.new_ones(size=(BN, 1))
            phi_weight_sum.index_add_(dim=0, index=triplet_ind[:, 0], source=phi_weight)
            node_pos_new /= phi_weight_sum
        else:
            raise NotImplementedError

        # Add residual connection
        if self.res_con:
            node_pos_new += node_pos

        # Unflatten node and batch dimension
        node_pos_new = node_pos_new.unflatten(dim=0, sizes=(B, N))

        return node_pos_new

    def compute_vec_msg(self, neigh_edges_unit_vec: torch.Tensor) -> torch.Tensor:
        """
        Computes the vector message aggregated onto node i.
        Args:
            neigh_edges_unit_vec: Unit vectors of the neighborhood's edges. shape=(E,3,3)
                E corresponds to the total number of node triplets considered.
                dim=1 indexes the 3 edge vectors x_ij, x_ik and x_jk that can be formed from the triplet [i,j,k].
                dim=2 corresponds to the 3D component of each vector.

        Returns:
            set of vectors of shape (E,3)
        """

        # Compute the cross product of x_ij with x_ik. The x_jk vector is not used.
        x_ijk_cross = torch.linalg.cross(neigh_edges_unit_vec[:, 0], neigh_edges_unit_vec[:, 1])
        return x_ijk_cross


class MessagePassingLayer(nn.Module):
    def __init__(self, msg_d: int, phi_msg: nn.Module, phi_node_pos: nn.Module = None,
                 phi_msg_weight: nn.Module = None, phi_node_pos_weight: nn.Module = None,
                 phi_node_feat: nn.Module = None, phi_edge_feat: nn.Module = None, phi_global_feat: nn.Module = None,
                 node_pos_agg='sum', node_feat_agg='sum', epsilon_x=1.0, epsilon=1e-8,
                 node_pos_res=True, node_feat_res=True, edge_feat_res=True, global_feat_res=True,
                 d_inv_edge_feat=False, node_feat_agg_in_edge_updt=None):
        super().__init__()
        self.msg_d = msg_d
        self.phi_msg = phi_msg
        self.phi_msg_weight = phi_msg_weight
        self.phi_node_pos = phi_node_pos
        self.phi_node_pos_weight = phi_node_pos_weight
        self.phi_node_feat = phi_node_feat
        self.phi_edge_feat = phi_edge_feat
        self.phi_global_feat = phi_global_feat

        self.node_pos_agg = node_pos_agg
        self.node_feat_agg = node_feat_agg

        self.node_pos_res = node_pos_res
        self.node_feat_res = node_feat_res
        self.edge_feat_res = edge_feat_res
        self.global_feat_res = global_feat_res

        # Determine if softmax weights are used for msg aggregation and position updates
        self.phi_msg_weight_is_softmax = (self.phi_msg_weight is not None
                                          and isinstance(self.phi_msg_weight[-1], LambdaMod)
                                          and self.phi_msg_weight[-1].name == 'local_softmax')
        self.phi_pos_weight_is_softmax = (self.phi_node_pos_weight is not None
                                          and isinstance(self.phi_node_pos_weight[-1], LambdaMod)
                                          and self.phi_node_pos_weight[-1].name == 'local_softmax')

        # Misc
        self.d_inv_edge_feat = d_inv_edge_feat
        self.node_feat_agg_in_edge_updt = node_feat_agg_in_edge_updt
        self.compute_neigh_size = any([self.node_feat_agg == c for c in ['sumN', 'mean']])
        self.compute_neigh_size |= any([self.node_pos_agg.startswith(c) for c in ['sumN', 'mean']])

        # Buffers
        self.sqrt2 = nn.parameter.Buffer(torch.tensor(2).sqrt())
        self.epsilon_x = nn.parameter.Buffer(torch.as_tensor(epsilon_x))
        self.epsilon = nn.parameter.Buffer(torch.tensor(epsilon))
        self.epsilon_d_inv = nn.parameter.Buffer(torch.tensor(config.epsilon_d_inv))

    def forward(self, node_pos: torch.Tensor = None, node_feat: torch.Tensor = None, edge_feat: torch.Tensor = None,
                global_feat: torch.Tensor = None,
                extra_node_feat: torch.Tensor = None, extra_edge_feat: torch.Tensor = None,
                extra_global_feat: torch.Tensor = None,
                nodes_adj: torch.Tensor = None, edges_ind: torch.Tensor = None, **kwargs):
        # Notation
        # x_i: position of node i
        # h_i: feature vector of node i
        # e_ij: feature vector of edge from node i to node j
        # g: global feature
        # m_ij: message passed from node i to node j
        # wm_ij: weight of message m_ij
        # wx_ij: weight of position update message from node j to node i
        tensor_model = node_feat if node_pos is None else node_pos
        B, N = tensor_model.shape[:2]
        device = tensor_model.device

        # Concatenate extra features
        if extra_node_feat is not None:
            node_feat = torch.cat([node_feat, extra_node_feat], dim=-1)
        if extra_edge_feat is not None:
            edge_feat = torch.cat([edge_feat, extra_edge_feat], dim=-1)
        if extra_global_feat is not None:
            global_feat = torch.cat([global_feat, extra_global_feat], dim=-1)

        # Combine the batch and node index to define a global index
        edges_ind = edges_ind[:, :1] * N + edges_ind[:, 1:]
        nodes_batch_ind = torch.arange(B, device=device).unsqueeze(-1).expand((-1, N)).flatten(0, 1)
        if node_pos is not None:
            node_pos = node_pos.flatten(0, 1)
        if node_feat is not None:
            node_feat = node_feat.flatten(0, 1)

        # Combine batch and node index of all kwargs whose name starts with 'node'
        reshaped_kwargs = []
        for kwarg_name, kwarg_val in kwargs.items():
            if kwarg_name.startswith('node') and kwarg_val is not None and kwarg_val.shape[:2] == (B, N):
                kwargs[kwarg_name] = kwarg_val.flatten(0, 1)
                reshaped_kwargs.append(kwarg_name)

        if node_pos is not None:
            edges_vec = torch.diff(node_pos[edges_ind], dim=-2).squeeze(dim=-2)  # v_01 = x_1 - x_0
            edges_norm = torch.linalg.norm(edges_vec, dim=-1, keepdim=True)
        else:
            edges_vec, edges_norm = None, None
        if self.d_inv_edge_feat:
            edges_norm_inv = 1 / (edges_norm + self.epsilon_d_inv)
        else:
            edges_norm_inv = None
        compute_kwargs = dict(node_pos=node_pos, node_feat=node_feat, edge_feat=edge_feat, edges_ind=edges_ind,
                              edges_vec=edges_vec, edges_norm=edges_norm, edges_norm_inv=edges_norm_inv,
                              global_feat=global_feat, nodes_batch_ind=nodes_batch_ind, nodes_adj=nodes_adj)
        node_pos_new, node_feat_new = self.compute_node_pos_and_feat(**compute_kwargs, **kwargs)
        edge_feat_new = self.compute_edge_feat(**compute_kwargs, **kwargs)
        global_feat_new = self.compute_global_feat(**compute_kwargs, **kwargs)

        # Re-insert the batch dimension
        if node_pos is not None:
            node_pos_new = node_pos_new.unflatten(dim=0, sizes=(B, N))
        if node_feat is not None:
            node_feat_new = node_feat_new.unflatten(dim=0, sizes=(B, N))
        for kwarg_name in reshaped_kwargs:
            kwargs[kwarg_name] = kwargs[kwarg_name].unflatten(dim=0, sizes=(B, N))

        return node_pos_new, node_feat_new, edge_feat_new, global_feat_new

    def compute_node_pos_and_feat(self, node_feat: torch.Tensor, edges_ind: torch.Tensor, nodes_batch_ind=None,
                                  node_pos: torch.Tensor = None, edge_feat: torch.Tensor = None,
                                  global_feat: torch.Tensor = None, nodes_adj: torch.Tensor = None,
                                  edges_vec: torch.Tensor = None, edges_norm: torch.Tensor = None,
                                  edges_norm_inv: torch.Tensor = None,
                                  **kwargs):
        # Notation
        # x_i: position of node i
        # h_i: feature vector of node i
        # e_ij: feature vector of edge from node i to node j
        # g: global feature
        # m_ij: message passed from node i to node j
        # wm_ij: weight of message m_ij
        # wx_ij: weight of position update message from node j to node i
        N, dtype, device = node_feat.shape[0], node_feat.dtype, node_feat.device

        # Compute the size of the neighborhood of each node. Only needed for certain aggregation methods
        if self.compute_neigh_size:
            if nodes_adj is not None:
                neigh_size = nodes_adj.flatten(0, 1).sum(dim=-1, keepdim=True)  # shape=(N,1)
            else:
                neigh_size = torch.bincount(edges_ind.flatten()).unsqueeze(-1)  # shape=(N,1)
        else:
            neigh_size = None

        # Messages need to be passed for each edge direction.
        # Re-order edge dimension as follows: [edge_1 forward, edge_1 backward, edge_2 forward, edge_2 backward, ...]
        # edges_ind consists of unique pairs of node indices connected by an edge
        src_ind = edges_ind.flatten()  # = [[edges_ind[0,0],edges_ind[0,1],edges_ind[1,0],...]
        target_ind = edges_ind.flip(dims=[-1]).flatten()  # = [[edges_ind[0,1],edges_ind[0,0],edges_ind[1,1],...]
        if node_pos is not None:
            all_edges_norm = edges_norm.repeat_interleave(2, dim=0)

        # Format input to the phi modules:
        phi_input = [node_feat[target_ind], node_feat[src_ind]]
        if edge_feat is not None:
            phi_input.append(edge_feat.repeat_interleave(2, dim=0))
        if node_pos is not None:
            phi_input.append(all_edges_norm)  # Add d_ij to phi input
            if self.d_inv_edge_feat:
                all_edges_norm_inv = edges_norm_inv.repeat_interleave(2, dim=0)  # Norm of each edge for each direction
                phi_input.append(all_edges_norm_inv)  # Add 1/d_ij to phi input
        if global_feat is not None:
            phi_input.append(global_feat[nodes_batch_ind[target_ind], :])
        phi_input = torch.cat(phi_input, dim=1)

        # Compute messages for updating the node features
        # m_ij = phi_msg(h_i, h_j, e_ij, g)
        node_msg = self.phi_msg(phi_input)  # shape=(N_edges,msg_d)

        # Compute the message weights wm_ij
        if self.phi_msg_weight is not None:
            node_msg_weight = self.phi_msg_weight(node_msg)

            # Normalize weights with softmax across the neighborhood
            if self.phi_msg_weight_is_softmax:
                node_msg_weight = index_softmax(node_msg_weight, dim=0, index=target_ind, size=(N, 1))
            node_msg = node_msg_weight * node_msg

        # Compute the message sum sum_j wm_ij m_ij
        node_msg_sum = torch.zeros((N, self.msg_d), dtype=node_msg.dtype, device=device)
        node_msg_sum.index_add_(dim=0, index=target_ind, source=node_msg)

        # Normalize the message sum
        if self.node_feat_agg == 'sumN':
            node_msg_sum = node_msg_sum / neigh_size.sqrt()
        elif self.node_feat_agg == 'mean':
            node_msg_sum = node_msg_sum / neigh_size
        elif self.node_feat_agg == 'weightN':
            # Only normalize if not using softmax. Otherwise, normalization is done intrinsically in the weights
            if not self.phi_msg_weight_is_softmax:
                node_msg_weight_sum = self.epsilon * torch.ones((N, 1), dtype=dtype, device=device)
                node_msg_weight_sum.index_add_(dim=0, index=target_ind, source=node_msg_weight)
                node_msg_sum = node_msg_sum / node_msg_weight_sum
        elif self.node_feat_agg == 'sum':
            pass
        else:
            raise NotImplementedError(f"feat_msg_agg={self.node_feat_agg!r} is not implemented")

        # Compute the new node features with the accumulated messages
        # h_new_i = h_i + phi_h(h_i, sum_j wm_ij m_ij)
        if self.phi_node_feat is not None:
            phi_h_input = [node_feat, node_msg_sum]
            phi_h_input = torch.cat(phi_h_input, dim=-1)
            node_features_new = self.phi_node_feat(phi_h_input)

            # Add residual connection
            if self.node_feat_res:
                node_features_new = node_features_new + node_feat
                if self.node_feat_agg.startswith('sumN'):
                    node_features_new = node_features_new / self.sqrt2
        else:
            node_features_new = None

        # Compute the new node positions
        # x_new_i = x_i + sum_j (x_i - x_j)/D_i wx_ij phi_x(h_i, h_j, d_ij, e_ij, g)  (D_i is defined below)
        if self.phi_node_pos is not None:
            node_pos_new = torch.zeros_like(node_pos)

            phi_node_pos_output = self.phi_node_pos(phi_input)  # shape=(N_edges,1)
            if self.phi_node_pos_weight is not None:
                phi_node_pos_weight = self.phi_node_pos_weight(phi_input)

                # Normalize weights with softmax across the neighborhood
                if self.phi_pos_weight_is_softmax:
                    phi_node_pos_weight = index_softmax(phi_node_pos_weight, dim=0, index=target_ind, size=(N, 1))
                phi_node_pos_output = phi_node_pos_weight * phi_node_pos_output

            # Compute the position difference associated with each edge for each direction
            all_edges_vec = torch.cat([-edges_vec.unsqueeze(-2), edges_vec.unsqueeze(-2)], dim=-2).flatten(0, 1)
            node_pos_summand = all_edges_vec * phi_node_pos_output

            if '_unnorm' not in self.node_pos_agg:
                node_pos_summand /= (all_edges_norm + self.epsilon_x)
            node_pos_new.index_add_(dim=0, index=target_ind, source=node_pos_summand)

            # Compute the normalization constant D_i
            if self.node_pos_agg.startswith('sumN'):
                node_pos_new = node_pos_new / torch.sqrt(2 * neigh_size + self.epsilon)
            elif self.node_pos_agg.startswith('mean'):
                node_pos_new = node_pos_new / (neigh_size + self.epsilon)
            elif self.node_pos_agg.startswith('weightN'):
                # Only normalize if not using softmax. Otherwise, normalization is done intrinsically in the weights
                if not self.phi_pos_weight_is_softmax:
                    node_pos_weight_sum = self.epsilon * torch.ones((N, 1), dtype=dtype, device=device)
                    node_pos_weight_sum.index_add_(dim=0, index=target_ind, source=phi_node_pos_weight)
                    node_pos_new = node_pos_new / node_pos_weight_sum
            elif self.node_pos_agg.startswith('sum'):
                pass
            else:
                raise NotImplementedError(f"pos_msg_agg={self.node_pos_agg!r} is not implemented")

            # Add residual connection
            if self.node_pos_res:
                node_pos_new = node_pos + node_pos_new
                if self.node_pos_agg.startswith('sumN'):
                    node_pos_new = node_pos_new / self.sqrt2
        else:
            node_pos_new = None

        # Checks
        if 0 and config.debug and node_pos is not None:
            pos_delta_mean_max = (node_pos_new - node_pos).mean(dim=1).abs().max()
            if pos_delta_mean_max > 1e-3:
                print('pos_delta_mean_max:', pos_delta_mean_max)

        return node_pos_new, node_features_new

    def compute_edge_feat(self, node_feat: torch.Tensor, edges_ind: torch.Tensor, edges_norm: torch.Tensor = None,
                          edges_norm_inv: torch.Tensor = None, edge_feat: torch.Tensor = None,
                          global_feat: torch.Tensor = None, nodes_batch_ind=None, **kwargs):
        # e_ij_new = e_ij + phi_e(HH_ij, d_ij, e_ij, g)
        #     H_ij = h_i + h_j   if self.node_feat_agg_in_edge_updt = 'sum'
        #          = |h_i - h_j| if self.node_feat_agg_in_edge_updt = 'abs_diff'
        #          = [h_i, h_j]  if self.node_feat_agg_in_edge_updt = None
        if self.phi_edge_feat is None:
            edge_feat_new = edge_feat
        else:
            # Aggregate node features for each pair of nodes connected by an edge
            if self.node_feat_agg_in_edge_updt == 'sum':
                node_feats = node_feat[edges_ind].sum(dim=-2)
            elif self.node_feat_agg_in_edge_updt == 'abs_diff':
                node_feats = node_feat[edges_ind].diff(dim=-2).flatten(-2, -1).abs()
            elif self.node_feat_agg_in_edge_updt is None:
                node_feats = node_feat[edges_ind].flatten(-2, -1)
            else:
                raise NotImplementedError(f"agg_method={self.node_feat_agg_in_edge_updt!r} is not implemented.")

            phi_input = [node_feats, edge_feat]
            if global_feat is not None:
                phi_input.append(global_feat[nodes_batch_ind[edges_ind[:, 0]], :])
            if edges_norm is not None:
                phi_input.append(edges_norm)  # Add d_ij to phi input
            if self.d_inv_edge_feat:
                phi_input.append(edges_norm_inv)  # Add 1/(d_ij + eps) to phi input
            phi_input = torch.cat(phi_input, dim=-1)
            edge_feat_new = self.phi_edge_feat(phi_input)

            # Add residual connection
            if self.edge_feat_res:
                edge_feat_new = edge_feat_new + edge_feat

        return edge_feat_new

    def compute_global_feat(self, node_feat: torch.Tensor, global_feat: torch.Tensor = None,
                            nodes_batch_ind: torch.Tensor = None, **kwargs):
        # g_new = g + sum_i phi_g(h_i, g)
        if global_feat is None:
            global_feat_new = global_feat
        else:
            global_feat_new = torch.zeros_like(global_feat)
            phi_input = torch.cat([node_feat, global_feat[nodes_batch_ind, :]], dim=-1)
            global_feat_new.index_add_(dim=0, index=nodes_batch_ind, source=self.phi_global_feat(phi_input))

            # Add residual connection
            if self.global_feat_res:
                global_feat_new = global_feat_new + global_feat

        return global_feat_new


class EGCL(MessagePassingLayer):
    def __init__(self, node_feat_d: int = 1, node_pos_d: int = 3, edge_feat_d: int = 0, nodes_feat_d_in: int = None,
                 edge_feat_extra_d: int = 0, global_feat_d: int = 0, msg_d: int = None, dropout=config.dropout,
                 act_fn=config.act_fn, bias=config.model_bias,
                 norm_layer=config.norm_layer, norm_layer_bias=config.norm_layer_bias,
                 node_pos_res=config.node_pos_res, node_feat_res=config.node_feat_res,
                 edge_feat_res=config.edge_feat_res, global_feat_res=config.global_feat_res,
                 node_pos_agg=config.pos_msg_agg, node_feat_agg=config.feat_msg_agg,
                 phi_node_pos_weight=config.phi_x_weight, epsilon_x=config.epsilon_x,
                 d_inv_edge_feat=config.d_inv_edge_feat, node_feat_agg_in_edge_updt=config.node_feat_agg_in_edge_updt,
                 phi_weight_act=config.phi_weight_act):
        # Notation
        # x_i: position of node i
        # h_i: feature vector of node i
        # e_ij: feature vector of edge from node i to node j
        # g: global feature vector
        # m_ij: message passed from node i to node j
        # wm_ij: weight of message m_ij
        # wx_ij: weight of position update message from node j to node i

        if msg_d is None:
            msg_d = node_feat_d
        if nodes_feat_d_in is None:
            nodes_feat_d_in = node_feat_d

        # Define the activation function used in modules that determine the attention weights
        if phi_weight_act == 'sigmoid':
            weight_act = nn.Sigmoid()
        elif phi_weight_act == 'exp':
            weight_act = LambdaMod(Lambda=lambda x: torch.exp(x))
        elif phi_weight_act == 'exp_clamped':
            weight_act = LambdaMod(Lambda=lambda x: torch.exp(x.clamp(min=-10, max=10)))
        elif phi_weight_act == 'local_softmax':
            weight_act = LambdaMod(Lambda=lambda x: x, name=phi_weight_act)
        else:
            raise ValueError(f"weight activation {phi_weight_act!r} is not supported.")

        # Define module that computes the messages m_ij passed from node i to node j:
        # m_ij = phi_msg(h_i, h_j, d_ij, e_ij, g)
        phi_input_dim = 2 * nodes_feat_d_in + edge_feat_d + edge_feat_extra_d + global_feat_d
        if node_pos_d > 0:
            phi_input_dim += 1  # d_ij
        if d_inv_edge_feat:
            phi_input_dim += 1  # d_ij^-1
        phi_msg = nn.Sequential(nn.Linear(phi_input_dim, msg_d, bias=bias),
                                act_fn,
                                nn.Linear(msg_d, msg_d, bias=bias),
                                act_fn)

        # Add normalization layers
        if norm_layer == 'layernorm':
            phi_msg.insert(2, nn.LayerNorm(msg_d, bias=norm_layer_bias))
            phi_msg.insert(5, nn.LayerNorm(msg_d, bias=norm_layer_bias))

        # Temporary add a debug hook to print the intermediate dtype of the first nn.Linear
        if 0 and config.debug and config.amp:
            def print_output_dtype_hook(module, input, output):
                print(f"{module} Output Type: {output.dtype}")

            phi_msg[0].register_forward_hook(print_output_dtype_hook)

        # Define the module that computes the weight of the messages:
        # wm_ij = phi_wm(m_ij)
        phi_msg_weight = nn.Sequential(nn.Linear(msg_d, 1, bias=bias), weight_act)

        # Define module that computes the new node features:
        # h_new_i = phi_h(h_i, sum_j wm_ij/N_i m_ij)
        # N_i depends on config.feat_msg_agg
        if node_feat_d > 0:
            phi_node_feat = nn.Sequential(nn.Linear(nodes_feat_d_in + msg_d, node_feat_d, bias=bias),
                                          act_fn,
                                          nn.Linear(node_feat_d, node_feat_d, bias=bias))

            if norm_layer == 'layernorm':
                phi_node_feat.insert(2, nn.LayerNorm(node_feat_d, bias=norm_layer_bias))
        else:
            phi_node_feat = None

        # Define module that computes the new node position:
        # x_new_i = x_i + sum_j (x_i - x_j)/D_ij wx_ij phi_x(h_i, h_j, d_ij, e_ij, g)
        # D_ij depends on config.node_pos_agg
        if node_pos_d > 0:
            phi_node_pos = nn.Sequential(nn.Linear(phi_input_dim, msg_d, bias=bias),
                                         act_fn,
                                         nn.Linear(msg_d, msg_d, bias=bias),
                                         act_fn,
                                         nn.Linear(msg_d, 1, bias=bias))

            # Add normalization layers after the activation function
            if norm_layer == 'layernorm':
                phi_node_pos.insert(2, nn.LayerNorm(msg_d, bias=norm_layer_bias))
                phi_node_pos.insert(5, nn.LayerNorm(msg_d, bias=norm_layer_bias))

            # Define module that determines the weight each phi_x output:
            # wx_ij = phi_wx(h_i, h_j, d_ij, e_ij, g)
            if phi_node_pos_weight:
                phi_node_pos_weight = nn.Sequential(nn.Linear(phi_input_dim, 1, bias=bias), weight_act)
            else:
                phi_node_pos_weight = None
        else:
            phi_node_pos, phi_node_pos_weight = None, None

        # Define module that computes the new edge features:
        # e_ij_new = phi_e(HH_ij, d_ij, e_ij, g)
        #     H_ij = h_i + h_j   if node_feat_agg_in_edge_updt = 'sum'
        #          = |h_i - h_j| if node_feat_agg_in_edge_updt = 'abs_diff'
        #          = [h_i, h_j]  if node_feat_agg_in_edge_updt = None
        if edge_feat_d > 0:
            if node_feat_agg_in_edge_updt in ['sum', 'abs_diff']:
                phi_edge_input_d = phi_input_dim - node_feat_d
            elif node_feat_agg_in_edge_updt is None:
                phi_edge_input_d = phi_input_dim
            phi_edge_feat = nn.Sequential(nn.Linear(phi_edge_input_d, edge_feat_d, bias=bias),
                                          act_fn,
                                          nn.Linear(edge_feat_d, edge_feat_d, bias=bias))

            # Add normalization layers after the activation function
            if norm_layer == 'layernorm':
                phi_edge_feat.insert(2, nn.LayerNorm(edge_feat_d, bias=norm_layer_bias))
        else:
            phi_edge_feat = None

        # Define module that computes the new global features:
        # g_new = g + sum_i phi_g(h_i, g)
        if global_feat_d > 0:
            phi_global_feat = nn.Sequential(nn.Linear(nodes_feat_d_in + global_feat_d, global_feat_d, bias=bias),
                                            act_fn,
                                            nn.Linear(global_feat_d, global_feat_d, bias=bias))

            # Add normalization layers after the activation function
            if norm_layer == 'layernorm':
                phi_global_feat.insert(2, nn.LayerNorm(global_feat_d, bias=norm_layer_bias))
        else:
            phi_global_feat = None

        # Add dropout layers BEFORE all nn.Linear layers in all phi modules.
        if dropout > 0:
            phi_mod = [phi_msg, phi_node_pos, phi_node_feat, phi_edge_feat, phi_global_feat]
            phi_mod = [p for p in phi_mod if p is not None]
            for phi_temp in phi_mod:
                Lin_mod_ind = [i for i, n in enumerate(phi_temp) if isinstance(n, nn.Linear)]
                for ind in reversed(Lin_mod_ind):
                    phi_temp.insert(ind, nn.Dropout(dropout))

        super().__init__(msg_d=msg_d,
                         phi_msg=phi_msg, phi_msg_weight=phi_msg_weight, phi_node_feat=phi_node_feat,
                         phi_node_pos=phi_node_pos, phi_node_pos_weight=phi_node_pos_weight,
                         phi_edge_feat=phi_edge_feat, phi_global_feat=phi_global_feat,
                         node_pos_res=node_pos_res, node_feat_res=node_feat_res,
                         edge_feat_res=edge_feat_res, global_feat_res=global_feat_res,
                         node_pos_agg=node_pos_agg, node_feat_agg=node_feat_agg, epsilon_x=epsilon_x,
                         d_inv_edge_feat=d_inv_edge_feat, node_feat_agg_in_edge_updt=node_feat_agg_in_edge_updt)
        self.node_feat_d = node_feat_d
        self.node_pos_d = node_pos_d
        self.edge_feat_d = edge_feat_d
        self.global_feat_d = global_feat_d

    def gen_rand_inputs(self, B=5, N=20):
        nodes_adj = symmetrize(torch.rand((B, N, N))) < 0.5
        edges_ind = torch.nonzero(torch.triu(nodes_adj, diagonal=1))
        node_pos = torch.randn((B, N, self.node_pos_d)) if self.node_pos_d > 0 else None
        node_feat = torch.randn((B, N, self.node_feat_d))
        edge_feat = torch.randn((edges_ind.shape[0], self.edge_feat_d)) if self.edge_feat_d > 0 else None
        global_feat = torch.randn((B, self.global_feat_d)) if self.global_feat_d > 0 else None
        inputs = dict(node_pos=node_pos, node_feat=node_feat, edge_feat=edge_feat, global_feat=global_feat,
                      edges_ind=edges_ind, nodes_adj=nodes_adj)
        return inputs

    @classmethod
    def test(cls, n=10, seed=10):
        torch.manual_seed(seed)
        agg_choices = ['sum', 'sumN', 'mean', 'weightN']
        for i in range(n):
            kwargs = dict(msg_d=torch.randint(1, 6, (1,)),
                          node_feat_d=torch.randint(1, 6, (1,)), node_pos_d=torch.randint(0, 3, (1,)),
                          edge_feat_d=torch.randint(0, 3, (1,)), global_feat_d=torch.randint(0, 3, (1,)),
                          node_pos_res=torch.rand((1,)) > 0.5, node_feat_res=torch.rand((1,)) > 0.5,
                          edge_feat_res=torch.rand((1,)) > 0.5, phi_node_pos_weight=True,
                          node_pos_agg=np.random.choice(agg_choices), node_feat_agg=np.random.choice(agg_choices),
                          norm_layer='LayerNorm', act_fn=nn.SiLU())
            gcl1 = cls(**kwargs)

            # Test equivariance
            if gcl1.node_pos_d > 0:
                gcl1.is_equivariant()

            # Test forward method and chaining
            inputs_dict = gcl1.gen_rand_inputs()
            output1 = gcl1(**inputs_dict)
            input2_dict = copy.copy(inputs_dict)
            input2_dict.update({k: v for k, v in zip(['node_pos', 'node_feat', 'edge_feat', 'global_feat'], output1)})
            output2 = gcl1(**input2_dict)

            print(f'Test {i + 1} passed with kwargs=\n{kwargs}')
        return output1

    def is_equivariant(self, tol=1e-5):
        """
        Tests if the module is SE(3) equivariant
        Args:
            tol: numerical tolerance used to evaluate the differences between the model output and its rotated output.

        Returns:
            Bool indicating whether the module is equivariant or not
        """
        # Create some random input position and features
        B, N = 10, 20
        rand_inputs = self.gen_rand_inputs(B=B, N=N)
        remove_centroid(rand_inputs['node_pos'], inplace=True)
        node_pos_out, _, _, _ = self(**rand_inputs)

        # Generate random rigid-body motions (trans+rot)
        rand_RBM = RigidBodyMotion.random(size=(B, 1), device=self.device)

        # Rotate the input positions and revaluate the model's output
        rand_inputs['node_pos'] = rand_RBM.transform(rand_inputs['node_pos'])
        node_pos_out_trans, _, _, _ = self(**rand_inputs)

        # Compare the output of the non-rotated input with the non-rotated output of the rotated input
        node_pos_trans_after = rand_RBM.transform(node_pos_out)
        rotated_pos_diff = node_pos_trans_after - node_pos_out_trans
        max_scale = torch.cat([node_pos_out_trans, node_pos_trans_after], dim=0).max()
        diff_max = rotated_pos_diff.abs().max() / max_scale
        is_equiv = diff_max < tol
        if not is_equiv:
            warnings.warn(f'{self.__class__.__name__} is not SE(3) equivariant.\nPos diff max>tol:{diff_max}>{tol}')

        # Test for permutation equivariance of the nodes
        node_pos_out, node_feat_out, edge_feat_out, global_feat_out = self(**rand_inputs)

        perm = torch.randperm(N)
        perm_inv = torch.argsort(perm)
        if rand_inputs['node_pos'] is not None:
            rand_inputs['node_pos'] = rand_inputs['node_pos'][:, perm, :]
        rand_inputs['node_feat'] = rand_inputs['node_feat'][:, perm, :]
        edges_ind = rand_inputs['edges_ind']
        rand_inputs['edges_ind'] = torch.cat([edges_ind[:, :1], perm_inv[edges_ind[:, 1:]]], dim=-1)
        rand_inputs['nodes_adj'] = rand_inputs['nodes_adj'][:, perm[:, None], perm[None, :]]
        node_pos_out_p, node_feat_out_p, edge_feat_out_p, global_feat_out_p = self(**rand_inputs)
        # The permutation test above doesn't fully test for permutation invariance.
        # edges_ind needs to be reordered so that edges_ind[:,1] < edges_ind[:,2] to model a permutation of the adjacency matrix.

        is_pos_perm_equiv = torch.isclose(node_pos_out_p, node_pos_out[:, perm, :], rtol=tol)
        if not is_pos_perm_equiv.all():
            is_equiv = False
            pos_diff = node_pos_out_p / node_pos_out[:, perm, :] - 1
            warnings.warn(f'{self.__class__.__name__} node position output is not permutation equivariant.'
                          f'\nNode pos diff max>tol:{pos_diff.abs().max()}>{tol}')

        is_node_feat_perm_equiv = torch.isclose(node_feat_out_p, node_feat_out[:, perm, :], rtol=tol)
        if not is_node_feat_perm_equiv.all():
            is_equiv = False
            feat_diff = node_feat_out_p / node_feat_out[:, perm, :] - 1
            warnings.warn(f'{self.__class__.__name__} node features output is not permutation equivariant.'
                          f'\nNode feat diff max>tol:{feat_diff.abs().max()}>{tol}')

        is_edge_feat_perm_equiv = torch.isclose(edge_feat_out_p, edge_feat_out, rtol=tol)
        if not is_edge_feat_perm_equiv.all():
            is_equiv = False
            feat_diff = edge_feat_out_p / edge_feat_out - 1
            warnings.warn(f'{self.__class__.__name__} edge features output is not permutation equivariant.'
                          f'\nEdge feat diff max>tol:{feat_diff.abs().max()}>{tol}')

        return is_equiv


class Gain(nn.Module):
    def __init__(self, init_val=1.0, signed=False, max_mag: torch.Tensor | float = torch.inf, sigmoid=False):
        super().__init__()
        assert not (max_mag == torch.inf and sigmoid), f"sigmoid=True is only possible when max_mag<inf."

        self.signed = nn.parameter.Buffer(torch.as_tensor(signed), persistent=True)
        self.max_mag = nn.parameter.Buffer(torch.as_tensor(max_mag), persistent=True)
        self.sigmoid = nn.parameter.Buffer(torch.as_tensor(sigmoid), persistent=True)
        init_val = torch.as_tensor(init_val)

        if max_mag < torch.inf:
            if self.sigmoid:
                if signed:
                    # gain = max_mag*(2*torch.sigmoid(weight)-1)
                    gain_inv = lambda x: torch.special.logit((x / max_mag + 1) / 2)
                else:
                    # gain = max_mag * (torch.sigmoid(weight) + 1)/2
                    gain_inv = lambda x: torch.special.logit(x / max_mag)
            else:
                gain_inv = lambda x: x
            self.weight = nn.Parameter(gain_inv(init_val))
        else:
            if self.signed:
                self.weight = nn.Parameter(init_val)
            else:
                self.weight = nn.Parameter(init_val.log().clamp_min(-18))
                torch.nn.utils.parametrizations.parametrize.register_parametrization(self, 'weight', Positive())

        assert self.weight.isfinite(), f"initial weight is not finite: weight={self.weight.data.item()}"

    def calculate_gain(self):
        gain = self.weight
        if self.max_mag < torch.inf:
            if self.sigmoid:
                gain = torch.sigmoid(gain)
                if self.signed:
                    gain = (2 * gain - 1)
                gain = gain * self.max_mag
            else:
                if self.signed:
                    gain_min, gain_max = -self.max_mag, self.max_mag
                else:
                    gain_min, gain_max = 0, self.max_mag

                # Clamp the gain value if it crossed the bounds.
                if not (gain_min <= gain <= gain_max):
                    with torch.no_grad():
                        gain.clamp_(min=gain_min, max=gain_max)
        return gain

    def forward(self, x: torch.Tensor):
        gain = self.calculate_gain()
        return gain * x

    def extra_repr(self) -> str:
        return f"signed={self.signed.data.item()},max_mag={self.max_mag.data.item()},sigmoid={self.sigmoid.data.item()}"


class RotationLayer(nn.Module):
    def __init__(self, d: int, bias: bool = False, device=None, dtype=None):
        super().__init__()
        self.d = d
        self.weight = nn.Parameter(self.init_weight(device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(d, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def init_weight(self, device=None, dtype=None):
        X = torch.randn(self.d, self.d, device=device, dtype=dtype)
        Q, R = torch.linalg.qr(X)
        d_det = torch.linalg.det(Q)
        if d_det < 0:
            Q[:, 0] = -Q[:, 0]

        # Use Inverse Cayley Transform to find skew-symmetric matrix: A = (Q - I)(Q + I)^-1
        I = torch.eye(self.d)
        A = torch.linalg.solve(Q + I + 1e-6 * I, Q - I)
        A = 0.5 * (A - A.T)

        # Extract the lower triangular values
        lower_indices = torch.tril_indices(self.d, self.d, offset=-1)
        weight = A[lower_indices[0], lower_indices[1]]
        return weight

    def construct_rot_matrix(self):
        device, dtype = self.weight.device, self.weight.dtype
        A = torch.zeros(self.d, self.d, device=device, dtype=dtype)
        lower_indices = torch.tril_indices(self.d, self.d, offset=-1, device=device)
        A[lower_indices[0], lower_indices[1]] = self.weight
        A = A - A.T

        # Cayley Transform: R = (I - A)^-1 * (I + A)
        I = torch.eye(self.d, device=device, dtype=dtype)
        return torch.linalg.solve(I - A, I + A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R = self.construct_rot_matrix()
        out = torch.matmul(x, R)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"d={self.d},bias={self.bias is not None}"


if __name__ == '__main__':
    B, N = 10, 4
    # Test EGCL layer
    EGCL.test(n=100)

    # Test equivariance of EGCL layers
    layer = EGCL(node_feat_d=5, edge_feat_d=4, global_feat_d=0, node_pos_agg='sum', phi_weight_act='local_softmax')
    layer.is_equivariant()

    # # Test EquiCrossProdLayer
    # cross_prod_layer = EquiCrossProdLayer(hidden_d=10, node_feat_d=10, phi_weight=True)
    # node_pos = torch.randn(B, N, 3)
    # node_feat = torch.randn(B, N, cross_prod_layer.node_feat_d)
    # triplet_ind = torch.randint(0, N, (B, 40, 3))
    # node_pos_new = cross_prod_layer(node_pos, node_feat, triplet_ind)
