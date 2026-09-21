import torch
import torch.nn as nn
import torch.nn.functional as F


class HRGCNConv(nn.Module):
    """
    Hierarchical Relation-Augmented Graph Convolutional Network Layer.
    Implements source-to-destination node type and edge type hierarchical transformations
    as formulated in Li et al. (IEEE DSAA 2023).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_node_types: int = 4,
        num_edge_types: int = 5,
        ablation: str = "full",  # 'full', 'no-edge-relation', 'no-node-relation', 'vanilla-hetgcn'
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.ablation = ablation

        # Determine number of relation-specific parameter matrices
        if ablation == "vanilla-hetgcn":
            self.num_eff_src = 1
            self.num_eff_dst = num_node_types
            self.num_eff_edges = 1
        elif ablation == "no-edge-relation":
            self.num_eff_src = num_node_types
            self.num_eff_dst = num_node_types
            self.num_eff_edges = 1
        elif ablation == "no-node-relation":
            self.num_eff_src = 1
            self.num_eff_dst = num_node_types
            self.num_eff_edges = num_edge_types
        else:  # full
            self.num_eff_src = num_node_types
            self.num_eff_dst = num_node_types
            self.num_eff_edges = num_edge_types

        # Dedicated weight transforms for each (src_type, dst_type, edge_type) tuple
        self.weights = nn.Parameter(
            torch.Tensor(self.num_eff_src, self.num_eff_dst, self.num_eff_edges, in_dim, out_dim)
        )
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        
        # Self-loop transformation per destination node type
        self.self_weights = nn.Parameter(torch.Tensor(num_node_types, in_dim, out_dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.self_weights)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor):
        """
        Args:
            x: Node feature tensor [num_nodes, in_dim]
            node_types: Node type indices [num_nodes]
            edge_index: Graph edge indices [2, num_edges] (src, dst)
            edge_types: Edge type indices [num_edges]
        """
        num_nodes = x.size(0)
        out = torch.zeros(num_nodes, self.out_dim, device=x.device)

        if edge_index.size(1) > 0:
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]

            # Compute node degrees for GCN normalization: 1 / sqrt(deg(src) * deg(dst))
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, dst_nodes, torch.ones_like(dst_nodes, dtype=torch.float))
            deg = torch.clamp(deg, min=1.0)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[src_nodes] * deg_inv_sqrt[dst_nodes]

            src_types = node_types[src_nodes]
            dst_types = node_types[dst_nodes]

            # Map to effective relation indices based on ablation mode
            eff_src = torch.zeros_like(src_types) if self.num_eff_src == 1 else src_types
            eff_dst = dst_types
            eff_edge = torch.zeros_like(edge_types) if self.num_eff_edges == 1 else edge_types

            # Vectorized message computation
            # For each unique relation tuple present in the batch, apply its transformation matrix
            unique_relations = torch.unique(torch.stack([eff_src, eff_dst, eff_edge], dim=1), dim=0)
            
            for rel in unique_relations:
                s_t, d_t, e_t = rel[0].item(), rel[1].item(), rel[2].item()
                mask = (eff_src == s_t) & (eff_dst == d_t) & (eff_edge == e_t)
                if not mask.any():
                    continue

                rel_src = src_nodes[mask]
                rel_dst = dst_nodes[mask]
                rel_norm = norm[mask].unsqueeze(1)

                # Linear transformation for this specific relation
                w = self.weights[s_t, d_t, e_t]  # [in_dim, out_dim]
                msg = torch.matmul(x[rel_src], w) * rel_norm

                out.index_add_(0, rel_dst, msg)

        # Apply self-loop transformations per destination node type
        for n_t in range(self.num_node_types):
            type_mask = (node_types == n_t)
            if type_mask.any():
                self_msg = torch.matmul(x[type_mask], self.self_weights[n_t])
                out[type_mask] = out[type_mask] + self_msg

        out = out + self.bias
        return F.leaky_relu(out, negative_slope=0.1)


class HRGCN(nn.Module):
    """
    HRGCN Graph-Level Anomaly Detection Model.
    Combines hierarchical relation message passing, global pooling, Deep SVDD, and Self-Supervised head.
    """
    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 32,
        out_dim: int = 16,
        num_node_types: int = 4,
        num_edge_types: int = 5,
        num_layers: int = 2,
        ablation: str = "full",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.ablation = ablation

        # GNN Convolution Stack
        self.convs = nn.ModuleList()
        self.convs.append(
            HRGCNConv(in_dim, hidden_dim, num_node_types, num_edge_types, ablation=ablation)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                HRGCNConv(hidden_dim, out_dim, num_node_types, num_edge_types, ablation=ablation)
            )

        # Self-Supervised Projection Head (Discriminator)
        self.ss_head = nn.Sequential(
            nn.Linear(out_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Deep SVDD Center Vector (fixed after initialization)
        self.register_buffer("svdd_center", torch.zeros(out_dim))
        self.center_initialized = False

    def forward_graph_embedding(self, x: torch.Tensor, node_types: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor):
        """Extracts node embeddings and performs global max-pooling to get graph embedding g."""
        h = x
        for conv in self.convs:
            h = conv(h, node_types, edge_index, edge_types)
        
        # Graph-level global pooling (Max-pooling over nodes as in HRGCN paper)
        g = torch.max(h, dim=0, keepdim=True)[0]  # [1, out_dim]
        return g, h

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor):
        g, h = self.forward_graph_embedding(x, node_types, edge_index, edge_types)
        ss_score = self.ss_head(g)  # Probability of anomaly [1, 1]
        return g, ss_score, h

    def init_svdd_center(self, embeddings: torch.Tensor, eps: float = 0.1):
        """Initializes the hypersphere center c as the mean of normal graph embeddings."""
        c = torch.mean(embeddings, dim=0)
        # Avoid center being too close to zero to prevent trivial solutions
        c[(torch.abs(c) < eps)] = eps
        self.svdd_center.copy_(c)
        self.center_initialized = True

    def compute_anomaly_score(self, x: torch.Tensor, node_types: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor, beta: float = 0.5):
        """
        Computes composite anomaly score:
        S(G) = ||g - c||^2 + beta * ss_score
        """
        self.eval()
        with torch.no_grad():
            g, ss_score, h = self.forward(x, node_types, edge_index, edge_types)
            dist = torch.sum((g - self.svdd_center) ** 2, dim=-1).item()
            ss_val = ss_score.item()
            total_score = dist + beta * ss_val
            return {
                "total_score": float(total_score),
                "svdd_dist": float(dist),
                "ss_score": float(ss_val),
                "graph_embedding": g.cpu().numpy().tolist()[0],
                "node_embeddings": h.cpu().numpy().tolist()
            }
