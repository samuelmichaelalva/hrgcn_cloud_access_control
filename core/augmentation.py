import random
import torch


class HetGDA:
    """
    Heterogeneous Graph Data Augmentation (HetGDA) for Self-Supervised Normality Learning.
    Generates pseudo-anomalies from normal graphs using:
    1. Heterogeneous Edge Perturbation
    2. Heterogeneous Edge Replacement
    3. Heterogeneous Node & Edge Type Swapping
    """
    def __init__(
        self,
        num_node_types: int = 4,
        num_edge_types: int = 5,
        perturbation_rate: float = 0.25,
        swap_rate: float = 0.2,
    ):
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.perturbation_rate = perturbation_rate
        self.swap_rate = swap_rate

    def edge_perturbation(self, edge_index: torch.Tensor, edge_types: torch.Tensor, num_nodes: int):
        """Randomly drops existing edges and introduces cross-entity non-standard links."""
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return edge_index, edge_types

        # 1. Edge removal (dropout)
        keep_mask = torch.rand(num_edges) > self.perturbation_rate
        if not keep_mask.any():
            keep_mask[0] = True  # Keep at least 1 edge

        kept_edges = edge_index[:, keep_mask]
        kept_types = edge_types[keep_mask]

        # 2. Add perturbed random edges
        num_add = max(1, int(num_edges * self.perturbation_rate))
        new_src = torch.randint(0, num_nodes, (num_add,))
        new_dst = torch.randint(0, num_nodes, (num_add,))
        new_types = torch.randint(0, self.num_edge_types, (num_add,))

        new_edges = torch.stack([new_src, new_dst], dim=0)
        aug_edge_index = torch.cat([kept_edges, new_edges], dim=1)
        aug_edge_types = torch.cat([kept_types, new_types], dim=0)

        return aug_edge_index, aug_edge_types

    def edge_replacement(self, edge_index: torch.Tensor, edge_types: torch.Tensor, num_nodes: int):
        """Replaces standard edges with high-risk abnormal relations."""
        num_edges = edge_index.size(1)
        if num_edges < 2:
            return edge_index, edge_types

        num_replace = max(1, int(num_edges * self.perturbation_rate))
        replace_indices = torch.randperm(num_edges)[:num_replace]

        aug_edge_index = edge_index.clone()
        aug_edge_types = edge_types.clone()

        for idx in replace_indices:
            # Replace target with a random node and assign a random edge type
            aug_edge_index[1, idx] = torch.randint(0, num_nodes, (1,)).item()
            aug_edge_types[idx] = torch.randint(0, self.num_edge_types, (1,)).item()

        return aug_edge_index, aug_edge_types

    def node_edge_type_swapping(self, node_types: torch.Tensor, edge_types: torch.Tensor):
        """Swaps node roles or edge types to simulate privilege escalation / hijacked identity."""
        aug_node_types = node_types.clone()
        aug_edge_types = edge_types.clone()

        # Swap a fraction of node types
        num_nodes = node_types.size(0)
        if num_nodes > 1 and random.random() < 0.8:
            swap_count = max(1, int(num_nodes * self.swap_rate))
            swap_nodes = torch.randperm(num_nodes)[:swap_count]
            for n in swap_nodes:
                aug_node_types[n] = (aug_node_types[n] + random.randint(1, self.num_node_types - 1)) % self.num_node_types

        # Swap edge types
        num_edges = edge_types.size(0)
        if num_edges > 0 and random.random() < 0.6:
            swap_e_count = max(1, int(num_edges * self.swap_rate))
            swap_edges = torch.randperm(num_edges)[:swap_e_count]
            for e in swap_edges:
                aug_edge_types[e] = (aug_edge_types[e] + random.randint(1, self.num_edge_types - 1)) % self.num_edge_types

        return aug_node_types, aug_edge_types

    def augment(self, x: torch.Tensor, node_types: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor):
        """
        Applies a randomized combination of HetGDA augmentation methods to produce an anomalous graph.
        """
        method = random.choice(["perturb", "replace", "swap", "all"])
        num_nodes = x.size(0)

        aug_x = x.clone()
        aug_node_types = node_types.clone()
        aug_edge_index = edge_index.clone()
        aug_edge_types = edge_types.clone()

        if method == "perturb":
            aug_edge_index, aug_edge_types = self.edge_perturbation(aug_edge_index, aug_edge_types, num_nodes)
        elif method == "replace":
            aug_edge_index, aug_edge_types = self.edge_replacement(aug_edge_index, aug_edge_types, num_nodes)
        elif method == "swap":
            aug_node_types, aug_edge_types = self.node_edge_type_swapping(aug_node_types, aug_edge_types)
        else:  # all
            aug_edge_index, aug_edge_types = self.edge_perturbation(aug_edge_index, aug_edge_types, num_nodes)
            aug_node_types, aug_edge_types = self.node_edge_type_swapping(aug_node_types, aug_edge_types)

        return aug_x, aug_node_types, aug_edge_index, aug_edge_types
