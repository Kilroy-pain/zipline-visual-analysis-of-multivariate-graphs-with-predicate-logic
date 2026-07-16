import torch
import numpy as np
from torch_geometric.data import Data

class ZipLine:
    def __init__(self, graph_data):
        """
        Initialize the ZipLine system with a multivariate graph.
        :param graph_data: A PyTorch Geometric Data object containing the graph structure and node attributes.
        """
        self.graph_data = graph_data

    def evaluate_predicate(self, predicate):
        """
        Evaluate a predicate on the graph data.
        :param predicate: A function that takes node attributes and adjacency information and returns a boolean mask.
        :return: A mask indicating which nodes satisfy the predicate.
        """
        node_attrs = self.graph_data.x
        edge_index = self.graph_data.edge_index
        return predicate(node_attrs, edge_index)

    def subgraph_selection(self, predicate):
        """
        Select a subgraph based on a predicate.
        :param predicate: A function that takes node attributes and adjacency information and returns a boolean mask.
        :return: A subgraph containing only the nodes that satisfy the predicate.
        """
        mask = self.evaluate_predicate(predicate)
        selected_nodes = torch.where(mask)[0]
        subgraph = self._extract_subgraph(selected_nodes)
        return subgraph

    def _extract_subgraph(self, selected_nodes):
        """
        Extract a subgraph given a set of selected nodes.
        :param selected_nodes: A tensor containing the indices of the selected nodes.
        :return: A PyTorch Geometric Data object representing the subgraph.
        """
        node_mask = torch.zeros(self.graph_data.num_nodes, dtype=torch.bool)
        node_mask[selected_nodes] = True

        edge_mask = node_mask[self.graph_data.edge_index[0]] & node_mask[self.graph_data.edge_index[1]]
        new_edge_index = self.graph_data.edge_index[:, edge_mask]

        # Remap node indices for the subgraph
        node_mapping = torch.full((self.graph_data.num_nodes,), -1, dtype=torch.long)
        node_mapping[selected_nodes] = torch.arange(len(selected_nodes))
        new_edge_index = node_mapping[new_edge_index]

        new_x = self.graph_data.x[selected_nodes]
        return Data(x=new_x, edge_index=new_edge_index)

    def learn_predicate(self, interaction_data):
        """
        Learn a predicate based on user interactions.
        :param interaction_data: A dictionary containing interaction data (e.g., selected nodes, brushed attributes).
        :return: A learned predicate function.
        """
        selected_nodes = interaction_data.get('selected_nodes', None)
        brushed_attributes = interaction_data.get('brushed_attributes', None)

        def learned_predicate(node_attrs, edge_index):
            mask = torch.zeros(node_attrs.size(0), dtype=torch.bool)
            if selected_nodes is not None:
                mask[selected_nodes] = True
            if brushed_attributes is not None:
                for attr_idx, value_range in brushed_attributes.items():
                    attr_mask = (node_attrs[:, attr_idx] >= value_range[0]) & (node_attrs[:, attr_idx] <= value_range[1])
                    mask = mask | attr_mask
            return mask

        return learned_predicate

if __name__ == '__main__':
    # Create dummy graph data
    num_nodes = 6
    num_features = 3
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2],
                               [1, 0, 3, 2, 5, 4, 2, 0]], dtype=torch.long)
    node_features = torch.tensor([[1.0, 0.5, 0.3],
                                   [0.8, 0.2, 0.1],
                                   [0.6, 0.4, 0.9],
                                   [0.3, 0.7, 0.5],
                                   [0.9, 0.1, 0.2],
                                   [0.4, 0.8, 0.6]], dtype=torch.float)

    graph_data = Data(x=node_features, edge_index=edge_index)

    # Initialize ZipLine
    zipline = ZipLine(graph_data)

    # Define a predicate: Select nodes where the first attribute is greater than 0.5
    def predicate(node_attrs, edge_index):
        return node_attrs[:, 0] > 0.5

    # Apply the predicate to select a subgraph
    subgraph = zipline.subgraph_selection(predicate)
    print("Subgraph nodes:", subgraph.x)
    print("Subgraph edges:", subgraph.edge_index)

    # Simulate user interaction data
    interaction_data = {
        'selected_nodes': torch.tensor([0, 2]),
        'brushed_attributes': {0: (0.4, 0.9)}
    }

    # Learn a predicate based on interaction data
    learned_pred = zipline.learn_predicate(interaction_data)

    # Apply the learned predicate to select a subgraph
    learned_subgraph = zipline.subgraph_selection(learned_pred)
    print("Learned Subgraph nodes:", learned_subgraph.x)
    print("Learned Subgraph edges:", learned_subgraph.edge_index)