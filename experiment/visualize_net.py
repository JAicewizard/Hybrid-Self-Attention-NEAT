import pickle
import sys
import os
import matplotlib.pyplot as plt
import networkx as nx
import math # For infinity

# Assume RecurrentNetwork is in 'recurrent_network.py' or available from neat.nn.recurrent
# If RecurrentNetwork is in a separate file, uncomment the line below and adjust path:
# from recurrent_network import RecurrentNetwork
# Otherwise, ensure 'neat' is installed and it's importable from there.
from neat.nn.recurrent import RecurrentNetwork

# Placeholder for BASE_DIR if it's not defined in experiment.configs.config
# If your BASE_DIR is defined elsewhere, ensure it's imported or set correctly.
try:
    from experiment.configs.config import BASE_DIR
except ImportError:
    print("Warning: BASE_DIR not found in experiment.configs.config. Using current directory.")
    BASE_DIR = "." # Default to current directory if not found


def load_net(fitness):
    """
    Loads a RecurrentNetwork object from a pickled file.

    Args:
        fitness (int): The fitness value used in the filename.

    Returns:
        RecurrentNetwork: The loaded recurrent neural network.

    Raises:
        FileNotFoundError: If the specified network file does not exist.
    """
    path = os.path.join(BASE_DIR, f"net_output_{fitness}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No saved network found at {path}")

    print(f"Loading network from: {path}")
    with open(path, 'rb') as f:
        net = pickle.load(f)
    return net


def topological_layering(G: nx.DiGraph, input_nodes: set, output_nodes: set):
    """
    Assigns a pseudo-layer index to each node using a topological pass.
    Recurrent connections are ignored for layout purposes to achieve a feed-forward
    -like visualization. Nodes are assigned to the earliest possible layer.

    Args:
        G (nx.DiGraph): The networkx graph representing the neural network.
        input_nodes (set): Set of input node IDs.
        output_nodes (set): Set of output node IDs.

    Returns:
        dict: A dictionary mapping node ID to its assigned layer index.
    """
    # Initialize layers: inputs at 0, others at infinity (unreachable)
    layers = {n: 0 for n in input_nodes}
    for node in G.nodes():
        if node not in layers:
            layers[node] = math.inf

    # Create a copy of the graph, removing recurrent edges for layering
    # A simple way to identify recurrent edges: if target is already at or before source's layer
    # This loop will converge when no more nodes can be pushed to a higher layer.
    
    # Use a Kahn's algorithm-like approach (based on in-degrees) for proper layering
    # This is more robust for general DAGs
    
    # Calculate initial in-degrees for non-input nodes
    in_degree = G.in_degree()
    queue = [n for n in input_nodes if in_degree[n] == 0 or n in input_nodes] # Start with true inputs

    # Initialize layers
    for n in G.nodes():
        layers[n] = -1 # Unassigned

    for n in input_nodes:
        layers[n] = 0
        if n not in queue: # Ensure all input nodes are in the queue initially
            queue.append(n)

    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1

        for v in G.successors(u):
            # Only consider non-recurrent edges for layering
            # A simple heuristic: if the successor's layer is not yet set or can be improved
            if layers[v] < layers[u] + 1: # If v is not yet placed or current path gives a higher layer
                layers[v] = layers[u] + 1
                # To ensure proper topological sort, we'd generally need to check if all predecessors are processed
                # For visualization, this greedy update works well for layering.
                if v not in queue: # Avoid adding duplicates to the queue if already processed
                    queue.append(v)
            
    # Handle nodes that might not be reachable from inputs (e.g., disconnected hidden nodes)
    # Assign them to a default layer (e.g., after outputs or a specific hidden layer)
    max_layer = max(layers.values()) if layers else 0
    for node in G.nodes():
        if layers[node] == -1: # If still unassigned (not reachable from inputs)
            layers[node] = max_layer + 1 # Place them after existing layers, or another heuristic
            
    return layers


def build_layout(G: nx.DiGraph, layers: dict):
    """
    Create a layout dictionary where each node is positioned based on its layer.
    Nodes within the same layer are spread out evenly vertically over [0, 1].
    Layers remain positioned at integer x-coordinates.

    Args:
        G (nx.DiGraph): The networkx graph.
        layers (dict): Dictionary mapping node ID to its layer index.

    Returns:
        dict: A dictionary mapping node ID to its (x, y) coordinates.
    """
    layer_nodes = {}
    for node, layer in layers.items():
        if layer != math.inf:  # Only consider placed nodes
            layer_nodes.setdefault(layer, []).append(node)

    pos = {}
    for layer_idx, nodes in sorted(layer_nodes.items()):
        sorted_nodes = sorted(nodes)
        num_nodes_in_layer = len(sorted_nodes)

        if num_nodes_in_layer == 1:
            # Just place single node at center vertically
            ys = [0.5]
        else:
            # Evenly space nodes between 1.0 (top) and 0.0 (bottom)
            ys = [1 - i / (num_nodes_in_layer - 1) for i in range(num_nodes_in_layer)]

        for node, y in zip(sorted_nodes, ys):
            pos[node] = (layer_idx, y)  # X = layer index, Y = vertical position normalized [0,1]

    return pos

def visualize_net(net: RecurrentNetwork):
    """
    Visualizes the structure of a RecurrentNetwork using NetworkX and Matplotlib.

    Args:
        net (RecurrentNetwork): The recurrent neural network to visualize.
    """
    G = nx.DiGraph()

    # Get node types
    input_nodes = set(net.input_nodes)
    output_nodes = set(net.output_nodes)
    all_known_nodes = set(net.values[0].keys()) # All nodes mentioned in 'values' init
    
    # Add all nodes to the graph first
    for node_id in all_known_nodes:
        G.add_node(node_id)
    
    # Add edges based on net.node_evals
    # net.node_evals contains (node_id, activation, aggregation, bias, response, links)
    # where 'links' are (input_id, weight) tuples for incoming connections to node_id.
    for node_id, activation, aggregation, bias, response, links in net.node_evals:
        # Node_id is the target of these links
        for input_id, weight in links:
            G.add_edge(input_id, node_id, weight=weight)
            
    # Identify hidden nodes after all nodes and edges are potentially added
    hidden_nodes = all_known_nodes - input_nodes - output_nodes

    print(f"Network Details:\n  Inputs: {sorted(list(input_nodes))}\n  Outputs: {sorted(list(output_nodes))}\n  Hidden: {sorted(list(hidden_nodes))}\n  Total Nodes: {len(G.nodes())}\n  Edges: {len(G.edges())}")

    # Layering for layout
    layers = topological_layering(G, input_nodes, output_nodes)
    pos = build_layout(G, layers)

    # Fallback for any nodes not placed by build_layout (should be rare with topological_layering)
    for node in G.nodes():
        if node not in pos:
            # Assign a default position; e.g., to the right of all layers
            pos[node] = (max(layers.values()) + 1, 0)

    # Node colors
    node_colors = []
    for node in G.nodes():
        if node in input_nodes:
            node_colors.append("lightgreen")
        elif node in output_nodes:
            node_colors.append("salmon")
        elif node in hidden_nodes: # Ensure hidden nodes are distinct
            node_colors.append("skyblue")
        else: # For any other nodes that might appear (e.g., from values but not node_evals)
            node_colors.append("lightgray")


    # Draw graph
    plt.figure(figsize=(16, 10)) # Adjust figure size for better readability
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, edgecolors='black', linewidths=0.5)

    # Draw node labels
    node_labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='bold')

    # Draw edges
    # Differentiate positive/negative weights and enable curved edges for better visualization of recurrence/parallel paths
    for u, v, data in G.edges(data=True):
        weight = data['weight']
        edge_color = 'blue' if weight > 0 else 'red'
        # Scale line thickness by absolute weight
        line_width = 2.5 # Max thickness can be adjusted
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u,v)],
                                width=line_width, alpha=0.7, edge_color=edge_color,
                                arrows=False) # Use curved edges

    plt.title("Recurrent NEAT Network Structure", fontsize=16)
    plt.xlabel("Layer")
    plt.ylabel("Node Position within Layer")
    plt.grid(True, linestyle=':', alpha=0.6) # Add a subtle grid

    # Adjust plot limits to provide some padding
    x_values, y_values = zip(*pos.values())
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    plt.xlim(x_min - 0.5, x_max + 0.5)
    plt.ylim(y_min - 1, y_max + 1) # More padding for Y-axis

    plt.show()


if __name__ == '__main__':
    # Add an explanation about BASE_DIR if it's defaulting
    if BASE_DIR == ".":
        print("Note: `BASE_DIR` for loading networks is currently set to the current directory.")
        print("If your network files are in a different location (e.g., 'results/'),")
        print("please ensure 'experiment.configs.config.py' exists and defines `BASE_DIR` correctly,")
        print("or manually adjust the `BASE_DIR` variable in this script.")

    if len(sys.argv) < 2:
        print("Usage: python visualize_net.py <fitness_value>")
        print("Example: python visualize_net.py 123")
        sys.exit(1)

    try:
        fitness_to_load = int(sys.argv[1])
        net_to_visualize = load_net(fitness_to_load)
        visualize_net(net_to_visualize)
    except ValueError:
        print(f"Error: Fitness value must be an integer. Got '{sys.argv[1]}'")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)