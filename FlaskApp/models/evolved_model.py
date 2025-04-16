# NEAT model class and loader
import neat
import torch
import torch.nn as nn
import pickle

# Load NEAT config
config_path = "neat_cfg_vis.txt"
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

#Load models KMeans++ and SVC

# Load the winner genome
with open("evolved_neat_model.pkl", "rb") as f:
    winner = pickle.load(f)

# Define the evolved PyTorch model
class EvolvedNN(nn.Module):
    def __init__(self, neat_net, genome):
        super().__init__()
        self.layers = nn.ModuleList()
        self.node_map = {}
        layer_index = 0
        neuron_index = 0

        for node_id in neat_net.input_nodes:
            self.node_map[node_id] = (layer_index, neuron_index)
            neuron_index += 1

        hidden_nodes = [n for n in genome.nodes if n not in neat_net.input_nodes and n not in neat_net.output_nodes]
        num_hidden = len(hidden_nodes)

        if num_hidden > 0:
            layer_index += 1
            neuron_index = 0
            self.layers.append(nn.Linear(len(neat_net.input_nodes), num_hidden))
            for node_id in hidden_nodes:
                self.node_map[node_id] = (layer_index, neuron_index)
                neuron_index += 1

        layer_index += 1
        neuron_index = 0
        input_size = num_hidden if num_hidden > 0 else len(neat_net.input_nodes)
        self.layers.append(nn.Linear(input_size, len(neat_net.output_nodes)))
        for node_id in neat_net.output_nodes:
            self.node_map[node_id] = (layer_index, neuron_index)
            neuron_index += 1

        self.load_neat_weights(genome)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 and i > 0:
                x = torch.relu(x)
        return x

    def load_neat_weights(self, genome):
        for conn in genome.connections.values():
            if conn.enabled:
                in_node, out_node = conn.key
                weight = conn.weight
                in_layer, in_neuron = self.node_map.get(in_node, (None, None))
                out_layer, out_neuron = self.node_map.get(out_node, (None, None))
                if in_layer is not None and out_layer is not None and out_layer < len(self.layers):
                    try:
                        self.layers[out_layer].weight.data[out_neuron, in_neuron] = weight
                        self.layers[out_layer].bias.data[out_neuron] = genome.nodes[out_node].bias
                    except IndexError:
                        pass
