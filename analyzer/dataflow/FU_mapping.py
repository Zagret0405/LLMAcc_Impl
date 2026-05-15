from LLMmodel import LayerGraph

class HardwareModel:
    """
    Builds a hardware model from a LayerGraph object.
    It determines the Functional Units (FUs) based on the deployment schedule
    and provides a function to calculate latency based on compute power allocation.
    Its focus is purely on hardware mapping, not on abstract model properties.
    """
    def __init__(self, layer_graph: LayerGraph, reuse_factor: int = 1, reuse_factors: dict = None):
        self.layer_graph = layer_graph
        self.reuse_factor = reuse_factor
        self.reuse_factors = reuse_factors if reuse_factors else {}
        
        self.fus = {}  # Stores FU info {fu_name: {'type': fu_type, 'reuse_factor': r}}
        self.fu_map = {}  # Maps node name to FU name {node_name: fu_name}
        
        # Group nodes by the temporal FU they belong to
        self.temporal_fu_nodes = {} 

        for node in self.layer_graph.nodes:
            node_name = node['name']
            deployment_type = node.get('deployment', 'temporal')
            
            if deployment_type == 'spatial':
                fu_name = node_name
                # Use specific reuse factor if available, else global default
                r = self.reuse_factors.get(fu_name, self.reuse_factor)
                self.fus[fu_name] = {'type': node['type'], 'reuse_factor': r}
                self.fu_map[node_name] = fu_name
            else:  # temporal
                fu_name = node['type']
                if fu_name not in self.fus:
                    # Use specific reuse factor if available, else global default
                    r = self.reuse_factors.get(fu_name, self.reuse_factor)
                    self.fus[fu_name] = {'type': fu_name, 'reuse_factor': r}
                    self.temporal_fu_nodes[fu_name] = []
                self.fu_map[node_name] = fu_name
                self.temporal_fu_nodes[fu_name].append(node_name)
