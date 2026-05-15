
import json
from collections import deque
from LLMconfig import TransformerModelConfig

class LayerGraph:
    """
    Represents the abstract computation graph of a single model layer.
    It handles parsing the graph structure, calculating MACs for each node,
    and providing a topological sort of the nodes. It is hardware-agnostic.
    """
    def __init__(self, arch_data: dict, model_config: TransformerModelConfig, system_config: dict):
        self.nodes = arch_data['layer_graph']['nodes']
        self.dimensions_map = arch_data['dimensions']
        
        self.resolved_dims = self._resolve_dims(model_config, system_config)
        
        self.dependencies = self._build_dependencies()
        self.sorted_nodes = self._topological_sort()
        self.node_macs = self._calculate_all_node_macs()

    def _build_dependencies(self) -> dict:
        deps = {node['name']: [] for node in self.nodes}
        node_map = {node['name']: node for node in self.nodes}
        for node in self.nodes:
            inputs_list = node.get('inputs', [])
            # Handle the old 'input' field for backward compatibility if needed
            if 'input' in node:
                inputs_list.append({'source': node['input'], 'type': 'blocking'}) # Default to blocking

            for dep_info in inputs_list:
                # dep_info is now an object like {'source': '...', 'type': '...'}
                if isinstance(dep_info, dict) and 'source' in dep_info:
                    source_node = dep_info['source']
                    if source_node in node_map:
                        deps[node['name']].append(dep_info)
                # Simple string handling for robustness, though format is now obj
                elif isinstance(dep_info, str) and dep_info in node_map:
                    deps[node['name']].append({'source': dep_info, 'type': 'blocking'})
        return deps

    def _topological_sort(self) -> list:
        in_degree = {node['name']: 0 for node in self.nodes}
        for node_name in self.dependencies:
            for parent in self.dependencies[node_name]:
                in_degree[node_name] += 1
        
        queue = deque([node['name'] for node in self.nodes if in_degree[node['name']] == 0])
        sorted_list = []
        
        while queue:
            u = queue.popleft()
            sorted_list.append(u)
            
            for v_node in self.nodes:
                v = v_node['name']
                if any(parent['source'] == u for parent in self.dependencies.get(v, [])):
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
        
        return sorted_list if len(sorted_list) == len(self.nodes) else []

    def _resolve_dims(self, model_config, system_config):
        resolved = {}
        for sym_name, path in self.dimensions_map.items():
            try:
                if path.startswith('model.'):
                    val = getattr(model_config, path.split('.', 1)[1])
                elif path.startswith('config.'):
                    val = system_config[path.split('.', 1)[1]]
                else:
                    val = int(path) # Assume it's a literal number if not a path
                resolved[sym_name] = val
            except (AttributeError, KeyError):
                resolved[sym_name] = 0 # Default to 0 if path is invalid
        return resolved

    def _evaluate_dim_expr(self, expr_str: str, dims: dict) -> int:
        """Evaluates a simple dimension expression string."""
        expr_str = str(expr_str).strip()
        if '*' in expr_str:
            parts = [p.strip() for p in expr_str.split('*')]
            result = 1
            for part in parts:
                if part in dims:
                    result *= dims[part]
                else:
                    try:
                        result *= int(part)
                    except ValueError:
                        print(f"Warning: Could not evaluate part '{part}' in expression '{expr_str}'")
                        return 0
            return result
        elif expr_str in dims:
            return dims[expr_str]
        else:
            try:
                return int(expr_str)
            except ValueError:
                print(f"Warning: Could not evaluate dimension '{expr_str}'")
                return 0

    def _calculate_all_node_macs(self) -> dict:
        node_macs = {}
        dims = self.resolved_dims
        for node in self.nodes:
            node_name = node['name']
            if 'mac_calc' in node:
                mac_expr = node['mac_calc']
                try:
                    vals = {}
                    for k, v_expr in mac_expr.items():
                        vals[k] = self._evaluate_dim_expr(v_expr, dims)
                    
                    if node['type'] == 'Linear':
                        result = vals.get('l', 1) * vals.get('d_in', 1) * vals.get('d_out', 1) * vals.get('b', 1)
                    elif node['type'] == 'BatchMatmul':
                        result = vals.get('m', 1) * vals.get('k', 1) * vals.get('n', 1) * vals.get('b', 1)
                    else:
                        result = 0
                    node_macs[node_name] = result

                except Exception as e:
                    print(f"Error calculating MACs for node {node_name}: {e}")
                    node_macs[node_name] = 0
            else:
                node_macs[node_name] = 0
        print(f"DEBUG: node_macs = {node_macs}")
        return node_macs

class ModelProfile:
    """
    Abstract base class for a model's computational profile.
    """
    def get_mac_breakdown(self, model: TransformerModelConfig, config: dict) -> dict:
        raise NotImplementedError

    def get_memory_footprint(self, model: TransformerModelConfig, config: dict) -> dict:
        raise NotImplementedError

class GenericProfile(ModelProfile):
    
    def __init__(self, arch_json_path: str):
        with open(arch_json_path, 'r') as f:
            self.arch_data = json.load(f)

    def get_mac_breakdown(self, model: TransformerModelConfig, config: dict) -> dict:
        # Create a LayerGraph instance to perform the calculation.
        # The MAC calculation logic now resides solely in LayerGraph.
        graph = LayerGraph(self.arch_data, model, config)
        
        attn_macs = 0
        ffn_macs = 0
        
        in_ffn_block = False
        # Iterate over the topologically sorted nodes from the graph object
        for node_name in graph.sorted_nodes:
            node = next((n for n in graph.nodes if n['name'] == node_name), None)
            if not node: continue

            # A simple way to distinguish attention vs FFN parts of the layer
            if node['name'] == 'attn_add':
                in_ffn_block = True
            
            # Get the pre-calculated MACs from the graph object
            node_mac = graph.node_macs.get(node_name, 0)

            if in_ffn_block:
                ffn_macs += node_mac
            else:
                attn_macs += node_mac

        return {
            'attention': attn_macs,
            'ffn': ffn_macs
        }
    
