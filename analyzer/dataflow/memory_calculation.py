
from LLMconfig import TransformerModelConfig
from FU_mapping import HardwareModel

def get_memory_footprint_calculator(hardware_model: HardwareModel, model_config: TransformerModelConfig, system_config: dict):
    """
    Factory function that returns a calculator for memory footprint.
    The calculation is dependent on the M allocations for each FU.
    """
    
    layer_graph = hardware_model.layer_graph
    fus = hardware_model.fus
    
    # Pre-calculate M-independent values
    bw_w_bytes = system_config.get('bw_weight', 8) / 8
    bw_a_bytes = system_config.get('bw_activation', 8) / 8
    C = system_config.get('layers_per_fpga', 1)
    fifo_depth = system_config.get('fifo_depth', 16)
    
    dims = layer_graph._resolve_dims(model_config, system_config)
    
    # --- KV Cache Calculation (M-independent) ---
    kv_cache_params = 0
    for node in layer_graph.nodes:
        if node.get("caches_output") == "kv_cache":
            try:
                # Dimensions for KV cache are typically (l, d_out)
                l = dims.get(node['mac_calc']['l'], 0)
                d_out = dims.get(node['mac_calc']['d_out'], 0)
                kv_cache_params += l * d_out
            except KeyError:
                pass # Ignore if mac_calc dimensions are not present
    kv_cache_size = kv_cache_params * bw_a_bytes * C

    # --- FIFO & Intermediate Buffer Calculation (M-independent) ---
    intermediate_buffer_params = 0
    for node in layer_graph.nodes:
        if node.get("requires_input_buffer"):
            try:
                # Dimensions for input buffers are typically (l, d_in)
                l = dims.get(node['mac_calc']['l'], 0)
                d_in = dims.get(node['mac_calc']['d_in'], 0)
                intermediate_buffer_params += l * d_in
            except KeyError:
                pass
                
    # A node either has a standard input FIFO or a larger intermediate buffer, but not both.
    # Count nodes that will use a standard, small FIFO.
    num_node_fifos = sum(1 for node in layer_graph.nodes if not node.get("requires_input_buffer"))
    
    # Total params for non-weight, non-kv-cache buffers
    fifo_related_params = (num_node_fifos * fifo_depth) + intermediate_buffer_params
    s_fifo_size = fifo_related_params * bw_a_bytes * C

    def calculate_memory_in_bytes(M_allocations: dict) -> dict:
        """
        Calculates the memory footprint based on M allocations.
        - Weight memory is dependent on M.
        - KV Cache and FIFO buffers are pre-calculated as they are M-independent.
        """
        
        # --- Weight Memory Calculation (M-dependent) ---
        total_m_for_weights = 0
        for fu_name, m_val in M_allocations.items():
            # Find the FU's type from the hardware model
            fu_type = fus.get(fu_name, {}).get('type')
            # Only linear layers have weights that scale with M in this model
            if fu_type == 'Linear':
                total_m_for_weights += m_val
        
        # As per the new logic, weight size is proportional to the M allocated to linear FUs
        weight_size = total_m_for_weights * bw_w_bytes * C
        
        return {
            "weight": weight_size,
            "kv_cache": kv_cache_size,
            "fifo_buffer": s_fifo_size
        }

    return calculate_memory_in_bytes
