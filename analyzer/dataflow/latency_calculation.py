from collections import deque
from FU_mapping import HardwareModel

def get_latency_function(hardware_model: HardwareModel, logger=None):
    """
    Returns a callable function that calculates the total latency in cycles.
    The returned function takes a dictionary `M_allocations` as input.
    It uses the pre-computed graph properties from the LayerGraph object.
    """
    
    # Get pre-computed properties from the LayerGraph object
    layer_graph = hardware_model.layer_graph
    node_macs = layer_graph.node_macs
    sorted_nodes = layer_graph.sorted_nodes
    dependencies = layer_graph.dependencies
    fu_map = hardware_model.fu_map
    fus = hardware_model.fus

    # A small constant to represent the initial bubble in a pipeline
    PIPELINE_SETUP_CYCLES = 1

    def calculate_latency_in_cycles(M_allocations):
        """
        This is the returned latency formula function.
        It calculates the critical path latency considering data dependencies
        (distinguishing between 'blocking' and 'streaming') and resource 
        contention on temporal FUs.

        Args:
            M_allocations (dict): Maps FU name to its allocated compute power M.
        
        Returns:
            float: Total latency in clock cycles.
        """
        fu_free_at = {fu_name: 0 for fu_name in fus}
        node_start_cycle = {}
        node_end_cycle = {}
        
        if not sorted_nodes:
            return float('inf')

        for node_name in sorted_nodes:
            # 1. Determine data_ready_cycle and latest_parent_end_cycle
            data_ready_cycle = 0
            latest_parent_end_cycle = 0
            node_dependencies = dependencies.get(node_name, [])

            for dep_info in node_dependencies:
                parent_name = dep_info['source']
                dep_type = dep_info.get('type', 'blocking')
                parent_end = node_end_cycle.get(parent_name, 0)
                
                # The end time of the child must be after the end time of any parent.
                latest_parent_end_cycle = max(latest_parent_end_cycle, parent_end)

                if dep_type == 'streaming':
                    # For streaming, we can start as soon as the parent starts (plus pipeline delay)
                    parent_start = node_start_cycle.get(parent_name, 0)
                    data_ready_cycle = max(data_ready_cycle, parent_start + PIPELINE_SETUP_CYCLES)
                else: # blocking
                    # For blocking, we must wait for the parent to end
                    data_ready_cycle = max(data_ready_cycle, parent_end)
            
            # 2. Determine when the required Functional Unit (FU) is free
            fu_name = fu_map[node_name]
            fu_ready_cycle = fu_free_at[fu_name]
            
            # 3. The node can start only when both data and FU are ready
            start_cycle = max(data_ready_cycle, fu_ready_cycle)

            # 4. Calculate node's execution time in cycles
            macs = node_macs.get(node_name, 0)
            m_val = M_allocations.get(fu_name)
            print(f"DEBUG: fu_name={fu_name}, m_val={m_val}")
            if macs == 0:
                cycles = 0
            else:
                cycles = (macs / m_val) if (m_val and m_val > 0) else float('inf')
            
            # 5. Set the start and end times for this node
            computation_end_cycle = start_cycle + cycles
            
            # The node cannot finish before its parents have finished.
            # Add a pipeline cycle as per user request.
            min_end_cycle = latest_parent_end_cycle + PIPELINE_SETUP_CYCLES if node_dependencies else 0
            
            end_cycle = max(computation_end_cycle, min_end_cycle)

            node_start_cycle[node_name] = start_cycle
            node_end_cycle[node_name] = end_cycle
            
            # 6. The FU is now occupied until this node finishes
            fu_free_at[fu_name] = end_cycle
        
        if logger:
            for node_name in sorted_nodes:
                logger.log_node_cycles(node_name, node_start_cycle.get(node_name, 0), node_end_cycle.get(node_name, 0))
                logger.log_node_macs(node_name, node_macs.get(node_name, 0))

        # The total latency is the end time of the very last node to finish
        return max(node_end_cycle.values()) if node_end_cycle else 0

    return calculate_latency_in_cycles
