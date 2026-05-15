class Analyzer:
    """Simple critical-path scheduler based on MAC/M."""

    def __init__(self, layer_graph, hardware_model):
        self.graph = layer_graph
        self.node_macs = layer_graph.node_macs
        self.fu_map = hardware_model.fu_map
        self.dependencies = layer_graph.dependencies

    def schedule(self, M_alloc):
        timeline = {}
        end_times = {}

        for node in self.graph.sorted_nodes:

            raw_parents = self.dependencies.get(node, [])

            # ----------------------------------------------------------------------
            # Robust dependency parsing:
            # Convert dict → name, ignore invalid entries, ignore missing parents.
            # ----------------------------------------------------------------------
            parent_names = []
            for p in raw_parents:
                if p is None:
                    continue
                if isinstance(p, dict):
                    name = p.get("name")
                    if name is not None:
                        parent_names.append(name)
                elif isinstance(p, str):
                    parent_names.append(p)
                else:
                    # unexpected type → ignore
                    continue

            # Keep only parents that already have end_times
            parent_names = [n for n in parent_names if n in end_times]

            # compute start cycle
            start_cycle = 0 if not parent_names else max(end_times[n] for n in parent_names)

            # MAC compute duration
            macs = self.node_macs[node]
            fu = self.fu_map[node]

            M = max(M_alloc.get(fu, 1e-9), 1e-9)
            duration = macs / M

            end_cycle = start_cycle + duration

            # record
            timeline[node] = (start_cycle, end_cycle)
            end_times[node] = end_cycle

        total_cycles = max(end_times.values()) if end_times else 0
        return timeline, total_cycles
