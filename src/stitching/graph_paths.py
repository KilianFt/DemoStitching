import networkx as nx


def shortest_path_nodes(
    gg,
    initial_state,
    target_state,
    start_node_candidates=None,
    goal_node_candidates=None,
    allow_candidate_relaxation=True,
):
    """Compute shortest gaussian-node path without mutating GaussianGraph internals.

    Returns gaussian node ids only (no temporary initial/target nodes), or None if no path exists.
    """
    init_id = "__TEMP_INITIAL__"
    target_id = "__TEMP_TARGET__"

    gg.graph.add_node(init_id, pos=initial_state)
    initial_edges = []
    for node_id, node_data in list(gg.graph.nodes(data=True)):
        if node_id in (init_id, target_id):
            continue
        if "mean" not in node_data:
            continue
        weight = gg.compute_edge_weight(initial_state, None, node_data["mean"])
        if weight is not None:
            initial_edges.append((node_id, float(weight)))

    gg.graph.add_node(target_id, pos=target_state)
    target_edges = []
    for node_id, node_data in list(gg.graph.nodes(data=True)):
        if node_id in (init_id, target_id):
            continue
        if "mean" not in node_data or "direction" not in node_data:
            continue
        weight = gg.compute_edge_weight(node_data["mean"], node_data["direction"], target_state)
        if weight is not None:
            target_edges.append((node_id, float(weight)))

    initial_edges_sorted = sorted(initial_edges, key=lambda t: t[1])
    target_edges_sorted = sorted(target_edges, key=lambda t: t[1])

    def _base_count(candidate_value, total):
        if total <= 0:
            return 0
        if candidate_value is None:
            return total
        try:
            k = int(candidate_value)
        except Exception:
            return total
        if k <= 0:
            return total
        return min(k, total)

    def _expand_counts(base, total):
        if total <= 0:
            return [0]
        base = max(1, min(int(base), total))
        counts = [base]
        while counts[-1] < total:
            nxt = min(total, counts[-1] * 2)
            if nxt == counts[-1]:
                break
            counts.append(nxt)
        if counts[-1] != total:
            counts.append(total)
        return counts

    base_start = _base_count(start_node_candidates, len(initial_edges_sorted))
    base_goal = _base_count(goal_node_candidates, len(target_edges_sorted))
    start_counts = _expand_counts(base_start, len(initial_edges_sorted))
    goal_counts = _expand_counts(base_goal, len(target_edges_sorted))

    attempts = []
    attempts.append((base_start, base_goal))  # strict local-local
    if allow_candidate_relaxation:
        attempts.extend((s, base_goal) for s in start_counts[1:])  # keep goal local
        attempts.extend((base_start, g) for g in goal_counts[1:])  # keep start local
        attempts.extend((s, g) for s in start_counts[1:] for g in goal_counts[1:])  # relax both

    # deduplicate while preserving order
    seen = set()
    unique_attempts = []
    for pair in attempts:
        if pair in seen:
            continue
        seen.add(pair)
        unique_attempts.append(pair)

    path = None
    try:
        for start_count, goal_count in unique_attempts:
            gg.graph.remove_edges_from(list(gg.graph.out_edges(init_id)))
            gg.graph.remove_edges_from(list(gg.graph.in_edges(target_id)))

            for node_id, weight in initial_edges_sorted[:start_count]:
                gg.graph.add_edge(init_id, node_id, weight=weight)
            for node_id, weight in target_edges_sorted[:goal_count]:
                gg.graph.add_edge(node_id, target_id, weight=weight)

            try:
                path = nx.shortest_path(gg.graph, source=init_id, target=target_id, weight="weight")
                if path is not None:
                    break
            except Exception:
                path = None
    finally:
        if init_id in gg.graph:
            gg.graph.remove_node(init_id)
        if target_id in gg.graph:
            gg.graph.remove_node(target_id)

    if path is None:
        return None
    return path[1:-1]
