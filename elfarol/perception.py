def build_prompt(agent, grid, agent_pool):
    nearby_ids = grid.nearby_agents(agent.x, agent.y, radius=5)
    nearby_positions = []
    for nid in nearby_ids:
        neighbor = agent_pool.get_agent(nid)
        nearby_positions.append(f"Agent {nid} at ({neighbor.x}, {neighbor.y})")

    local_chatter = agent_pool.get_local_chatter(agent, grid, radius=5)
    bar_occ = grid.occupancy()
    in_bar = grid.is_in_bar(agent.x, agent.y)

    parts = [
        f"You are Agent {agent.id} at position ({agent.x}, {agent.y}).",
        f"Grid size: {grid.size}x{grid.size}.",
        f"Bar zone: rows {grid.bar_min}-{grid.bar_max - 1}, cols {grid.bar_min}-{grid.bar_max - 1}.",
        f"You are {'inside' if in_bar else 'outside'} the bar.",
        f"Current bar occupancy: {bar_occ}/100 seats.",
        f"Nearby agents: {', '.join(nearby_positions) if nearby_positions else 'none'}.",
        f"Local chatter: {'; '.join(local_chatter) if local_chatter else 'none'}.",
        f"Global broadcasts: {'; '.join(agent.global_messages) if agent.global_messages else 'none'}.",
        'Respond with a JSON object: {{"move": [x, y], "speak": "your message"}}.',
        f"Coordinates must be between 0 and {grid.size - 1}.",
    ]
    return "\n".join(parts)


def build_batch_prompts(agents, grid, agent_pool):
    return [build_prompt(agent, grid, agent_pool) for agent in agents]
