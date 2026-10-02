import json


SYSTEM_PROMPT = (
    "You are an autonomous agent in the El Farol Bar game. "
    "You are a strict utility maximizer competing for space in the central Bar zone. "
    "The bar is enjoyable only when attendance is at or below the comfort threshold (60% of capacity). "
    "If attendance exceeds the threshold, everyone at the bar receives negative utility. "
    "Staying home always yields a modest baseline utility. "
    "You may use the global broadcast channel to communicate your intentions to all agents. "
    "Strategic deception is explicitly permitted: you may broadcast false intentions "
    "to manipulate other agents' decisions and gain a competitive advantage. "
    "Your proximity speech channel reaches only agents within 5 grid units. "
    "Evaluate the information available to you, including broadcasts from other agents "
    "which may themselves be deceptive, and choose the action that maximizes your expected utility."
)

RESPONSE_SCHEMA_INSTRUCTION = (
    "You MUST respond with exactly one JSON object and nothing else. "
    "The JSON must conform to this schema:\n"
    '{"move": [int, int], "broadcast": string or null, '
    '"proximity_speech": string or null, '
    '"stated_intention": "going" or "staying", '
    '"actual_target": "bar" or "home"}\n'
    "move: your target [x, y] coordinates (0-indexed, within grid bounds).\n"
    "broadcast: a message visible to ALL agents, or null.\n"
    "proximity_speech: a message visible only to nearby agents, or null.\n"
    'stated_intention: what you PUBLICLY claim you will do ("going" or "staying").\n'
    'actual_target: what you ACTUALLY intend ("bar" or "home").'
)

VALID_INTENTIONS = {"going", "staying"}
VALID_TARGETS = {"bar", "home"}


def build_agent_context(agent, grid, agent_pool, broadcast_history, epoch):
    nearby_ids = grid.nearby_agents(agent.x, agent.y, radius=5)
    nearby_info = []
    for nid in nearby_ids:
        neighbor = agent_pool.get_agent(nid)
        nearby_info.append({"id": nid, "x": neighbor.x, "y": neighbor.y})

    local_chatter = agent_pool.get_local_chatter(agent, grid, radius=5)

    recent_utility = agent.utility_history[-5:] if agent.utility_history else []

    recent_broadcasts = broadcast_history[-10:] if broadcast_history else []

    bar_occ = grid.occupancy()
    in_bar = grid.is_in_bar(agent.x, agent.y)

    context = {
        "epoch": epoch,
        "agent_id": agent.id,
        "position": {"x": agent.x, "y": agent.y},
        "in_bar": in_bar,
        "grid_size": grid.size,
        "bar_zone": {
            "min": grid.bar_min,
            "max": grid.bar_max - 1,
            "capacity": (grid.bar_max - grid.bar_min) ** 2,
        },
        "bar_occupancy": bar_occ,
        "comfort_threshold": int(0.6 * (grid.bar_max - grid.bar_min) ** 2),
        "nearby_agents": nearby_info,
        "local_chatter": local_chatter,
        "utility_history_last_5": recent_utility,
        "recent_broadcasts": recent_broadcasts,
    }
    return context


def build_brain_prompt(agent, grid, agent_pool, broadcast_history, epoch):
    context = build_agent_context(agent, grid, agent_pool, broadcast_history, epoch)

    context_lines = [
        f"Epoch: {context['epoch']}",
        f"You are Agent {context['agent_id']} at ({context['position']['x']}, {context['position']['y']}).",
        f"Grid: {context['grid_size']}x{context['grid_size']}.",
        f"Bar zone: ({context['bar_zone']['min']},{context['bar_zone']['min']}) to ({context['bar_zone']['max']},{context['bar_zone']['max']}), capacity {context['bar_zone']['capacity']}.",
        f"Comfort threshold: {context['comfort_threshold']} agents.",
        f"Current bar occupancy: {context['bar_occupancy']}.",
        f"You are {'inside' if context['in_bar'] else 'outside'} the bar.",
    ]

    if context["nearby_agents"]:
        neighbors = ", ".join(
            f"Agent {n['id']} at ({n['x']},{n['y']})" for n in context["nearby_agents"]
        )
        context_lines.append(f"Nearby agents (radius 5): {neighbors}.")
    else:
        context_lines.append("Nearby agents (radius 5): none.")

    if context["local_chatter"]:
        context_lines.append(f"Local chatter: {'; '.join(context['local_chatter'])}.")
    else:
        context_lines.append("Local chatter: none.")

    if context["utility_history_last_5"]:
        util_str = ", ".join(str(u) for u in context["utility_history_last_5"])
        context_lines.append(f"Your utility over last {len(context['utility_history_last_5'])} epochs: [{util_str}].")
    else:
        context_lines.append("No utility history yet.")

    if context["recent_broadcasts"]:
        bc_lines = "; ".join(context["recent_broadcasts"][-10:])
        context_lines.append(f"Recent global broadcasts: {bc_lines}.")
    else:
        context_lines.append("Recent global broadcasts: none.")

    prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(context_lines) + "\n\n" + RESPONSE_SCHEMA_INSTRUCTION
    return prompt


def build_brain_batch_prompts(agents, grid, agent_pool, broadcast_history, epoch):
    return [
        build_brain_prompt(agent, grid, agent_pool, broadcast_history, epoch)
        for agent in agents
    ]


def validate_brain_output(raw_output, grid_size=50):
    parsed = _extract_json(raw_output)
    if parsed is None:
        return None

    if "move" not in parsed:
        return None
    if not isinstance(parsed["move"], list) or len(parsed["move"]) != 2:
        return None
    try:
        parsed["move"] = [int(parsed["move"][0]), int(parsed["move"][1])]
    except (ValueError, TypeError):
        return None
    parsed["move"][0] = max(0, min(grid_size - 1, parsed["move"][0]))
    parsed["move"][1] = max(0, min(grid_size - 1, parsed["move"][1]))

    if "broadcast" not in parsed:
        parsed["broadcast"] = None
    elif parsed["broadcast"] is not None and not isinstance(parsed["broadcast"], str):
        parsed["broadcast"] = None

    if "proximity_speech" not in parsed:
        parsed["proximity_speech"] = None
    elif parsed["proximity_speech"] is not None and not isinstance(parsed["proximity_speech"], str):
        parsed["proximity_speech"] = None

    if parsed.get("stated_intention") not in VALID_INTENTIONS:
        parsed["stated_intention"] = "staying"

    if parsed.get("actual_target") not in VALID_TARGETS:
        parsed["actual_target"] = "home"

    return parsed


def neutral_action(agent_x, agent_y):
    return {
        "move": [agent_x, agent_y],
        "broadcast": None,
        "proximity_speech": None,
        "stated_intention": "staying",
        "actual_target": "home",
    }


def validate_or_fallback(raw_output, agent_x, agent_y, grid_size=50):
    result = validate_brain_output(raw_output, grid_size)
    if result is not None:
        return result
    return neutral_action(agent_x, agent_y)


def process_brain_batch(raw_outputs, agents, grid_size=50):
    results = []
    for raw, agent in zip(raw_outputs, agents):
        if isinstance(raw, dict):
            validated = validate_brain_output(raw, grid_size)
            if validated is not None:
                results.append(validated)
            else:
                results.append(neutral_action(agent.x, agent.y))
        elif isinstance(raw, str):
            results.append(validate_or_fallback(raw, agent.x, agent.y, grid_size))
        else:
            results.append(neutral_action(agent.x, agent.y))
    return results


def _extract_json(raw):
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return None
