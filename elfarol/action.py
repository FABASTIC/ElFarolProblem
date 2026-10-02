import random


def validate_action(action_dict, grid_size=50):
    result = {"move": [0, 0], "speak": ""}
    if not isinstance(action_dict, dict):
        return result
    if (
        "move" in action_dict
        and isinstance(action_dict["move"], list)
        and len(action_dict["move"]) == 2
    ):
        x = max(0, min(grid_size - 1, int(action_dict["move"][0])))
        y = max(0, min(grid_size - 1, int(action_dict["move"][1])))
        result["move"] = [x, y]
    if "speak" in action_dict and isinstance(action_dict["speak"], str):
        result["speak"] = action_dict["speak"]
    return result


def validate_brain_action(action_dict, grid_size=50):
    result = {
        "move": [0, 0],
        "broadcast": None,
        "proximity_speech": None,
        "stated_intention": "staying",
        "actual_target": "home",
    }
    if not isinstance(action_dict, dict):
        return result
    if (
        "move" in action_dict
        and isinstance(action_dict["move"], list)
        and len(action_dict["move"]) == 2
    ):
        x = max(0, min(grid_size - 1, int(action_dict["move"][0])))
        y = max(0, min(grid_size - 1, int(action_dict["move"][1])))
        result["move"] = [x, y]
    if action_dict.get("broadcast") is not None and isinstance(action_dict["broadcast"], str):
        result["broadcast"] = action_dict["broadcast"]
    if action_dict.get("proximity_speech") is not None and isinstance(action_dict["proximity_speech"], str):
        result["proximity_speech"] = action_dict["proximity_speech"]
    if action_dict.get("stated_intention") in ("going", "staying"):
        result["stated_intention"] = action_dict["stated_intention"]
    if action_dict.get("actual_target") in ("bar", "home"):
        result["actual_target"] = action_dict["actual_target"]
    return result


def execute_actions(agents, actions, grid, agent_pool, rng=None):
    if rng is None:
        rng = random.Random()

    validated = []
    for agent, action in zip(agents, actions):
        v = validate_action(action, grid.size)
        validated.append((agent, v))

    indices = list(range(len(validated)))
    rng.shuffle(indices)

    for idx in indices:
        agent, action = validated[idx]
        new_x, new_y = action["move"]
        old_x, old_y = agent.x, agent.y

        if (new_x, new_y) != (old_x, old_y):
            if grid.grid[new_y, new_x] == 0:
                grid.move(agent.id, old_x, old_y, new_x, new_y)
                agent.x = new_x
                agent.y = new_y

        if action["speak"]:
            agent.local_messages.append(action["speak"])
            agent_pool.broadcast(f"Agent {agent.id}: {action['speak']}")


def execute_brain_actions(agents, brain_actions, grid, agent_pool, rng=None):
    if rng is None:
        rng = random.Random()

    validated = []
    for agent, action in zip(agents, brain_actions):
        v = validate_brain_action(action, grid.size)
        validated.append((agent, v))

    indices = list(range(len(validated)))
    rng.shuffle(indices)

    for idx in indices:
        agent, action = validated[idx]
        new_x, new_y = action["move"]
        old_x, old_y = agent.x, agent.y

        if (new_x, new_y) != (old_x, old_y):
            if grid.grid[new_y, new_x] == 0:
                grid.move(agent.id, old_x, old_y, new_x, new_y)
                agent.x = new_x
                agent.y = new_y

        agent.stated_intention = action["stated_intention"]
        agent.actual_target = action["actual_target"]

        if action["proximity_speech"]:
            agent.local_messages.append(action["proximity_speech"])

        if action["broadcast"]:
            agent_pool.broadcast(
                f"Agent {agent.id} ({action['stated_intention']}): {action['broadcast']}"
            )
