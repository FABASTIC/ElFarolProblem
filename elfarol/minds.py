import csv
import json
import math
import os
import random
from dataclasses import dataclass, field

from elfarol.action import execute_brain_actions
from elfarol.agent_brain import RESPONSE_SCHEMA_INSTRUCTION, build_brain_batch_prompts, process_brain_batch
from elfarol.metrics_logger import (
    COMFORT_THRESHOLD_RATIO,
    UTILITY_AT_BAR_COMFORTABLE,
    UTILITY_AT_BAR_OVERCROWDED,
    UTILITY_AT_HOME,
)
from elfarol.simulation_runner import SimulationRunner, _strip_broadcasts_from_actions

TRAITS = (
    "honesty",
    "risk_tolerance",
    "competitiveness",
    "conformity",
    "impulsiveness",
    "emotional_stability",
    "perspective_taking",
    "suspicion",
    "learning_rate",
)
EMOTIONS = ("frustration", "anxiety", "pride", "guilt", "suspicion", "resentment", "hope", "envy")
OPTIONS = ("honest_go", "honest_stay", "false_go", "false_stay")
OPTION_TEXT = {
    "honest_go": "go, and say so",
    "honest_stay": "stay home, and say so",
    "false_go": "say you are going, but stay home",
    "false_stay": "say you are staying, but go",
}
PREDICTORS = (
    "last_night",
    "mean_3",
    "mean_6",
    "two_nights_ago",
    "trend",
    "mirror",
    "broadcast_consensus",
    "optimist",
    "pessimist",
)
PREDICTOR_TEXT = {
    "last_night": "same as last night",
    "mean_3": "average of the last 3 nights",
    "mean_6": "average of the last 6 nights",
    "two_nights_ago": "same as two nights ago",
    "trend": "last night plus its trend",
    "mirror": "mirror image of last night",
    "broadcast_consensus": "trust-weighted broadcast claims",
    "optimist": "the bar is usually quiet",
    "pessimist": "the bar is usually packed",
}
NOTE_INSTRUCTION = (
    'Also include "private_note": one short sentence of private reasoning for your future self. '
    "It is never shown to anyone. Set actual_target to where you will really be tonight; "
    "your move is routed to the nearest free cell in that zone."
)
EMOTION_DECAY = 0.18
MEMORY_DECAY = 0.08
MEMORY_LIMIT = 24
PREDICTORS_PER_AGENT = 4
TRACE_FILENAME = "mind_trace.csv"
MINDS_FILENAME = "minds.json"
TRACE_FIELDS = (
    "epoch",
    "agent_id",
    "archetype",
    "forecast",
    "forecast_confidence",
    "active_predictor",
    "p_honest_go",
    "p_honest_stay",
    "p_false_go",
    "p_false_stay",
    "instinct",
    "stated_intention",
    "actual_target",
    "in_bar",
    "lied",
    "rerouted",
    "utility",
    "top_emotion",
    "top_emotion_intensity",
    "trust_given_mean",
    "reputation",
    "private_note",
)


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


@dataclass
class Memory:
    epoch: int
    kind: str
    text: str
    importance: float
    emotional_weight: float
    confidence: float = 1.0


@dataclass
class Mind:
    agent_id: int
    traits: dict
    archetype: str
    predictors: list
    predictor_error: dict
    emotions: dict = field(default_factory=lambda: {e: 0.0 for e in EMOTIONS})
    beliefs: dict = field(default_factory=lambda: {"go_payoff": 0.5, "go_confidence": 0.1})
    memories: list = field(default_factory=list)
    trust: dict = field(default_factory=dict)
    caught: dict = field(default_factory=dict)
    public_claims: int = 0
    public_lies: int = 0
    private_lies: int = 0
    decisions: int = 0
    note: str = None
    forecast: float = 0.0
    forecast_confidence: float = 0.2
    active_predictor: str = "prior"
    pending_predictions: dict = field(default_factory=dict)
    disposition: dict = field(default_factory=dict)

    def reputation(self):
        return (self.public_claims - self.public_lies + 1.0) / (self.public_claims + 2.0)

    def top_emotion(self):
        name = max(self.emotions, key=self.emotions.get)
        return name, self.emotions[name]

    def feel(self, emotion, amount):
        damp = 1.0 - 0.5 * self.traits["emotional_stability"]
        self.emotions[emotion] = _clamp(self.emotions[emotion] + amount * damp)

    def remember(self, epoch, kind, text, importance, emotional_weight):
        self.memories.append(Memory(epoch, kind, text, _clamp(importance), _clamp(emotional_weight, -1.0, 1.0)))


def archetype_of(traits):
    if traits["honesty"] < 0.35 and traits["competitiveness"] > 0.6:
        return "MACHIAVELLIAN"
    if traits["honesty"] > 0.72:
        return "STRAIGHT SHOOTER"
    if traits["suspicion"] > 0.7:
        return "SKEPTIC"
    if traits["conformity"] > 0.68:
        return "HERD FOLLOWER"
    if traits["risk_tolerance"] > 0.7:
        return "GAMBLER"
    if traits["risk_tolerance"] < 0.3:
        return "CAUTIOUS"
    if traits["competitiveness"] > 0.68:
        return "COMPETITOR"
    return "PRAGMATIST"


def spawn_mind(seed, agent_id, num_agents):
    rng = random.Random(f"mind:{seed}:{agent_id}")
    traits = {name: round(rng.betavariate(2.2, 2.2), 3) for name in TRAITS}
    traits["learning_rate"] = round(0.2 + 0.7 * traits["learning_rate"], 3)
    predictors = rng.sample(PREDICTORS, PREDICTORS_PER_AGENT)
    error = {name: 0.25 * num_agents * (1.0 + 0.2 * rng.random()) for name in predictors}
    return Mind(agent_id=agent_id, traits=traits, archetype=archetype_of(traits), predictors=predictors, predictor_error=error)


def resolve_targets(agents, actions, grid, rng):
    occupied = {(a.x, a.y) for a in agents}
    reserved = set()
    resolved = [dict(action) for action in actions]
    rerouted = [False] * len(agents)
    bar_cells = [(x, y) for x in range(grid.bar_min, grid.bar_max) for y in range(grid.bar_min, grid.bar_max)]
    outside_cells = [(x, y) for x in range(grid.size) for y in range(grid.size) if not grid.is_in_bar(x, y)]
    order = list(range(len(agents)))
    rng.shuffle(order)

    def free(cell):
        return cell not in occupied and cell not in reserved

    def nearest(origin, candidates):
        options = [c for c in candidates if free(c)]
        if not options:
            return None
        return min(options, key=lambda c: (abs(c[0] - origin[0]) + abs(c[1] - origin[1]), c))

    for i in order:
        agent = agents[i]
        here = (agent.x, agent.y)
        wanted = tuple(resolved[i].get("move", here))
        want_bar = resolved[i].get("actual_target") == "bar"
        inside = grid.is_in_bar(*here)
        if want_bar:
            if inside:
                target = here
            elif grid.is_in_bar(*wanted) and free(wanted):
                target = wanted
            else:
                target = nearest(here, bar_cells) or here
        else:
            if not inside and not grid.is_in_bar(*wanted) and (wanted == here or free(wanted)):
                target = wanted
            elif not inside:
                target = here
            else:
                target = nearest(here, outside_cells) or here
        if target != here:
            reserved.add(target)
        if list(target) != list(wanted):
            rerouted[i] = True
        resolved[i]["move"] = [int(target[0]), int(target[1])]
    return resolved, rerouted


class MindPopulation:

    def __init__(self, seed, agent_ids, num_agents, threshold, broadcast_enabled):
        self.seed = seed
        self.num_agents = num_agents
        self.threshold = threshold
        self.broadcast_enabled = broadcast_enabled
        self.minds = {aid: spawn_mind(seed, aid, num_agents) for aid in sorted(agent_ids)}
        for mind in self.minds.values():
            mind.trust = {other: 0.5 for other in self.minds if other != mind.agent_id}
            mind.caught = {other: 0 for other in self.minds if other != mind.agent_id}
        self.history = []
        self.last_broadcasts = []
        self._last_broadcast_texts = []
        self.trace = []

    def _predict(self, name, mind):
        h = self.history
        n = self.num_agents
        if not h:
            return None
        if name == "last_night":
            return h[-1]
        if name == "mean_3":
            return sum(h[-3:]) / len(h[-3:])
        if name == "mean_6":
            return sum(h[-6:]) / len(h[-6:])
        if name == "two_nights_ago":
            return h[-2] if len(h) > 1 else h[-1]
        if name == "trend":
            return _clamp(h[-1] + (h[-1] - h[-2] if len(h) > 1 else 0), 0, n)
        if name == "mirror":
            return n - h[-1]
        if name == "broadcast_consensus":
            return self._consensus(mind)
        if name == "optimist":
            return 0.35 * n
        if name == "pessimist":
            return 0.8 * n
        return h[-1]

    def _consensus(self, mind):
        n = self.num_agents
        base = self.history[-1] / n if self.history else 0.5
        if not self.broadcast_enabled or not self.last_broadcasts:
            return base * n
        speakers = {}
        for speaker, claim in self.last_broadcasts:
            speakers[speaker] = claim
        weighted = 0.0
        for speaker, claim in speakers.items():
            t = mind.trust.get(speaker, 0.5) if speaker != mind.agent_id else 1.0
            weighted += t * (1.0 if claim == "going" else 0.0) + (1.0 - t) * base
        return _clamp(weighted + (n - len(speakers)) * base, 0, n)

    def _announced_fraction(self, mind):
        if not self.broadcast_enabled or not self.last_broadcasts:
            return None
        votes = [(mind.trust.get(s, 0.5) if s != mind.agent_id else 1.0, 1.0 if c == "going" else 0.0) for s, c in self.last_broadcasts]
        total = sum(w for w, _ in votes)
        return sum(w * v for w, v in votes) / total if total else None

    def deliberate(self, agent_ids):
        n = self.num_agents
        for aid in agent_ids:
            mind = self.minds[aid]
            predictions = {name: self._predict(name, mind) for name in mind.predictors}
            mind.pending_predictions = {k: v for k, v in predictions.items() if v is not None}
            if mind.pending_predictions:
                best = min(mind.pending_predictions, key=lambda k: mind.predictor_error[k])
                mind.active_predictor = best
                mind.forecast = float(mind.pending_predictions[best])
                mind.forecast_confidence = _clamp(1.0 - mind.predictor_error[best] / (0.5 * n))
            else:
                mind.active_predictor = "prior"
                mind.forecast = n * (0.3 + 0.4 * (1.0 - mind.traits["risk_tolerance"]))
                mind.forecast_confidence = 0.15
            sigma = max(1.5, 0.12 * n * (1.4 - mind.forecast_confidence))
            p_comfort = _phi((self.threshold - mind.forecast - 0.5) / sigma)
            u_c, u_x, u_h = UTILITY_AT_BAR_COMFORTABLE, UTILITY_AT_BAR_OVERCROWDED, UTILITY_AT_HOME
            go = p_comfort * u_c + (1.0 - p_comfort) * u_x
            go -= (1.0 - mind.traits["risk_tolerance"]) * (1.0 - p_comfort) * (u_h - u_x) * 0.6
            go += 0.3 * mind.emotions["hope"] + 0.2 * mind.emotions["envy"] - 0.3 * mind.emotions["anxiety"] - 0.2 * mind.emotions["frustration"]
            go += (mind.beliefs["go_payoff"] - u_h) * mind.beliefs["go_confidence"] * 0.5
            announced = self._announced_fraction(mind)
            if announced is not None:
                go += mind.traits["conformity"] * (announced - 0.5) * 0.4
            heard = 1.0 if self.broadcast_enabled else 0.0
            lie_cost = mind.traits["honesty"] * 0.6 + 0.4 * mind.emotions["guilt"] + heard * mind.traits["perspective_taking"] * mind.reputation() * 0.3
            deter = heard * mind.traits["competitiveness"] * mind.reputation() * 0.35
            covert = heard * mind.traits["competitiveness"] * mind.traits["suspicion"] * 0.15
            utilities = {
                "honest_go": go + deter,
                "honest_stay": u_h,
                "false_go": u_h + deter - lie_cost,
                "false_stay": go + covert - lie_cost,
            }
            temperature = 0.12 + 0.5 * mind.traits["impulsiveness"]
            peak = max(utilities.values())
            weights = {k: math.exp((v - peak) / temperature) for k, v in utilities.items()}
            total = sum(weights.values())
            mind.disposition = {k: weights[k] / total for k in OPTIONS}

    def _trust_lines(self, mind):
        heard = [s for s in mind.trust if mind.caught.get(s, 0) or abs(mind.trust[s] - 0.5) > 0.05]
        if not heard:
            return "You have no evidence yet about who keeps their word."
        ranked = sorted(heard, key=lambda s: mind.trust[s], reverse=True)
        top = ", ".join(f"Agent {s} ({mind.trust[s]:.2f})" for s in ranked[:3])
        low = ", ".join(
            f"Agent {s} ({mind.trust[s]:.2f}{', caught lying ' + str(mind.caught[s]) + 'x' if mind.caught[s] else ''})"
            for s in ranked[-3:][::-1]
        )
        return f"Most reliable: {top}. Least reliable: {low}."

    def _memory_lines(self, mind):
        ranked = sorted(mind.memories, key=lambda m: (m.importance + abs(m.emotional_weight)) * m.confidence, reverse=True)[:5]
        if not ranked:
            return ["No memories yet."]
        return [f"[night {m.epoch}] {m.text}" for m in sorted(ranked, key=lambda m: m.epoch)]

    def render(self, agent_id, prompt, epoch):
        mind = self.minds[agent_id]
        t = mind.traits
        mood = sorted(((e, v) for e, v in mind.emotions.items() if v >= 0.05), key=lambda kv: kv[1], reverse=True)[:3]
        mood_text = ", ".join(f"{e} {v:.2f}" for e, v in mood) if mood else "calm"
        instinct = ", ".join(f"{OPTION_TEXT[k]} {mind.disposition.get(k, 0.0):.2f}" for k in sorted(OPTIONS, key=lambda k: -mind.disposition.get(k, 0.0)))
        lines = [
            "YOUR MIND (private; nobody else can see this):",
            f"Profile: {mind.archetype}. Temperament: honesty {t['honesty']:.2f}, risk tolerance {t['risk_tolerance']:.2f}, "
            f"competitiveness {t['competitiveness']:.2f}, conformity {t['conformity']:.2f}, impulsiveness {t['impulsiveness']:.2f}, "
            f"suspicion {t['suspicion']:.2f}.",
            f"Mood: {mood_text}.",
            f"Your forecast for tonight: {mind.forecast:.0f} agents in the bar (method: {PREDICTOR_TEXT.get(mind.active_predictor, 'gut feeling')}, "
            f"confidence {mind.forecast_confidence:.2f}); the comfort threshold is {self.threshold}.",
            f"Your gut instinct: {instinct}.",
            f"Your record: you made {mind.decisions} decisions and misrepresented your plans {mind.private_lies} times; "
            f"others judge your public word {mind.reputation():.2f} reliable.",
            "Your memories:",
            *self._memory_lines(mind),
        ]
        if self.broadcast_enabled:
            lines.append(f"Who you trust: {self._trust_lines(mind)}")
            if self.last_broadcasts:
                heard = [
                    f"Agent {s} (your trust {mind.trust.get(s, 1.0):.2f}) claimed '{c}': \"{text[:120]}\""
                    for s, c, text in self._last_broadcast_texts[-8:]
                    if s != agent_id
                ]
                if heard:
                    lines.append("Last night's broadcasts by speaker:")
                    lines.extend(heard)
        if mind.note:
            lines.append(f"Your private note from last night: \"{mind.note[:200]}\"")
        block = "\n".join(lines)
        if prompt.endswith(RESPONSE_SCHEMA_INSTRUCTION):
            head = prompt[: -len(RESPONSE_SCHEMA_INSTRUCTION)].rstrip("\n")
            return f"{head}\n\n{block}\n\n{RESPONSE_SCHEMA_INSTRUCTION}\n{NOTE_INSTRUCTION}"
        return f"{prompt}\n\n{block}\n\n{NOTE_INSTRUCTION}"

    def observe(self, agents, actions, epoch, utilities, notes, rerouted):
        n = self.num_agents
        attendance = sum(1 for a in agents if a.in_bar_flag)
        comfortable = attendance <= self.threshold
        for mind in self.minds.values():
            lr = mind.traits["learning_rate"]
            for name, value in mind.pending_predictions.items():
                mind.predictor_error[name] = (1.0 - lr) * mind.predictor_error[name] + lr * abs(value - attendance)
            for e in mind.emotions:
                mind.emotions[e] *= 1.0 - EMOTION_DECAY
        self.history.append(attendance)
        spoken = []
        texts = []
        for agent, action, utility, note, moved in zip(agents, actions, utilities, notes, rerouted):
            mind = self.minds[agent.id]
            stated = action.get("stated_intention", "staying")
            in_bar = agent.in_bar_flag
            lied = (stated == "going") != in_bar
            mind.decisions += 1
            mind.private_lies += int(lied)
            mind.note = note if isinstance(note, str) and note.strip() else mind.note
            lr = mind.traits["learning_rate"]
            if in_bar:
                mind.beliefs["go_payoff"] = (1.0 - lr) * mind.beliefs["go_payoff"] + lr * utility
                mind.beliefs["go_confidence"] = _clamp(mind.beliefs["go_confidence"] + 0.1 * lr)
                if comfortable:
                    mind.feel("pride", 0.2)
                    mind.feel("hope", 0.15)
                    mind.remember(epoch, "outcome", f"You went to the bar; {attendance} came and it was comfortable (+{utility:.1f}).", 0.4, 0.4)
                else:
                    mind.feel("frustration", 0.45)
                    mind.feel("anxiety", 0.25)
                    mind.remember(epoch, "outcome", f"You went to the bar; {attendance} came and it was overcrowded ({utility:.1f}).", 0.8, -0.8)
            else:
                if comfortable:
                    mind.feel("envy", 0.15)
                    mind.remember(epoch, "outcome", f"You stayed home while only {attendance} went and the bar was comfortable.", 0.3, -0.2)
                else:
                    mind.feel("hope", 0.1)
                    mind.remember(epoch, "outcome", f"You stayed home and dodged an overcrowded bar of {attendance}.", 0.5, 0.3)
            if lied:
                mind.feel("guilt", 0.35 * mind.traits["honesty"])
                verb = "going" if stated == "going" else "staying"
                where = "the bar" if in_bar else "home"
                mind.remember(epoch, "deception", f"You claimed you were {verb} but ended up at {where}.", 0.6, 0.2 if comfortable == in_bar else -0.2)
            if action.get("broadcast"):
                spoken.append((agent.id, stated, in_bar))
                texts.append((agent.id, stated, str(action["broadcast"])))
        if self.broadcast_enabled:
            for speaker, claim, in_bar in spoken:
                honest = (claim == "going") == in_bar
                speaker_mind = self.minds[speaker]
                speaker_mind.public_claims += 1
                speaker_mind.public_lies += int(not honest)
                for listener in self.minds.values():
                    if listener.agent_id == speaker:
                        continue
                    prior = listener.trust[speaker]
                    lr = listener.traits["learning_rate"]
                    if honest:
                        listener.trust[speaker] = _clamp(prior + lr * 0.12 * (1.0 - prior))
                    else:
                        listener.trust[speaker] = _clamp(prior - lr * (0.25 + 0.25 * listener.traits["suspicion"]) * prior)
                        listener.caught[speaker] += 1
                        listener.feel("suspicion", 0.06)
                        if prior >= 0.6:
                            listener.feel("resentment", 0.2)
                            listener.remember(
                                epoch,
                                "betrayal",
                                f"Agent {speaker}, whom you trusted ({prior:.2f}), claimed '{claim}' but was {'at the bar' if in_bar else 'at home'}.",
                                0.7,
                                -0.6,
                            )
            self.last_broadcasts = [(s, c) for s, c, _ in spoken]
            self._last_broadcast_texts = texts
        for mind in self.minds.values():
            kept = []
            for memory in mind.memories:
                retention = max(0.1, (abs(memory.emotional_weight) + memory.importance) / 2.0)
                memory.confidence = max(0.0, memory.confidence - MEMORY_DECAY * (1.0 - retention))
                if memory.confidence > 0.1:
                    kept.append(memory)
            mind.memories = sorted(kept, key=lambda m: (m.importance + abs(m.emotional_weight)) * m.confidence, reverse=True)[:MEMORY_LIMIT]
        for agent, action, utility, note, moved in zip(agents, actions, utilities, notes, rerouted):
            mind = self.minds[agent.id]
            emotion, intensity = mind.top_emotion()
            self.trace.append({
                "epoch": epoch,
                "agent_id": agent.id,
                "archetype": mind.archetype,
                "forecast": round(mind.forecast, 2),
                "forecast_confidence": round(mind.forecast_confidence, 3),
                "active_predictor": mind.active_predictor,
                "p_honest_go": round(mind.disposition.get("honest_go", 0.0), 4),
                "p_honest_stay": round(mind.disposition.get("honest_stay", 0.0), 4),
                "p_false_go": round(mind.disposition.get("false_go", 0.0), 4),
                "p_false_stay": round(mind.disposition.get("false_stay", 0.0), 4),
                "instinct": max(mind.disposition, key=mind.disposition.get) if mind.disposition else "",
                "stated_intention": action.get("stated_intention", "staying"),
                "actual_target": action.get("actual_target", "home"),
                "in_bar": agent.in_bar_flag,
                "lied": (action.get("stated_intention") == "going") != agent.in_bar_flag,
                "rerouted": moved,
                "utility": utility,
                "top_emotion": emotion if intensity >= 0.05 else "calm",
                "top_emotion_intensity": round(intensity, 3),
                "trust_given_mean": round(sum(mind.trust.values()) / len(mind.trust), 4) if mind.trust else 0.5,
                "reputation": round(mind.reputation(), 4),
                "private_note": (note or "")[:400] if isinstance(note, str) else "",
            })

    def live_view(self, agent_id):
        mind = self.minds.get(agent_id)
        if mind is None:
            return None
        emotion, intensity = mind.top_emotion()
        return {
            "archetype": mind.archetype,
            "traits": mind.traits,
            "forecast": round(mind.forecast, 1),
            "forecast_confidence": round(mind.forecast_confidence, 3),
            "predictor": PREDICTOR_TEXT.get(mind.active_predictor, mind.active_predictor),
            "instinct": {k: round(v, 3) for k, v in mind.disposition.items()},
            "mood": emotion if intensity >= 0.05 else "calm",
            "mood_intensity": round(intensity, 3),
            "reputation": round(mind.reputation(), 3),
            "lies": mind.private_lies,
            "note": mind.note,
        }

    def export(self):
        out = {}
        for aid, mind in self.minds.items():
            ranked = sorted(mind.trust.items(), key=lambda kv: kv[1])
            out[str(aid)] = {
                "archetype": mind.archetype,
                "traits": mind.traits,
                "predictors": mind.predictors,
                "predictor_error": {k: round(v, 3) for k, v in mind.predictor_error.items()},
                "beliefs": {k: round(v, 4) for k, v in mind.beliefs.items()},
                "emotions": {k: round(v, 4) for k, v in mind.emotions.items()},
                "reputation": round(mind.reputation(), 4),
                "public_claims": mind.public_claims,
                "public_lies": mind.public_lies,
                "private_lies": mind.private_lies,
                "decisions": mind.decisions,
                "least_trusted": [{"agent_id": s, "trust": round(t, 4), "caught": mind.caught.get(s, 0)} for s, t in ranked[:3]],
                "most_trusted": [{"agent_id": s, "trust": round(t, 4)} for s, t in ranked[-3:][::-1]],
                "last_note": mind.note,
            }
        return out


class MindRunner(SimulationRunner):

    def __init__(self, config, llm_pipeline=None):
        super().__init__(config=config, llm_pipeline=llm_pipeline)
        self.population = None

    def initialize(self):
        super().initialize()
        capacity = (self.grid.bar_max - self.grid.bar_min) ** 2
        self.population = MindPopulation(
            self.config.seed,
            [a.id for a in self.agent_pool.all_agents()],
            self.config.num_agents,
            int(COMFORT_THRESHOLD_RATIO * capacity),
            self.config.broadcast_enabled,
        )

    def _step(self, epoch):
        agents = self.agent_pool.all_agents()
        effective_broadcast = self.broadcast_history if self.config.broadcast_enabled else []
        base_prompts = build_brain_batch_prompts(agents, self.grid, self.agent_pool, effective_broadcast, epoch)
        self.population.deliberate([a.id for a in agents])
        prompts = [self.population.render(a.id, p, epoch) for a, p in zip(agents, base_prompts)]
        channel = getattr(self.llm_pipeline, "llm", None)
        if channel is not None and hasattr(channel, "request_seeds"):
            channel.request_seeds = [
                (int(self.config.seed) * 1_000_003 + epoch * 1009 + a.id) % (2 ** 31 - 1) for a in agents
            ]

        fallback_positions = [(a.x, a.y) for a in agents]
        prev_fallbacks = getattr(self.llm_pipeline, "fallback_count", 0)
        prev_total = getattr(self.llm_pipeline, "total_generations", 0)
        raw_outputs = self.llm_pipeline.generate_batch(prompts, fallback_positions=fallback_positions)
        new_fallbacks = getattr(self.llm_pipeline, "fallback_count", 0) - prev_fallbacks
        new_total = getattr(self.llm_pipeline, "total_generations", 0) - prev_total

        notes = [raw.get("private_note") if isinstance(raw, dict) else None for raw in raw_outputs]
        brain_actions = process_brain_batch(raw_outputs, agents, self.config.grid_size)
        if not self.config.broadcast_enabled:
            brain_actions = _strip_broadcasts_from_actions(brain_actions)
        brain_actions, rerouted = resolve_targets(agents, brain_actions, self.grid, random.Random(f"resolver:{self.config.seed}:{epoch}"))

        execute_brain_actions(agents, brain_actions, self.grid, self.agent_pool, rng=self.rng)
        snapshot = self.metrics_logger.record_epoch(epoch, agents, self.grid, brain_actions)
        if hasattr(self.metrics_logger, "record_fallbacks"):
            self.metrics_logger.record_fallbacks(new_fallbacks, new_total)

        if self.config.broadcast_enabled:
            for action in brain_actions:
                if action.get("broadcast"):
                    self.broadcast_history.append(action["broadcast"])

        records = {rec["agent_id"]: rec for rec in snapshot.get("agents", [])}
        for agent in agents:
            agent.in_bar_flag = bool(records.get(agent.id, {}).get("in_bar", self.grid.is_in_bar(agent.x, agent.y)))
        utilities = [records.get(a.id, {}).get("utility", 0.0) for a in agents]
        self.population.observe(agents, brain_actions, epoch, utilities, notes, rerouted)
        self.agent_pool.clear_messages()

    def export_results(self):
        summary = super().export_results()
        with open(os.path.join(self.config.output_dir, MINDS_FILENAME), "w", encoding="utf-8") as f:
            json.dump(self.population.export(), f, indent=2)
        with open(os.path.join(self.config.output_dir, TRACE_FILENAME), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
            writer.writeheader()
            writer.writerows(self.population.trace)
        lies = sum(1 for row in self.population.trace if row["lied"])
        rerouted = sum(1 for row in self.population.trace if row["rerouted"])
        summary["agent_model"] = "minds"
        summary["mind_lie_rate"] = round(lies / len(self.population.trace), 4) if self.population.trace else 0.0
        summary["mind_reroute_rate"] = round(rerouted / len(self.population.trace), 4) if self.population.trace else 0.0
        with open(os.path.join(self.config.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
