import json
import csv
import os
from collections import defaultdict


BAR_CAPACITY = 100
COMFORT_THRESHOLD_RATIO = 0.6
UTILITY_AT_BAR_COMFORTABLE = 1.0
UTILITY_AT_BAR_OVERCROWDED = -1.0
UTILITY_AT_HOME = 0.3


class MetricsLogger:

    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.epoch_snapshots = []
        self.agent_cumulative_utility = defaultdict(float)
        self.agent_deception_counts = defaultdict(int)
        self.agent_truthful_counts = defaultdict(int)
        self.broadcast_log = []
        self.attendance_history = []
        self.fallback_count = 0
        self.total_generations = 0
        self.regex_fallback_rate = 0.0

    def record_fallbacks(self, fallbacks, total):
        self.fallback_count += fallbacks
        self.total_generations += total
        if self.total_generations > 0:
            self.regex_fallback_rate = round(self.fallback_count / self.total_generations, 4)
        else:
            self.regex_fallback_rate = 0.0

    def get_regex_fallback_rate(self):
        return self.regex_fallback_rate


    def compute_utility(self, agent, grid, bar_occupancy, bar_capacity):
        threshold = int(COMFORT_THRESHOLD_RATIO * bar_capacity)
        in_bar = grid.is_in_bar(agent.x, agent.y)
        if in_bar:
            if bar_occupancy <= threshold:
                return UTILITY_AT_BAR_COMFORTABLE
            else:
                return UTILITY_AT_BAR_OVERCROWDED
        return UTILITY_AT_HOME

    def record_epoch(self, epoch, agents, grid, brain_actions):
        bar_occ = grid.occupancy()
        bar_capacity = (grid.bar_max - grid.bar_min) ** 2
        threshold = int(COMFORT_THRESHOLD_RATIO * bar_capacity)

        agent_records = []
        epoch_deception_count = 0
        epoch_truthful_count = 0
        epoch_total_utility = 0.0

        for agent, action in zip(agents, brain_actions):
            utility = self.compute_utility(agent, grid, bar_occ, bar_capacity)
            agent.utility_history.append(utility)
            self.agent_cumulative_utility[agent.id] += utility

            in_bar = grid.is_in_bar(agent.x, agent.y)
            actual_location = "bar" if in_bar else "home"

            stated = action.get("stated_intention", "staying")
            declared_target = action.get("actual_target", "home")

            is_deceptive = self._check_deception(stated, actual_location)
            if is_deceptive:
                epoch_deception_count += 1
                self.agent_deception_counts[agent.id] += 1
            else:
                epoch_truthful_count += 1
                self.agent_truthful_counts[agent.id] += 1

            epoch_total_utility += utility

            agent_records.append({
                "agent_id": agent.id,
                "x": agent.x,
                "y": agent.y,
                "in_bar": in_bar,
                "utility": utility,
                "cumulative_utility": self.agent_cumulative_utility[agent.id],
                "stated_intention": stated,
                "actual_target": declared_target,
                "actual_location": actual_location,
                "is_deceptive": is_deceptive,
                "broadcast": action.get("broadcast"),
            })

        total_agents = len(agents)
        truthfulness_ratio = (
            epoch_truthful_count / total_agents if total_agents > 0 else 0.0
        )
        deception_index = (
            epoch_deception_count / total_agents if total_agents > 0 else 0.0
        )

        broadcast_msgs = [
            a.get("broadcast") for a in brain_actions
            if a.get("broadcast") is not None
        ]
        self.broadcast_log.extend(broadcast_msgs)
        self.attendance_history.append(bar_occ)

        broadcast_correlation = self._compute_broadcast_correlation(epoch)

        snapshot = {
            "epoch": epoch,
            "bar_attendance": bar_occ,
            "bar_capacity": bar_capacity,
            "comfort_threshold": threshold,
            "attendance_over_threshold": bar_occ > threshold,
            "total_utility": epoch_total_utility,
            "mean_utility": epoch_total_utility / total_agents if total_agents > 0 else 0.0,
            "deception_index": deception_index,
            "truthfulness_ratio": truthfulness_ratio,
            "deception_count": epoch_deception_count,
            "truthful_count": epoch_truthful_count,
            "broadcast_count": len(broadcast_msgs),
            "broadcast_correlation": broadcast_correlation,
            "agents": agent_records,
        }

        self.epoch_snapshots.append(snapshot)
        return snapshot

    def _check_deception(self, stated_intention, actual_location):
        stated_going = stated_intention == "going"
        actually_at_bar = actual_location == "bar"
        if stated_going and not actually_at_bar:
            return True
        if not stated_going and actually_at_bar:
            return True
        return False

    def _compute_broadcast_correlation(self, epoch):
        if len(self.attendance_history) < 2:
            return 0.0
        prev_attendance = self.attendance_history[-2] if len(self.attendance_history) >= 2 else 0
        curr_attendance = self.attendance_history[-1]
        shift = curr_attendance - prev_attendance

        recent_broadcast_count = 0
        if self.epoch_snapshots:
            recent_broadcast_count = sum(
                1 for s in self.epoch_snapshots[-3:]
                if s.get("broadcast_count", 0) > 0
            )

        if recent_broadcast_count == 0:
            return 0.0
        return round(shift / max(1, recent_broadcast_count), 4)

    def get_agent_deception_ratio(self, agent_id):
        total_d = self.agent_deception_counts.get(agent_id, 0)
        total_t = self.agent_truthful_counts.get(agent_id, 0)
        total = total_d + total_t
        if total == 0:
            return 0.0
        return total_d / total

    def get_population_truthfulness(self):
        total_d = sum(self.agent_deception_counts.values())
        total_t = sum(self.agent_truthful_counts.values())
        total = total_d + total_t
        if total == 0:
            return 1.0
        return total_t / total

    def export_json(self, filepath=None):
        if filepath is None and self.output_dir is not None:
            filepath = os.path.join(self.output_dir, "epoch_metrics.json")
        elif filepath is None:
            filepath = "epoch_metrics.json"

        export_data = []
        for snap in self.epoch_snapshots:
            epoch_export = {k: v for k, v in snap.items() if k != "agents"}
            epoch_export["agent_count"] = len(snap.get("agents", []))
            export_data.append(epoch_export)

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        return filepath

    def export_csv(self, filepath=None):
        if filepath is None and self.output_dir is not None:
            filepath = os.path.join(self.output_dir, "epoch_metrics.csv")
        elif filepath is None:
            filepath = "epoch_metrics.csv"

        fieldnames = [
            "epoch", "bar_attendance", "bar_capacity", "comfort_threshold",
            "attendance_over_threshold", "total_utility", "mean_utility",
            "deception_index", "truthfulness_ratio", "deception_count",
            "truthful_count", "broadcast_count", "broadcast_correlation",
        ]

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for snap in self.epoch_snapshots:
                row = {k: snap.get(k, "") for k in fieldnames}
                writer.writerow(row)
        return filepath

    def export_agent_csv(self, filepath=None):
        if filepath is None and self.output_dir is not None:
            filepath = os.path.join(self.output_dir, "agent_epoch_details.csv")
        elif filepath is None:
            filepath = "agent_epoch_details.csv"

        fieldnames = [
            "epoch", "agent_id", "x", "y", "in_bar", "utility",
            "cumulative_utility", "stated_intention", "actual_target",
            "actual_location", "is_deceptive", "broadcast",
        ]

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for snap in self.epoch_snapshots:
                for rec in snap.get("agents", []):
                    row = {"epoch": snap["epoch"]}
                    row.update({k: rec.get(k, "") for k in fieldnames if k != "epoch"})
                    writer.writerow(row)
        return filepath

    def summary(self):
        if not self.epoch_snapshots:
            return {}
        total_epochs = len(self.epoch_snapshots)
        avg_attendance = (
            sum(s["bar_attendance"] for s in self.epoch_snapshots) / total_epochs
        )
        avg_deception = (
            sum(s["deception_index"] for s in self.epoch_snapshots) / total_epochs
        )
        avg_utility = (
            sum(s["mean_utility"] for s in self.epoch_snapshots) / total_epochs
        )
        return {
            "total_epochs": total_epochs,
            "average_bar_attendance": round(avg_attendance, 4),
            "average_deception_index": round(avg_deception, 4),
            "average_mean_utility": round(avg_utility, 4),
            "population_truthfulness": round(self.get_population_truthfulness(), 4),
            "regex_fallback_rate": getattr(self, "regex_fallback_rate", 0.0),
        }

