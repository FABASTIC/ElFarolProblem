import json
import os
from unittest.mock import MagicMock

import pytest

from experiment import (
    _AGENT_SYSTEM_PROMPT,
    _InstructChannel,
    _chat_messages,
    _degrade_profile,
    _destroy_vllm,
    _save_comparison,
    run_experiment,
)
from elfarol.simulation_runner import VLLMBatchPipeline


class _Choice:

    def __init__(self, text):
        self.text = text


class _Output:

    def __init__(self, text):
        self.outputs = [_Choice(text)]


class FakeTokenizer:

    chat_template = "llama"
    bos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<s>" + "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"

    def encode(self, text, add_special_tokens=True):
        ids = [ord(c) % 1000 + 1 for c in text]
        return [0] + ids if add_special_tokens else ids


class FakeLLM:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.batches = []

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("no tokenizer")
        return self.tokenizer

    def generate(self, prompts, sampling_params=None, **kwargs):
        self.batches.append(list(prompts))
        return [_Output('{"move": [1, 1]}') for _ in prompts]


class TestChatMessages:

    def test_system_prompt_moves_to_system_role(self):
        messages = _chat_messages(_AGENT_SYSTEM_PROMPT + "\n\nEpoch: 3")
        assert messages[0] == {"role": "system", "content": _AGENT_SYSTEM_PROMPT}
        assert messages[1] == {"role": "user", "content": "Epoch: 3"}

    def test_foreign_prompt_stays_user_turn(self):
        assert _chat_messages("hello") == [{"role": "user", "content": "hello"}]


class TestInstructChannel:

    def test_chat_template_emits_token_ids(self):
        llm = FakeLLM(FakeTokenizer())
        channel = _InstructChannel(llm, max_model_len=4096, max_new_tokens=256, chat_template=True)
        channel.generate(["hello"], None)
        sent = llm.batches[0][0]
        assert isinstance(sent, dict) and sent["prompt_token_ids"][0] != 0
        assert channel.templated_count == 1 and channel.prompt_count == 1

    def test_raw_mode_matches_engine_tokenisation(self):
        llm = FakeLLM(FakeTokenizer())
        channel = _InstructChannel(llm, max_model_len=4096, max_new_tokens=256, chat_template=False)
        channel.generate(["abc"], None)
        assert llm.batches[0][0]["prompt_token_ids"] == FakeTokenizer().encode("abc")
        assert channel.templated_count == 0

    def test_overlong_prompt_is_truncated_below_context_limit(self):
        llm = FakeLLM(FakeTokenizer())
        channel = _InstructChannel(llm, max_model_len=64, max_new_tokens=16, chat_template=False)
        prompt = "H" * 40 + "M" * 200 + "T" * 40
        channel.generate([prompt], None)
        ids = llm.batches[0][0]["prompt_token_ids"]
        assert len(ids) < 64
        assert ids[1] == ord("H") + 1 and ids[-1] == ord("T") + 1
        assert channel.overflow_count == 1

    def test_missing_tokenizer_passes_prompts_through(self):
        llm = FakeLLM(None)
        channel = _InstructChannel(llm, max_model_len=2048, max_new_tokens=256)
        channel.generate(["plain"], None)
        assert llm.batches[0] == ["plain"]


class TestTeardownHooks:

    def test_destroy_shuts_down_engine_core(self):
        llm = MagicMock()
        _destroy_vllm(llm)
        llm.llm_engine.engine_core.shutdown.assert_called_once()

    def test_destroy_tolerates_missing_engine(self):
        _destroy_vllm(None)

    def test_degrade_profile_halves_batching_with_floors(self):
        base = {"gpu_memory_utilization": 0.85, "max_num_seqs": 64, "max_num_batched_tokens": 2048, "kv_cache_dtype": "auto"}
        assert _degrade_profile(base, 1) == base
        assert _degrade_profile(base, 2)["max_num_batched_tokens"] == 1024
        assert _degrade_profile(base, 6)["max_num_batched_tokens"] == 512
        assert _degrade_profile(base, 6)["max_num_seqs"] == 8

    def test_comparison_write_leaves_no_temp_files(self, tmp_path):
        _save_comparison({"control": {"total_epochs": 1}}, str(tmp_path))
        assert sorted(os.listdir(tmp_path)) == ["comparison.json"]


def _pipeline():
    def respond(prompts, sampling_params=None, **kwargs):
        return [_Output('{"move": [9, 9], "broadcast": "hi", "proximity_speech": null, "stated_intention": "going", "actual_target": "bar"}') for _ in prompts]

    llm = MagicMock()
    llm.generate.side_effect = respond
    return VLLMBatchPipeline(llm, sampling_params=None)


class TestSweepArmor:

    def test_live_state_and_manifests_are_published(self, tmp_path):
        summaries, _ = run_experiment(
            model_name="test-model",
            base_output_dir=str(tmp_path),
            num_agents=4,
            num_epochs=3,
            grid_size=20,
            seeds=[7, 8],
            dry_run_pipeline=_pipeline(),
        )
        with open(tmp_path / "live_state.json", "r", encoding="utf-8") as f:
            live = json.load(f)
        assert live["status"] == "complete"
        assert [t["status"] for t in live["trials"]] == ["complete"] * 4
        assert len(live["trials"]) == 4
        for key, summary in summaries.items():
            assert summary["trial"] == key
            assert summary["status"] == "complete"
            with open(tmp_path / summary["trial_dir"] / "trial_manifest.json", "r", encoding="utf-8") as f:
                assert json.load(f)["seed"] == summary["seed"]

    def test_resume_skips_verified_trials(self, tmp_path):
        kwargs = dict(model_name="test-model", base_output_dir=str(tmp_path), num_agents=4, num_epochs=3, grid_size=20, seeds=[7, 8])
        run_experiment(dry_run_pipeline=_pipeline(), **kwargs)
        stamp = os.path.getmtime(tmp_path / "control_seed_7" / "epoch_metrics.csv")
        idle = _pipeline()
        summaries, _ = run_experiment(dry_run_pipeline=idle, resume=True, **kwargs)
        assert idle.llm.generate.call_count == 0
        assert os.path.getmtime(tmp_path / "control_seed_7" / "epoch_metrics.csv") == stamp
        assert all(s["runtime"].get("resumed") for s in summaries.values())

    def test_resume_reruns_mismatched_configuration(self, tmp_path):
        kwargs = dict(model_name="test-model", base_output_dir=str(tmp_path), num_agents=4, grid_size=20, seeds=[7])
        run_experiment(dry_run_pipeline=_pipeline(), num_epochs=2, **kwargs)
        rerun = _pipeline()
        run_experiment(dry_run_pipeline=rerun, resume=True, num_epochs=3, **kwargs)
        assert rerun.llm.generate.call_count == 6
