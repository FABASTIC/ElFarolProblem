import pytest
from elfarol.llm import MockLLMPipeline, LLMPipeline


class TestMockLLMPipeline:

    def test_default_returns_fallback(self):
        mock = MockLLMPipeline()
        results = mock.generate_batch(
            ["prompt1", "prompt2"],
            fallback_positions=[(10, 20), (30, 40)],
        )
        assert len(results) == 2
        assert results[0] == {"move": [10, 20], "speak": ""}
        assert results[1] == {"move": [30, 40], "speak": ""}

    def test_default_no_fallback(self):
        mock = MockLLMPipeline()
        results = mock.generate_batch(["prompt1"])
        assert results[0] == {"move": [0, 0], "speak": ""}

    def test_custom_response_fn(self):
        def fn(idx, prompt):
            return {"move": [idx, idx], "speak": f"agent {idx}"}

        mock = MockLLMPipeline(response_fn=fn)
        results = mock.generate_batch(["p0", "p1", "p2"])
        assert results[0]["move"] == [0, 0]
        assert results[1]["move"] == [1, 1]
        assert results[2]["speak"] == "agent 2"

    def test_batch_size_matches_prompts(self):
        mock = MockLLMPipeline()
        for n in [1, 10, 50]:
            prompts = [f"p{i}" for i in range(n)]
            fallbacks = [(i, i) for i in range(n)]
            results = mock.generate_batch(prompts, fallback_positions=fallbacks)
            assert len(results) == n


class TestLLMPipelineParseAction:

    def setup_method(self):
        self.pipeline = LLMPipeline.__new__(LLMPipeline)

    def test_valid_json(self):
        text = '{"move": [10, 20], "speak": "hello"}'
        result = self.pipeline._parse_action(text)
        assert result == {"move": [10, 20], "speak": "hello"}

    def test_json_with_surrounding_text(self):
        text = 'Here is my action: {"move": [5, 5], "speak": "go"} done'
        result = self.pipeline._parse_action(text)
        assert result == {"move": [5, 5], "speak": "go"}

    def test_missing_move_key(self):
        text = '{"speak": "hello"}'
        result = self.pipeline._parse_action(text)
        assert result is None

    def test_missing_speak_key(self):
        text = '{"move": [1, 2]}'
        result = self.pipeline._parse_action(text)
        assert result is None

    def test_invalid_move_type(self):
        text = '{"move": "bad", "speak": "hello"}'
        result = self.pipeline._parse_action(text)
        assert result is None

    def test_move_wrong_length(self):
        text = '{"move": [1], "speak": "hello"}'
        result = self.pipeline._parse_action(text)
        assert result is None

    def test_speak_not_string(self):
        text = '{"move": [1, 2], "speak": 123}'
        result = self.pipeline._parse_action(text)
        assert result is None

    def test_invalid_json(self):
        text = "not json at all"
        result = self.pipeline._parse_action(text)
        assert result is None

    def test_empty_string(self):
        result = self.pipeline._parse_action("")
        assert result is None

    def test_float_coords_converted(self):
        text = '{"move": [1.5, 2.9], "speak": "ok"}'
        result = self.pipeline._parse_action(text)
        assert result["move"] == [1, 2]
