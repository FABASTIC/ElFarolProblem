import json


class LLMPipeline:

    def __init__(
        self,
        model_name,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        dtype="half",
    ):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=128,
            top_p=0.9,
        )

    def generate_batch(self, prompts, fallback_positions=None):
        outputs = self.llm.generate(prompts, self.sampling_params)
        results = []
        for i, output in enumerate(outputs):
            text = output.outputs[0].text.strip()
            parsed = self._parse_action(text)
            if parsed is None and fallback_positions is not None:
                parsed = {"move": list(fallback_positions[i]), "speak": ""}
            elif parsed is None:
                parsed = {"move": [0, 0], "speak": ""}
            results.append(parsed)
        return results

    def _parse_action(self, text):
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            if "move" not in data or "speak" not in data:
                return None
            if not isinstance(data["move"], list) or len(data["move"]) != 2:
                return None
            data["move"] = [int(data["move"][0]), int(data["move"][1])]
            if not isinstance(data["speak"], str):
                return None
            return data
        except (ValueError, json.JSONDecodeError, TypeError):
            return None


class MockLLMPipeline:

    def __init__(self, response_fn=None):
        self.response_fn = response_fn

    def generate_batch(self, prompts, fallback_positions=None):
        if self.response_fn is not None:
            return [self.response_fn(i, p) for i, p in enumerate(prompts)]
        results = []
        for i in range(len(prompts)):
            if fallback_positions is not None:
                pos = fallback_positions[i]
                results.append({"move": list(pos), "speak": ""})
            else:
                results.append({"move": [0, 0], "speak": ""})
        return results
