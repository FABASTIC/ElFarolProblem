import argparse
from elfarol.simulation_runner import (
    RunConfig,
    SimulationRunner,
    _build_vllm_pipeline,
    VLLMBatchPipeline,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model", type=str, default="casperhansen/llama-3-8b-instruct-awq")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/smoke_test")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    config = RunConfig(
        model_name=args.model,
        num_agents=args.agents,
        num_epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    if args.mock:
        from elfarol.llm import MockLLMPipeline
        pipeline = MockLLMPipeline()
    else:
        llm, sampling_params = _build_vllm_pipeline(config)
        pipeline = VLLMBatchPipeline(llm=llm, sampling_params=sampling_params)

    runner = SimulationRunner(config=config, llm_pipeline=pipeline)
    logger = runner.run()

    print(f"Simulation complete: {args.epochs} epochs, {args.agents} agents")
    print(f"Final bar occupancy: {logger.attendance_history[-1] if logger.attendance_history else 0}")
    print(f"Regex fallback rate: {getattr(logger, 'regex_fallback_rate', 0.0)}")


if __name__ == "__main__":
    main()
