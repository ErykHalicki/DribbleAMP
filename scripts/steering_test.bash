cd MimicKit

python mimickit/run.py --mode train --num_envs 1024 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_steering_env.yaml --agent_config data/agents/amp_task_humanoid_agent.yaml --visualize false --out_dir output/
