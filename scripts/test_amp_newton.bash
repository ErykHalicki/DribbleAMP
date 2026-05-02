cd MimicKit

python mimickit/run.py --mode train --num_envs 1 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_humanoid_env.yaml --agent_config data/agents/amp_humanoid_agent.yaml --visualize false --out_dir output/
