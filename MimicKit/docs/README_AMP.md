# AMP - Adversarial Motion Priors

![AMP](../images/AMP_teaser.png)

"AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"
(https://xbpeng.github.io/projects/AMP/index.html).

---

To train a AMP model, use the following command:
```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/amp_humanoid_env.yaml --agent_config data/agents/amp_humanoid_agent.yaml --visualize false --out_dir output/
```
To test a AMP model, run the following command:
```
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/amp_humanoid_env.yaml --agent_config data/agents/amp_humanoid_agent.yaml --visualize true --model_file data/models/amp_humanoid_spinkick_model.pt
```
The default configuration [`data/agents/amp_humanoid_agent.yaml`](../data/agents/amp_humanoid_agent.yaml), trains controllers only with an imitation objective, without any task objectives. Controllers can be trained with a combination of imitation and task objectives using [`data/agents/amp_task_humanoid_agent.yaml`](../data/agents/amp_task_humanoid_agent.yaml) with the following command:
```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/amp_location_humanoid_env.yaml --agent_config data/agents/amp_task_humanoid_agent.yaml --visualize false --out_dir output/
```
To test the model, run the following command:
```
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/amp_location_humanoid_env.yaml --agent_config data/agents/amp_task_humanoid_agent.yaml --visualize true --model_file data/models/amp_location_humanoid_model.pt
```
The weights used to balance the imitation and task rewards are specified by `disc_reward_weight` and `task_reward_weight` in the agent configuration file [`data/agents/amp_task_humanoid_agent.yaml`](../data/agents/amp_task_humanoid_agent.yaml). These parameters can be used to control how closely the model follows the motion data versus optimizing the task objective.

## Citation
```
@article{
	2021-TOG-AMP,
	author = {Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
	title = {AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control},
	journal = {ACM Trans. Graph.},
	issue_date = {August 2021},
	volume = {40},
	number = {4},
	month = jul,
	year = {2021},
	articleno = {1},
	numpages = {15},
	url = {http://doi.acm.org/10.1145/3450626.3459670},
	doi = {10.1145/3450626.3459670},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {motion control, physics-based character animation, reinforcement learning},
}
```
