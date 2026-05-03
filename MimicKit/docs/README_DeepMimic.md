# DeepMimic

![DeepMimic](../images/DeepMimic_teaser.png)

"DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills"
(https://xbpeng.github.io/projects/DeepMimic/index.html).

---

To train a DeepMimic model, use the following command:
```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/deepmimic_humanoid_env.yaml --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml --visualize false --out_dir output/
```
To test a DeepMimic model, run the following command:
```
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/deepmimic_humanoid_env.yaml --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml --visualize true --model_file data/models/deepmimic_humanoid_spinkick_model.pt
```
The motion data used to train the controller can be specified through `motion_file` in [`data/envs/deepmimic_humanoid_env.yaml`](../data/envs/deepmimic_humanoid_env.yaml). The default configuration trains a controller to imitate a single motion clip. To train a more general controller that can imitate different motion clips, `motion_file` can be used to specify a dataset file, located in [`data/datasets/`](../data/datasets/), which will train a controller to imitate multiple motion clips.

## Citation
```
@article{
	2018-TOG-deepMimic,
	author = {Peng, Xue Bin and Abbeel, Pieter and Levine, Sergey and van de Panne, Michiel},
	title = {DeepMimic: Example-guided Deep Reinforcement Learning of Physics-based Character Skills},
	journal = {ACM Trans. Graph.},
	issue_date = {August 2018},
	volume = {37},
	number = {4},
	month = jul,
	year = {2018},
	issn = {0730-0301},
	pages = {143:1--143:14},
	articleno = {143},
	numpages = {14},
	url = {http://doi.acm.org/10.1145/3197517.3201311},
	doi = {10.1145/3197517.3201311},
	acmid = {3201311},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {motion control, physics-based character animation, reinforcement learning},
}
```
