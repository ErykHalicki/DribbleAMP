# AWR - Advantage-Weighted Regression

![AWR](../images/AWR_teaser.png)

"Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning"
(https://xbpeng.github.io/projects/AWR/index.html).

---

To train an AWR model, use the following command:
```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/deepmimic_humanoid_env.yaml --agent_config data/agents/deepmimic_humanoid_awr_agent.yaml --visualize false --out_dir output/
```
To test a DeepMimic model, run the following command:
```
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/deepmimic_humanoid_env.yaml --agent_config data/agents/deepmimic_humanoid_awr_agent.yaml --visualize true --model_file data/models/deepmimic_humanoid_awr_spinkick_model.pt
```
The motion data used to train the controller can be specified through `motion_file` in [`data/envs/deepmimic_humanoid_env.yaml`](../data/envs/deepmimic_humanoid_env.yaml). The default configuration trains a controller to imitate a single motion clip. To train a more general controller that can imitate different motion clips, `motion_file` can be used to specify a dataset file, located in [`data/datasets/`](../data/datasets/), which will train a controller to imitate multiple motion clips.

## Citation
```
@article{
	AWRPeng19,
	author = {Xue Bin Peng and Aviral Kumar and Grace Zhang and Sergey Levine},
	title = {Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning},
	journal = {CoRR},
	volume = {abs/1910.00177},
	year = {2019},
	url = {https://arxiv.org/abs/1910.00177},
	archivePrefix = {arXiv},
	eprint = {1910.00177},
	timestamp = {Tue, 01 October 2019 11:27:50 +0200},
	bibsource = {dblp computer science bibliography, https://dblp.org}
}
```