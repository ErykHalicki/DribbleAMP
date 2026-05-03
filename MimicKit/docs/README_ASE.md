# ASE - Adversarial Skill Embeddings

![ASE](../images/ASE_teaser.png)

"ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters"
(https://xbpeng.github.io/projects/ASE/index.html).

---

To train an ASE model, use the following command:
```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/ase_humanoid_sword_shield_env.yaml --agent_config data/agents/ase_humanoid_agent.yaml --visualize false --out_dir output/
```
To test an ASE model, run the following command:
```
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/ase_humanoid_sword_shield_env.yaml --agent_config data/agents/ase_humanoid_agent.yaml --visualize true --model_file data/models/ase_humanoid_sword_shield_model.pt
```

## Citation
```
@article{
	2022-TOG-ASE,
	author = {Peng, Xue Bin and Guo, Yunrong and Halper, Lina and Levine, Sergey and Fidler, Sanja},
	title = {ASE: Large-scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters},
	journal = {ACM Trans. Graph.},
	issue_date = {August 2022},
	volume = {41},
	number = {4},
	month = jul,
	year = {2022},
	articleno = {94},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {motion control, physics-based character animation, reinforcement learning}
}
```
