# LCP - Lipschitz-Constrained Policies

![LCP](../images/LCP_teaser.png)

"Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies"
(https://xbpeng.github.io/projects/LCP/index.html).

---

To train an LCP model, use the following command:
```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/deepmimic_g1_env.yaml --agent_config data/agents/lcp_g1_agent.yaml --visualize false --out_dir output/
```
To test an LCP model, run the following command:
```
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/deepmimic_g1_env.yaml --agent_config data/agents/lcp_g1_agent.yaml --visualize true --model_file data/models/lcp_g1_walk_model.pt
```
The weight for the LCP smoothness loss is specified by `lcp_weight` in [`data/agents/lcp_g1_agent.yaml`](../data/agents/lcp_g1_agent.yaml). This is the crucial parameter to tune for LCP to ensure smooth behaviors while maintaining effective task performance. This parameter may need to be tuned for different tasks and motions.

The [`LCP agent`](../mimickit/learning/lcp_agent.py) is designed as a wrapper class, which can inherit from other agent classes and apply the LCP loss to the actor of those agents. The default LCP agent inherits from the PPO agent, but this can be replaced by other agents.


## Citation
```
@article{
	chen2025lcp,
	title = {Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies},
	author = {Zixuan Chen and Xialin He and Yen-Jen Wang and Qiayuan Liao and Yanjie Ze and Zhongyu Li and S. Shankar Sastry and Jiajun Wu and Koushil Sreenath and Saurabh Gupta and Xue Bin Peng},
	journal={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
	year = {2025}
}
```