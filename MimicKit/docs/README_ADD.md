# ADD - Adversarial Differential Discriminator

![ADD](../images/ADD_teaser.png)

"Physics-Based Motion Imitation with Adversarial Differential Discriminators"
(https://xbpeng.github.io/projects/ADD/index.html).

---

To train an ADD model, use the following command:
```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/add_humanoid_env.yaml --agent_config data/agents/add_humanoid_agent.yaml --visualize false --out_dir output/
```
To test an ADD model, run the following command:
```
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/add_humanoid_env.yaml --agent_config data/agents/add_humanoid_agent.yaml --visualize true --model_file data/models/add_humanoid_spinkick_model.pt
```

## Citation
```
@inproceedings{
	zhang2025ADD,
    author={Zhang, Ziyu and Bashkirov, Sergey and Yang, Dun and Shi, Yi and Taylor, Michael and Peng, Xue Bin},
    title = {Physics-Based Motion Imitation with Adversarial Differential Discriminators},
    year = {2025},
    booktitle = {SIGGRAPH Asia 2025 Conference Papers (SIGGRAPH Asia '25 Conference Papers)}
}
```
