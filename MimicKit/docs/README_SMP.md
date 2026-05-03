# SMP - Score-Matching Motion Priors

![SMP](../images/SMP_teaser.png)

"SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control" (https://xbpeng.github.io/projects/SMP/index.html).

## Train Task Policies with the LaFAN1 Prior

Use the pretrained LaFAN1 prior to train a task policy for the `location`, `steering`, or `dodgeball` task:

```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/smp_location_humanoid_env.yaml --agent_config data/agents/smp_task_humanoid_agent.yaml --visualize false --out_dir output/
```

The default agent config [`data/agents/smp_task_humanoid_agent.yaml`](../data/agents/smp_task_humanoid_agent.yaml) enables Generative State Initialization with `enable_gsi: True`. **No motion data is used during policy training.**

To test a trained model, run:

```
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/smp_location_humanoid_env.yaml --agent_config data/agents/smp_task_humanoid_agent.yaml --visualize true --model_file data/models/smp_location_humanoid_model.pt
```

## Train New Priors

A new prior can be trained with the following command:

```
python tools/diffusion_model/train_tinymdm.py --cfg_path tools/diffusion_model/config/tinymdm_multi_clip.yaml --out_dir output/smp_prior
```

The motion dataset is specified in the config file [`tools/diffusion_model/config/tinymdm_multi_clip.yaml`](../tools/diffusion_model/config/tinymdm_multi_clip.yaml) via `motion_file`.

Once the prior has been trained, it can be used to train a policy by modifying the agent config [`data/agents/smp_task_humanoid_agent.yaml`](../data/agents/smp_task_humanoid_agent.yaml) to use the new prior. This can be done by setting the fields in the agent config:

- `smp_prior_cfg: output/smp_prior/diffusion_config.yaml`
- `smp_prior_model: output/smp_prior/model.pt`

Then to train a policy, run:

```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/smp_steering_humanoid_env.yaml --agent_config data/agents/smp_task_humanoid_agent.yaml --visualize false --out_dir output/
```

The balance between task and prior rewards is specified by `task_reward_weight` and `smp_reward_weight` in [`data/agents/smp_task_humanoid_agent.yaml`](../data/agents/smp_task_humanoid_agent.yaml). These parameters can be used to control how closely the model follows the motion prior versus optimizing the task objective.

Other useful hyperparameters to tune are:

- `sds_loss_scale`
- `diffusion_steps`

Empirically, the tuning priority is:

```text
smp_reward_weight > sds_loss_scale >= diffusion_steps
```

## Train Single-Clip Models

To train a prior on a single motion clip, run:

```
python tools/diffusion_model/train_tinymdm.py --cfg_path tools/diffusion_model/config/tinymdm_single_clip.yaml --out_dir output/smp_prior
```
The motion data used to train the prior can be specified through `motion_file` in [`tools/diffusion_model/config/tinymdm_single_clip.yaml`](../tools/diffusion_model/config/tinymdm_single_clip.yaml).

Then train the policy:

```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/smp_humanoid_env.yaml --agent_config data/agents/smp_humanoid_agent.yaml --visualize false --out_dir output/
```

To test the policy:

```
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/smp_humanoid_env.yaml --agent_config data/agents/smp_humanoid_agent.yaml --visualize true --model_file data/models/smp_humanoid_spinkick_model.pt
```
GSI is disabled for the single-clip experiments. Therefore, the motion data used for state initialization must be specified through `motion_file` in [`data/envs/smp_humanoid_env.yaml`](../data/envs/smp_humanoid_env.yaml).

## Citation

```bibtex
@article{
	mu2026smp,
	title = {SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control},
	author = {Mu, Yuxuan and Zhang, Ziyu and Shi, Yi and Yang, Dun and
             Matsumoto, Minami and Imamura, Kotaro and Tevet, Guy and
             Guo, Chuan and Taylor, Michael and Shu, Chang and
             Xi, Pengcheng and Peng, Xue Bin},
	journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH 2026)},
	year = {2026}
}
```
