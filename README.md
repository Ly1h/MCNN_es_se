# MCNN_es_se

## How to run

## Setup
Create env, install pytorch, install requirements.
```bash
conda create -n MCNN_env python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
Setup mujoco210 by following the instructions from https://github.com/openai/mujoco-py#install-mujoco.
In case you run across a gcc error, please follow the trouble shooting instructions [here if you have sudo access](https://github.com/openai/mujoco-py#ubuntu-installtion-troubleshooting) or [here otherwise](https://github.com/openai/mujoco-py/issues/627#issuecomment-1383054926).

Install this package
```bash
pip install -e .
```
## Train and evaluate
cd diffusion_BC
python main.py --algo mcnn_bc --env_name pen-human-v1 --device 0 --ms online --lr_decay --num_memories_frac 0.1 --Lipz 1.0 --lamda 1.0
```
Replace `pen-human-v1` with any of the other tasks such as (hammer-human-v1, pen-human-v1, relocate-human-v1, door-human-v1, hammer-expert-v1, pen-expert-v1, relocate-expert-v1, door-expert-v1, carla-lane-v0).

### Create Memories with Neural Gas
Create memories:
```bash
python mems_obs/create_gng_incrementally.py --name pen-human-v1 --num_memories_frac 0.1
```
Replace name with any of the other tasks and num_memories_frac with any value less than 1. In the paper, we use 0.025, 0.05, and 0.1 for num_memories_frac. (Note: simply use `--name kitchen` for the franka kitchen task)

Update (downloaded) datasets by adding memory and memory_target to every transition:
```bash
python mems_obs/update_data.py --name pen-human-v1 --num_memories_frac 0.1
```
Similar to above, replace name with any of the other tasks and num_memories_frac with any value less than 1.

### Create Random Memories
Create random subset of all observations as memories and update (downloaded) datasets by adding memory and memory_target to every transition:
```bash
python mems_obs/update_data_random_mems.py --name pen-human-v1 --num_memories_frac 0.1
```
Similar to above, replace name with any of the other tasks and num_memories_frac with any value less than 1.
