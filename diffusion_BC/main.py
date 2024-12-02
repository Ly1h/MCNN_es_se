import argparse
import gym
import numpy as np
import os, sys
import torch
import json
import pickle
import tqdm
import tree
import imageio
import pickle
import time
import evosax

import d4rl
from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial import KDTree 

# append outer folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from offlinerlkit.utils.load_dataset import qlearning_dataset_percentbc
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.scaler import StandardScaler

from evil.utils.utils import RealReward, get_expert_obsv_and_actions
from evil.irl.irl_plus import IRLPlus
from evil.utils.parser import get_parser
from evil.utils.plot import plot

import wandb
from evil.utils.utils import (
    get_irl_config,
    get_plot_filename,
    is_irl,
)
from evil.irl.irl import IRL
from evil.irl.bc import BC
from evil.irl.rl import RL
from evil.utils.env_utils import get_env, get_test_params, is_brax_env
import jax.numpy as jnp
from evil.utils.utils import LossType, RewardWrapper, generate_config
import matplotlib.pyplot as plt
from evil.utils.utils import (
    RewardNetwork,
    get_observation_size,
    RewardNetworkPessimistic,
    RewardNetworkPenalty,
)

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
import jax

print("Visible devices", jax.devices())

from evil.training.ppo_v2_cont_irl import (
    make_train as make_train_cont,
    eval as eval_cont,
)
from evil.training.ppo_v2_irl import make_train, eval
from evil.utils.utils import RewardNetwork



# Search for additional_reward
def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in range(len(observations)):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

def qlearning_dataset_with_timeouts(env, dataset=None,
                                    terminate_on_end=False,
                                    disable_goal=True,
                                    **kwargs):
    if dataset is None:
        dataset = env.get_dataset()

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    realdone_ = []
    if "infos/goal" in dataset:
        if not disable_goal:
            dataset["observations"] = np.concatenate(
                [dataset["observations"], dataset['infos/goal']], axis=1)
        else:
            pass

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i]
        new_obs = dataset['observations'][i + 1]
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done_bool = bool(dataset['terminals'][i])
        realdone_bool = bool(dataset['terminals'][i])
        if "infos/goal" in dataset:
            final_timestep = True if (dataset['infos/goal'][i] !=
                                    dataset['infos/goal'][i + 1]).any() else False
        else:
            final_timestep = dataset['timeouts'][i]

        if i < N - 1:
            done_bool += final_timestep

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        realdone_.append(realdone_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_)[:],
        'terminals': np.array(done_)[:],
        'realterminals': np.array(realdone_)[:],
    }

def load_trajectories(name: str, env, dataset, fix_antmaze_timeout=True):
    if "antmaze" in name and fix_antmaze_timeout:
        dataset = qlearning_dataset_with_timeouts(env)
    
    dones_float = np.zeros_like(dataset['rewards'])

    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] -
                        dataset['next_observations'][i]
                        ) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1

    if 'realterminals' in dataset:
        # We updated terminals in the dataset, but continue using
        # the old terminals for consistency with original IQL.
        masks = 1.0 - dataset['realterminals'].astype(np.float32)
    else:
        masks = 1.0 - dataset['terminals'].astype(np.float32)
    traj = split_into_trajectories(
        observations=dataset['observations'].astype(np.float32),
        actions=dataset['actions'].astype(np.float32),
        rewards=dataset['rewards'].astype(np.float32),
        masks=masks,
        dones_float=dones_float.astype(np.float32),
        next_observations=dataset['next_observations'].astype(np.float32))
    return traj

def get_expert_traj(name: str, env, dataset, num_top_episodes=10):
    """Load expert demonstrations."""
    # Load trajectories from the given dataset
    trajs = load_trajectories(name, env, dataset)
    if num_top_episodes < 0:
        print("Loading the entire dataset as demonstrations")
        return trajs

    if 'antmaze' in name:
        returns = [sum([t[2] for t in traj]) / (1e-4 + np.linalg.norm(traj[0][0][:2])) for traj in trajs]
    else:
        returns = [sum([t[2] for t in traj]) for traj in trajs]
    idx = np.argpartition(returns, -num_top_episodes)[-num_top_episodes:]
    
    return [trajs[i] for i in idx]

def get_dataset_return(name, env, dataset):
    trajs = load_trajectories(name, env, dataset)
    episode_return = []
    for transition in trajs:
        N = len(transition)
        reward = 0
        for i in range(N):
            reward += transition[i][2]
        episode_return.append(reward)
    print(episode_return)
    print(np.max(np.array(episode_return)))
    print(np.min(np.array(episode_return)))

def merge_trajectories(trajs):
    flat = []
    for traj in trajs:
        for transition in traj:
            flat.append(transition)
    return tree.map_structure(lambda *xs: np.stack(xs), *flat)


class ReplayBuffer1(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def snapshot_src(src, target, exclude_from):
    make_dir(target)
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)


def squashing_func(distance, action_dim, beta=0.5, scale=1.0, no_action_dim=False):
    if no_action_dim:
        squashed_value = scale * np.exp(-beta * distance)
    else:
        squashed_value = scale * np.exp(-beta * distance/action_dim)
    
    return squashed_value
    
def rewarder(kd_tree, key, num_k, action_dim, beta, scale, no_action_dim=False):
    
    distance, _ = kd_tree.query(key, k=[num_k], workers=-1)
    reward = squashing_func(distance, action_dim, beta, scale, no_action_dim)
    return reward

# AUC for policy
def is_irl(es_config):
    return (
        RealReward[es_config["real_reward"]] == RealReward.IRL_STATE
        or RealReward[es_config["real_reward"]] == RealReward.IRL_STATE_ACTION
        or LossType[es_config["loss"]] == LossType.IRL
        or LossType[es_config["loss"]] == LossType.AUC_TWO_STEP
        or es_config["real_reward"] == RealReward.IRL_STATE
        or es_config["real_reward"] == RealReward.IRL_STATE_ACTION
        or es_config["loss"] == LossType.IRL
        or es_config["loss"] == LossType.AUC_TWO_STEP
    )

def es_loss(metrics):
        if is_irl(es_config):
            print("Optimizing for IRL AUC")
            if es_config["loss_last_only"]:
                return metrics["last_irl_return"].mean()
            else:
                # return metrics["avg_episode_irl_return"].mean()
                return metrics["irl_auc"].mean()
        else:
            print("Optimizing for Real AUC")
            print(es_config["loss_last_only"])
            # return metrics["avg_episode_real_return"].mean()
            if es_config["loss_last_only"]:
                return metrics["last_real_return"].mean()
            else:
                return metrics["real_auc"].mean()


hyperparameters = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 1},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 2},
    'walker2d-medium-v2':            {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 1.0,  'top_k': 1},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 2.0,  'top_k': 0},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 2},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 1},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 7.0,  'top_k': 0},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 2},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 0.5,   'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 2},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0,  'top_k': 2},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 1},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 3.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 4.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 3.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 1},
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-expert-v1':                 {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.1,   'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 8.0,  'top_k': 2},
    'hammer-human-v1':               {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'hammer-expert-v1':              {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'relocate-human-v1':             {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'relocate-expert-v1':            {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'door-human-v1':                 {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'door-expert-v1':                {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'kitchen-complete-v0':           {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 250 , 'gn': 9.0,  'top_k': 2},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
}

def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args):
    # Load buffer
    if args.algo == 'mcnn_bc':
        dataset = qlearning_dataset_percentbc(args.env_name, args.chosen_percentage, args.num_memories_frac, use_random_memories=False, prefix=f'../')
        data_sampler = ReplayBuffer(
            buffer_size=len(dataset["observations"]),
            obs_shape=(state_dim,),
            obs_dtype=np.float32,
            action_dim=action_dim,
            action_dtype=np.float32,
            device=device
        )
        data_sampler.load_dataset(dataset)
        obs_mean, obs_std = data_sampler.normalize_obs()
        utils.print_banner('Loaded mcnn buffer')
        # scaler for normalizing observations at inference
        scaler = StandardScaler(mu=obs_mean, std=obs_std)
    else:
        dataset = d4rl.qlearning_dataset(env)
        data_sampler = Data_Sampler(dataset, device, args.reward_tune)
        utils.print_banner('Loaded buffer')

    if args.algo == 'ql':
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      max_q_backup=args.max_q_backup,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      eta=args.eta,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_maxt=args.num_epochs,
                      grad_norm=args.gn)
    elif args.algo == 'bc':
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      lr=args.lr)
    elif args.algo == 'mcnn_bc':
        from agents.mcnn_bc_diffusion import Diffusion_MCNN_BC as Agent
        agent = Agent(state_dim=state_dim,
                      action_dim=action_dim,
                      max_action=max_action,
                      device=device,
                      discount=args.discount,
                      tau=args.tau,
                      beta_schedule=args.beta_schedule,
                      n_timesteps=args.T,
                      lr=args.lr,
                      Lipz=args.Lipz,
                      lamda=args.lamda,
                      scaler=scaler,
                      dataset=dataset)

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.)
    writer = None  # SummaryWriter(output_dir)

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(data_sampler,
                                  iterations=iterations,
                                  batch_size=args.batch_size,
                                  log_writer=writer)
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # Logging
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        logger.record_tabular('Trained Epochs', curr_epoch)
        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
        logger.dump_tabular()

        # Evaluation
        eval_res, eval_res_std, eval_norm_res, eval_norm_res_std, loss = eval_policy(agent, args.env_name, args.seed,
                                                                               eval_episodes=args.eval_episodes, scaler=scaler if args.algo == 'mcnn_bc' else None, 
                                                                               output_dir=output_dir, save_videos=args.save_videos, curr_epoch=curr_epoch)
        evaluations.append([eval_res, eval_res_std, eval_norm_res, eval_norm_res_std,
                            np.mean(loss_metric['bc_loss']), np.mean(loss_metric['ql_loss']),
                            np.mean(loss_metric['actor_loss']), np.mean(loss_metric['critic_loss']),
                            curr_epoch])
        np.save(os.path.join(output_dir, "eval"), evaluations)
        logger.record_tabular('Average Episodic Reward', eval_res)
        logger.record_tabular('Average Episodic N-Reward', eval_norm_res)
        logger.dump_tabular()

        bc_loss = np.mean(loss_metric['bc_loss'])
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)

        metric = bc_loss

        if args.save_best_model:
            agent.save_model(output_dir, curr_epoch)

    # Model Selection: online or offline
    scores = np.array(evaluations)
    if args.ms == 'online':
        best_id = np.argmax(scores[:, 2])
        best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                    'best normalized score avg': scores[best_id, 2],
                    'best normalized score std': scores[best_id, 3],
                    'best raw score avg': scores[best_id, 0],
                    'best raw score std': scores[best_id, 1]}
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':
        bc_loss = scores[:, 4]
        top_k = min(len(bc_loss) - 1, args.top_k)
        where_k = np.argsort(bc_loss) == top_k
        best_res = {'model selection': args.ms, 'epoch': scores[where_k][0][-1],
                    'best normalized score avg': scores[where_k][0][2],
                    'best normalized score std': scores[where_k][0][3],
                    'best raw score avg': scores[where_k][0][0],
                    'best raw score std': scores[where_k][0][1]}

        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    
    self.actor_optimizer.zero_grad() 
    loss.backward() 
    self.actor_optimizer.step() 
    # writer.close()

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, scaler=None, output_dir=None, save_videos=False, curr_epoch=None):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    # if curr_epoch < args.num_epochs * 80 // 100:
    #     save_videos = False # overwrite the default save_videos to False until the end of training
    if save_videos:
        video_folder = f'{output_dir}/saved_videos'
        os.makedirs(video_folder, exist_ok=True)
    total_loss = 0
    lamda_1 = 0.4
    lamda_2 = 0.3
    lamda_3 = 0.3
    scores = []
    for ep in range(eval_episodes):
        traj_return = 0.
        arrs = []
        state, done = eval_env.reset(), False
        while not done:
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)

            replay_buffer = ReplayBuffer1(state_dim, action_dim)
            dataset = d4rl.qlearning_dataset(env)
            replay_buffer.convert_D4RL(dataset)
    
            data = get_expert_traj(args.env_name, env, dataset, num_top_episodes=1)
            data = merge_trajectories(data)
    
            if args.mode == 'sas':
                data = np.hstack([data[0], data[1], data[5]])  # stack state and action, and next state
                kd_tree = KDTree(data)
                # query every sample
                key = np.hstack([replay_buffer.state, replay_buffer.action, replay_buffer.next_state])
            elif args.mode == 'sa':
                data = np.hstack([data[0], data[1]])  # stack state and action
                kd_tree = KDTree(data)
                # query every sample
                key = np.hstack([replay_buffer.state, replay_buffer.action])
            elif args.mode == 'ss':
                data = np.hstack([data[0], data[5]])  # stack state and next state
                kd_tree = KDTree(data)
                # query every sample
                key = np.hstack([replay_buffer.state, replay_buffer.next_state])

            additional_reward = rewarder(kd_tree, key, args.k, action_dim, args.beta, args.scale, args.no_action_dim)
    
            replay_buffer.reward = additional_reward

            # adding reward bias
            if 'antmaze' in args.env_name:
                replay_buffer.reward -= args.bias * args.scale
                                        
            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                # IQL
                "discount": args.discount,
                "tau": args.tau,
                "temperature": args.temperature,
                "expectile": args.expectile,
                "dropout_rate": float(args.dropout_rate) if args.dropout_rate is not None else None,
            }

            if save_videos:
                curr_frame = eval_env.sim.render(width=640, height=480, mode='offscreen', camera_name=None, device_id=0)
                arrs.append(curr_frame[::-1, :, :])

            reward=reward+additional_reward
            L1_loss = np.mean(loss_metric['bc_loss'])
            L2_loss = -torch.sum(torch.tensor(additional_reward, dtype=torch.float32, device=device))
            L3_loss = jax.vmap(es_loss)
            total_loss = lamda_1 * L1_loss +lamda_2 * L2_loss + lamda_3 * L3_loss
            traj_return += reward
        scores.append(traj_return)
        if save_videos:
            skvideo.io.vwrite( f'{video_folder}/epoch_{curr_epoch}_episode_{ep+1}.mp4', np.asarray(arrs))

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    return avg_reward, std_reward, avg_norm_score, std_norm_score, total_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default='exp_1', type=str)                    # Experiment ID
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="walker2d-medium-expert-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--dir", default="results", type=str)                    # Logging directory
    parser.add_argument("--seed", default=0, type=int)                         # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default='vp', type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="ql", type=str)  # ['mcnn_bc', 'bc', 'ql']
    parser.add_argument("--ms", default='offline', type=str, help="['online', 'offline']")
    # parser.add_argument("--top_k", default=1, type=int)

    # parser.add_argument("--lr", default=3e-4, type=float)
    # parser.add_argument("--eta", default=1.0, type=float)
    # parser.add_argument("--max_q_backup", action='store_true')
    # parser.add_argument("--reward_tune", default='no', type=str)
    # parser.add_argument("--gn", default=-1.0, type=float)

    # MCNN
    parser.add_argument('--chosen-percentage', type=float, default=1.0)
    parser.add_argument('--num_memories_frac', type=float, default=0.1)
    parser.add_argument('--Lipz', type=float, default=1.0)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--save_videos', action='store_true', default=False)

    # related Parameters
    parser.add_argument('--mode', default='sas', type=str) # different modes of search, support sas, sa, ss
    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--beta", default=0.5, type=float)                      # coefficient in distance
    parser.add_argument("--scale", default=1.0, type=float)
    parser.add_argument("--no_action_dim", action="store_true", default=False)     # whether to involve action dimension
    parser.add_argument("--bias", default=1.0, type=float)
    parser.add_argument("--temperature", default=3.0, type=float)
    parser.add_argument("--expectile", default=0.7, type=float)
    parser.add_argument("--dropout_rate", default=None)

    args = parser.parse_args()

    if args.save_videos:
        import skvideo.io

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f'{args.dir}'

    args.num_epochs = hyperparameters[args.env_name]['num_epochs']
    args.eval_freq = hyperparameters[args.env_name]['eval_freq']
    args.eval_episodes = 10 if 'v2' in args.env_name else 100
    if args.save_videos:
        args.eval_episodes = 10 # overwrite the default eval_episodes to fewer episodes for saving videos

    args.lr = hyperparameters[args.env_name]['lr']
    args.eta = hyperparameters[args.env_name]['eta']
    args.max_q_backup = hyperparameters[args.env_name]['max_q_backup']
    args.reward_tune = hyperparameters[args.env_name]['reward_tune']
    args.gn = hyperparameters[args.env_name]['gn']
    args.top_k = hyperparameters[args.env_name]['top_k']

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay: file_name += '|lr_decay'
    file_name += f'|ms-{args.ms}'
    if args.algo == 'mcnn_bc': 
        file_name += f'|chosen-{args.chosen_percentage}|mem-{args.num_memories_frac}|Lipz-{args.Lipz}|lamda-{args.lamda}'

    if args.ms == 'offline': file_name += f'|k-{args.top_k}'
    file_name += f'|{args.seed}'

    results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(version=f"Diffusion-Policies-RL")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    train_agent(env,
                state_dim,
                action_dim,
                max_action,
                args.device,
                results_dir,
                args)
