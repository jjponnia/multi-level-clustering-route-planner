# import ppo_attention_core
from grid_world_env import gridworld_env
from buffers import PathPlanningBuffer
import numpy as np
import torch
from algos_scratch_pad.utils.logx import EpochLogger
from algos_scratch_pad.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from algos_scratch_pad.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from torch.optim import Adam
# from route_planning_model import GAT_TSP
from path_planning_model import PathPlanningModel
# from attention_model.route_planning_model import RoutePlanningModel
from train_path_planning_w_expertise import train_path_planning_w_expertise
# from evaluate_baseline import evaluate_model_performance
from evaluate_path_model_baseline import evaluate_model_performance

import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
import itertools
from torch_geometric.data import Data
import copy


def il_path_planning(env_fn, seed=0, steps_per_epoch=500, epochs=10000, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01,
        logger_kwargs=dict(), save_freq=10, evaluate_model=10, test_batch=100):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_scalar = GradScaler()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    target_weight = 0.5
    action_weight = 0.5

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    num_targets = env.num_targets
    # baseline_env = env_fn()
    # num_acts = env.num_targets
    # num_agents = env.num_agents
    # obs_dim = 2 * num_acts + 2 * num_agents
    obs_size = 2 * env.num_obstacles + 2 * env.num_agents + 2 * env.num_targets + 1
    # obs_dim = 9
    # n_heads = 4


    hidden_sizes = 128
    # route_model = GAT_TSP(input_dim=obs_dim, hidden_dim=hidden_sizes, heads=n_heads).to(device)
    # baseline_model = GAT_TSP(input_dim=obs_dim, hidden_dim=hidden_sizes, heads=n_heads).to(device)
    path_model = PathPlanningModel(obs_dim=obs_size, n_targets=env.num_targets, n_actions=env.n_acts).to(device)
    # baseline_model = PathPlanningModel(obs_dim=obs_dim, n_acts=4).to(device)

    path_model_optimizer = Adam(path_model.parameters(), lr=pi_lr)
    # baseline_model_optimizer = Adam(baseline_model.parameters(), lr=pi_lr)
    path_planning_buf = PathPlanningBuffer(obs_size, num_targets)
    # baseline_route_planning_buf = PathPlanningBuffer(baseline_env.num_targets)

    def compute_imitation_loss(data):
        # data = dict(obs=self.obs_buf, mask=self.mask_buf, act=self.act_buf, route=self.route_buf,
        #             route_mask=self.route_mask_buf)
        obs = torch.as_tensor(data['obs'], dtype=torch.float32).to(device)
        action_mask = torch.as_tensor(data['mask'], dtype=torch.bool).to(device)
        acts = torch.as_tensor(data['act'], dtype=torch.long).to(device)
        route = torch.as_tensor(data['route'], dtype=torch.long).to(device)
        route_mask = torch.as_tensor(data['route_mask'], dtype=torch.bool).to(device)

        target_dist, action_dist = path_model(obs, route_mask, action_mask)

        target_probs = target_dist.probs
        action_probs = action_dist.probs

        # print(f"target_probs: {target_probs}")
        # print(f"route: {route}")

        target_loss = F.cross_entropy(target_probs, route)
        action_loss = F.cross_entropy(action_probs, acts)

        imitation_loss = target_weight * target_loss + action_weight * action_loss

        # imitation_loss = F.cross_entropy(logits, acts)

        return imitation_loss


    def update_imitation_network():
        # print(f"hello")
        path_model_optimizer.zero_grad()

        data = path_planning_buf.get()
        path_planning_buf.reset()

        imitation_loss = compute_imitation_loss(data)
        logger.store(Loss=imitation_loss.item())

        net_scalar.scale(imitation_loss).backward()
        net_scalar.unscale_(path_model_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(path_model.parameters(), 0.5)
        net_scalar.step(path_model_optimizer)
        net_scalar.update()


    for epoch in range(epochs):
        env.reset()

        # train_w_baseline(env, baseline_env, route_model, baseline_model, route_planning_buf, baseline_route_planning_buf, logger, steps_per_epoch)
        demon_probs = np.random.rand()

        if demon_probs < 1.0:
            # # print(f"Training with expertise at epoch: {epoch}")
            train_path_planning_w_expertise(env, path_planning_buf, logger, steps_per_epoch)
            update_imitation_network()

        # what do I need to do here?

        #820, 590, 430, 390, 200, 80, 60, 20, 10

        # if epoch % evaluate_model == 0 and epoch > 0:
        #     testing_episodes = 100
        #     total_rew_online, total_rew_baseline = evaluate_model_performance(env, baseline_env, path_model, baseline_model, testing_episodes)
        #     print(f"Online model performance: {total_rew_online / testing_episodes} Baseline model performance: {total_rew_baseline / testing_episodes}")
        #     if total_rew_online < total_rew_baseline:
        #         baseline_model.load_state_dict(path_model.state_dict())
        #         print(f"updated baseline model with online model at epoch: {epoch}")
        #     else:
        #         path_model.load_state_dict(baseline_model.state_dict())
        #         # baseline_path_optimizer.load_state_dict(route_path_optimizer.state_dict())

        logger.log_tabular('Epoch', epoch)
        # logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('OptPathLength', average_only=True)
        logger.log_tabular('Loss', average_only=True)
        logger.dump_tabular()

    torch.save(path_model.state_dict(), 'imitation_model8.pth')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()

#   mpi_fork(args.cpu)  # run parallel code with mpi

    # from spinup.utils.run_utils import setup_logger_kwargs
    from algos_scratch_pad.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    il_path_planning(lambda: gridworld_env(), steps_per_epoch=500)