"""Launch file for the discrete (dc) and continuous (cc) PPO algorithm with TD updates

This script instantiate the gym environment, the agent, and start the training
"""

import argparse
import os

import gym
import yaml
import safety_gym
from env.make_env import make_env


# from agent import PPO
from agnt import PPO
from utils.tracker import Tracker

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()

if not cfg['setup']['use_gpu']:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['PYTHONHASHSEED'] = str(seed)

parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, help='Gym env', default=cfg['train']['name'])
parser.add_argument('-epochs', type=int, help='Epochs', default=cfg['train']['n_episodes'])
parser.add_argument('-stepmax', type=int, help='Max step', default=cfg['train']['max_step'])
parser.add_argument('-verbose', type=int, help='Save stats freq', default=cfg['train']['verbose'])
parser.add_argument('-std', type=float, help='σ for Gauss noise', default=cfg['agent']['std'])
parser.add_argument('-std_scale', type=float, help='σ scaling', default=cfg['agent']['std_scale'])


def main(params):
    config = vars(parser.parse_args())

    # env = gym.make(config['env'])
    env = make_env(config['env'])
    env.seed(seed)

    
    agent = PPO(env, cfg['agent'])
    tag = params['tag']

    # Initiate the tracker for stats
    tracker = Tracker(
        config['env'], #env.unwrapped.spec.id,
        tag,
        seed,
        cfg['agent'], 
        ['Epoch', 'Ep_Reward', 'Cost']
    )

    # Train the agent
    agent.train(
        tracker,
        n_episodes=config['epochs'], 
        n_step=config['stepmax'],
        verbose=config['verbose'],
        params=cfg['agent'],
        hyperp=config
    )

if __name__ == "__main__":
    main(cfg)
