#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:54:38 2019

@author: edward
"""
import time, logging

import torch

from arguments import parse_a2c_args
from multi_env import MultiEnv
from models import CNNPolicy
from a2c_agent import A2CAgent
from utils import initialize_logging


def train():
    args = parse_a2c_args()
    output_dir = initialize_logging(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    num_updates = int(args.num_frames) // args.num_steps // args.num_environments
    print('Total number of iterations: ', num_updates)
    # Create the train and test environments with Multiple processes
    train_envs = MultiEnv(args.simulator, args.num_environments, args, is_train=True)
    test_envs = MultiEnv(args.simulator, args.num_environments, args, is_train=False)
    
    obs_shape = train_envs.obs_shape
    print('Observation Shape: ', obs_shape)
    
    # The agent's policy network and training algorithm A2C
    policy = CNNPolicy(obs_shape, args).to(device)
    agent = A2CAgent(policy, 
                     args.hidden_size,
                     value_weight=args.value_loss_coef, 
                     entropy_weight=args.entropy_coef, 
                     num_steps=args.num_steps, 
                     num_parallel=args.num_environments,
                     gamma=args.gamma,
                     lr=args.learning_rate,
                     opt_alpha=args.alpha,
                     opt_momentum=args.momentum,
                     max_grad_norm=args.max_grad_norm,
                     use_rot=args.use_rot,
                     use_gray=args.use_gray)
    
    start_j = 0
    if args.reload_model:
        checkpoint_idx = args.reload_model.split(',')[1]
        checkpoint_filename = '../'+args.reload_model.split(',')[0]#'{}models/checkpoint_{}.pth.tar'.format(output_dir, checkpoint_idx)        
        agent.load_model(checkpoint_filename)
        start_j = (int(checkpoint_idx) // args.num_steps // args.num_environments) + 1
    print('Total number of iterations: ', num_updates)
    print('Starting iteration: ', start_j)
    print('Number of steps per iteration: ', args.num_steps)
    obs = train_envs.reset()
    start = time.time()
    print('Starting training...')

    
    for j in range(start_j, num_updates):
        print('Iteration: ', j)
        if not args.skip_eval and j % args.eval_freq == 0:
            total_num_steps = (j + 1) * args.num_environments * args.num_steps
            mean_rewards, game_times = agent.evaluate(test_envs, j, total_num_steps)
            logging.info(mean_rewards)
            logging.info(game_times)
        
        for step in range(args.num_steps):
            
            action = agent.get_action(obs, step)
            obs, reward, done, info = train_envs.step(action)
            agent.add_rewards_masks(reward, done, step)
          
        report = agent.update(obs)
        
        if j % 100 == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_environments * args.num_steps
            save_num_steps = (start_j) * args.num_environments * args.num_steps
            FPS = int((total_num_steps - save_num_steps) / (end - start)),
            
            logging.info(report.format(j, total_num_steps, FPS))  
        
        if j % 100==0:#args.model_save_rate == 0:
            print('Saving model...')
            total_num_steps = (j + 1) * args.num_environments * args.num_steps
            #agent.save_policy(total_num_steps, args, output_dir)
            agent.custom_save()
            for tta_method in agent.tta_methods:
                tta_method.save()
    
    agent.custom_save()
    for tta_method in agent.tta_methods:
        tta_method.save()
    # cancel the env processes    
    train_envs.cancel()
    test_envs.cancel()
    
    
if __name__ == '__main__':
    train()
    
       