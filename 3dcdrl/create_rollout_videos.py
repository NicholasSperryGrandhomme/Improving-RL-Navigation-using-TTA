#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:19:33 2018

@author: anonymous
"""
import os
import torch
import numpy as np
from arguments import parse_a2c_args
from multi_env import MultiEnv
from models import CNNPolicy
from a2c_agent import *
from utils import initialize_logging
from doom_environment import DoomEnvironment

import cv2
import pickle
 
from moviepy.editor import ImageSequenceClip
from PIL import Image


def batch_from_obs(obs, batch_size=32):
	"""Converts an obs (C,H,W) to a batch (B,C,H,W) of given size"""
	if isinstance(obs, torch.Tensor):
		if len(obs.shape)==3:
			obs = obs.unsqueeze(0)
		return obs.repeat(batch_size, 1, 1, 1)

	if len(obs.shape)==3:
		obs = np.expand_dims(obs, axis=0)
	return np.repeat(obs, repeats=batch_size, axis=0)

def make_movie(policy, env, filename, args, n_runs=50, use_tta=False,
               use_rot=False, use_gray=False, name='', view=None, txt_pos=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time_taken = []
    losses = []
    for i in range(n_runs):
        if use_tta:
            if use_rot:
                path = 'policy.pth.tar'
            else:
                path='policy_TTA_GRAY.pth.tar'

            checkpoint = torch.load('tta_models/'+path, map_location=device)
            policy = CNNPolicy((3,64,112), args).to(device)
            policy.load_state_dict(checkpoint['model'])
            policy.eval()

            if use_rot or use_gray:
                tta_agent = TTAAgent(use_rot=use_rot,obs_shape=(3,64,112), hidden_size=128)
                tta_agent.load()
                tta_agent.copy_conv_weights(policy.conv_head)
        state = torch.zeros(1, args.hidden_size)
        mask = torch.ones(1,1)
        obss = []
        pos_list = []
            
        obs = env.reset().astype(np.float32)
        done = False

        while not done:
            #Gamma correction
            obs = 255*np.power(obs/255.0, args.gamma_val)
            #Inverse image
            if args.inverse:
                obs = 255 - obs
            obss.append(obs)
            with torch.no_grad():
                result = policy(torch.from_numpy(obs).unsqueeze(0), state, mask)

            action = result['actions']
            state = result['states']

            obs, reward, done, _ = env.step(action.item())
            
            if view != None and txt_pos != None:
                x, y, _ = env.get_player_position()
                pos_list.append([x, y])
            
            if use_tta and (use_rot or use_gray):
                batch_next_obs = batch_from_obs(torch.Tensor(obs).to(device), batch_size=16)
                # Adapt using rotation prediction
                losses.append(tta_agent.update_tta(batch_next_obs))          

            obs = obs.astype(np.float32)
        time_taken.append(len(obss)/int(30/args.frame_skip))

    if use_tta:
        if use_rot:
            tta_type='rotation'
        elif use_gray:
            tta_type='grayscale'
        else:
            tta_type = 'tta_OFF'
    else:
        tta_type='baseline'
    pickle.dump(time_taken, open(f'TTA_videos/{tta_type}/{name}.pkl', 'wb'))
    
    print(len(obss))       
    print(f'Average time taken: {np.mean(time_taken):.2f}s')   
    print(f'TTA mean loss: {np.mean(losses):.3f}')   
    
    observations = [o.transpose(1,2,0) for o in obss]
    clip = ImageSequenceClip(observations, fps=int(30/args.frame_skip))
    clip.write_videofile(filename)
    
    if view != None and txt_pos != None:
        # saving the view of the agent and the position
        # of the last run
        pos_txt = open(txt_pos, "w+")
        for p in pos_list:
            pos_txt.write("%d,%d\r\n" % (p[0], p[1]))
        pos_txt.close()
        
        for c, o in enumerate(observations):
            im = Image.fromarray(o)
            fig_name = view_name + str(c) + ".png"
            im.save(view_path + fig_name)

def evaluate_saved_model():
    args = parse_a2c_args()
    USE_TTA = args.use_tta
    USE_ROT = args.use_rot
    USE_GRAY = args.use_gray
    exp_name = args.experiment_name
    SV_VW_POS = args.save_view_position

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape = (3, args.screen_height, args.screen_width)
    policy = CNNPolicy(obs_shape, args).to(device)
    
    #Load Agent
    if USE_TTA:
        if USE_ROT:
            path = 'policy.pth.tar'
        else:
            path='policy_TTA_GRAY.pth.tar'
        checkpoint = torch.load('tta_models/'+path, map_location=device)
    else:
        path = 'saved_models/labyrinth_9_checkpoint_0198658048.pth.tar'
        checkpoint = torch.load(path, map_location=device)
    policy.load_state_dict(checkpoint['model'])
    policy.eval()

    assert args.model_checkpoint, 'No model checkpoint found'
    assert os.path.isfile(args.model_checkpoint), 'The model could not be loaded'
 
    for i in range(args.num_mazes_test):
        #env = MultiEnv(args.simulator, args.num_environments, args, is_train=True)
        env = DoomEnvironment(args, idx=i, is_train=True, use_shaping=args.use_shaping, fixed_scenario=True)
        name='False'
        if USE_TTA:
            if USE_ROT:
                tta_type='rotation'
            elif USE_GRAY:
                tta_type='grayscale'
            else:
                tta_type = 'tta_OFF'
        else:
            tta_type = 'baseline'
        print(tta_type)
        if SV_VW_POS:
            view_name = f'map_creation/TTA_view/{tta_type}/'
            txt_pos_track_name = f'map_creation/TTA_position/{tta_type}/{exp_name}.txt'
            print('Saving view and positions of the agent.')
        else:
            view_name = None
            txt_pos_track_name = None
        movie_name = f'TTA_videos/{tta_type}/{exp_name}.mp4'
        print('Creating movie {}'.format(movie_name))
        make_movie(policy, env, movie_name, args, n_runs=100, 
                   use_tta=USE_TTA, use_rot=USE_ROT, use_gray=USE_GRAY, name=exp_name, view=view_name, txt_pos=txt_pos_track_name)
        
        
if __name__ == '__main__':
    evaluate_saved_model()
    
    
