# Improving a RL Navigation Agent using Test Time Adaptation methods
This repository contains the code use to evaluate the Robustness of an A2C Navigation Agent and improving it using TTA methods. This project was done for the course *Visual Intelligence: Machines and Minds* as part of the coursework.

## Team members
* Oscar Lucas Blazquez
* Oliver Becker
* Nicholas Sperry

## Project Layout
The base code of this project (so the Agent we are trying to improve) was taken from [[here]](https://github.com/edbeeching/3d_control_deep_rl)[[Paper]](https://arxiv.org/abs/1904.01806). We implemented the TTA methods from the *Self-Supervised Policy Adaptation during Deployment* repository [[GitHub]](https://github.com/nicklashansen/policy-adaptation-during-deployment)[[Paper]](https://arxiv.org/abs/2007.04309).

## Installation

### Requirements
The code only seems to be able to run on Linux, as we had many issues trying to run it on Mac and Windows. This is likely due to the older ViZDoom version used. Other than that, it seemed to work on Google Colab fine.

### Instructions
1. Install [ViZDoom](https://github.com/mwydmuch/ViZDoom)
2. Clone this repository
3. Install the `requirements.txt` file

These should be all the requirements, but if not follow the instructions from the original repository [here](https://github.com/edbeeching/3d_control_deep_rl).

## Evaluating Agent Robustness
First follow these two steps:
1. Clone the repository
2. `cd` into `3dcdrl/`

There is also a Jupyter notebook (`3dcdrl/TTA Evalutation.ipynb`) with some similar explanations on how to run these tests.

### Baseline Evaluation
To evaluate the baseline, unchanged Agent run the following command:
        
       python create_rollout_videos.py --limit_actions \
       --scenario_dir  scenarios/custom_scenarios/labyrinth/9/test/ \
       --scenario original.cfg  --model_checkpoint \
       tta_models/policy.pth.tar \
       --multimaze --num_mazes_test 1 --num_environments 1 --num_actions 5 \
       --exp_name original
       
Here you can change the `--scenario` flag to any of the following:

* ***original.cfg*** - Original map
* ***mossy_walls.cfg*** - Walls have a mossy texture
* ***decreased_light.cfg*** - Map is darker
* ***increased_light.cfg*** - Map is brighter
* ***swapped_ceil_floor.cfg*** - Floor and Ceiling textures swapped, and map is brighter.

The command above also creates a video of the Agent navigating through the map as well as a pickle file containing all the times taken over
each run. To pick a name, use the `--exp_name` flag followed by whatever you want to name these files (e.g. `--exp_name original_baseline`. These files will be saved in the `3dcdrl/TTA_videos/baseline/` directory (for the baseline Agent).

You can also use the `--gamma_value` flag to set the amount of gamma correction, for exmaple `--gamma_value 1.5`. A higher gamma value results in darker images. You can also use the `--inverse` flag to create negative images that will be fed into the agent.

### TTA Evaluation
We implemented TTA with rotation prediction from the *Self-Supervised Policy Adaptation during Deployment* [paper](https://github.com/nicklashansen/policy-adaptation-during-deployment), as well as a similar method that uses TTA with grayscale prediction. To test one **OR** the other, you must use the `--use_tta` flag followed by `--use_rot` or `use_gray` to use TTA with rotation prediction or TTA with grayscale prediction respectively. Here are two examples.

**Evaluating on the original map using TTA with Rotation prediction**:

       python create_rollout_videos.py --limit_actions \
       --scenario_dir  scenarios/custom_scenarios/labyrinth/9/test/ \
       --model_checkpoint tta_models/policy.pth.tar \
       --multimaze --num_mazes_test 1 --num_environments 1 --num_actions 5 \
       --scenario original.cfg --use_tta --use_rot --exp_name rotation_original
       
The command above will then create a navigation video as well as the time taken for each run in the `3dcdrl/TTA_videos/rotation/` directory called `rotation_original.mp4` and `rotation_original.pkl` respectively.
       
**Evaluating on the mossy wall map using TTA with Grayscale prediction**:

       python create_rollout_videos.py --limit_actions \
       --scenario_dir  scenarios/custom_scenarios/labyrinth/9/test/ \
       --model_checkpoint tta_models/policy.pth.tar \
       --multimaze --num_mazes_test 1 --num_environments 1 --num_actions 5 \
       --scenario mossy_walls.cfg --use_tta --use_gray --exp_name grayscale_mossy

The command above will then create a navigation video as well as the time taken for each run in the `3dcdrl/TTA_videos/grayscale/` directory called `grayscale_mossy.mp4` and `grayscale_mossy.pkl` respectively.

### Ablation Experiment
By evaluating the Agent and just setting the `--use_tta` flag, the Agent trained using the grayscale TTA will be loaded, but TTA will be switched off during evaluation. This a sort of Ablation test since we are removing the TTA at test time, but using the Agent that was trained by it. For example:

        python create_rollout_videos.py --limit_actions \
       --scenario_dir  scenarios/custom_scenarios/labyrinth/9/test/ \
       --model_checkpoint tta_models/policy.pth.tar \
       --multimaze --num_mazes_test 1 --num_environments 1 --num_actions 5 \
       --scenario mossy_walls.cfg --use_tta --exp_name ablation_mossy
       
Will create the video and pickle file in the `3dcdrl/TTA_videos/tta_OFF/` directory
