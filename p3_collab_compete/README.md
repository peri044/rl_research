[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Collaboration and Competition
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.


Clone [DRLND](https://github.com/udacity/deep-reinforcement-learning/) to begin the project. Checkout the p3_collab-compet folder for more details.

![Trained Agent][image1]
## Files 
* Tennis.ipynb - Jupyter notebook to run the project 
* Tennis.py - Python script to run the project
* model.py - model file which has the architectures for actor and critic networks.
* ddpg_agent.py - ddpg_agent which handles the step updates, experience replay and fixed Q-target techniques for the actor and critic model.
* checkpoint_actor.pth - Model checkpoint file which has the weights of the actor network. 
* checkpoint_critic.pth - Model checkpoint file which has the weights of the critic network. 

## Dependencies

For Windows, assuming you have Python 3.6 and pip installed, these are the additional packages I had to install

```
pip install matplotlib
pip install torchvision
pip install google
pip install gprcio
``` 
Set python path to unity agents
```
set PYTHONPATH=%PYTHONPATH%;<path to deep-reinforcement-learning folder>\python
```

You can visualize the training on Windows by downloading the environment [Tennis env for Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Instructions to run the project
```
python Tennis.py
```
## Algorithm

We use Deep Deterministic Policy Gradient (DDPG) algorithm to solve this environment. Two agents use their own actor/critic network and store their experiences in a shared replay buffer.

The architecture of the DDPG agent is as shown 

![Alt text](arch.png?raw=true "Title")

Hyper parameters are as follows:
* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 128        # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 2e-4         # learning rate of the actor 
* LR_CRITIC = 2e-4        # learning rate of the critic
* WEIGHT_DECAY = 0        # L2 weight decay

## Results
The game was solved in 2279 episodes with the average reward of 0.51. The rewards plot looks as follows

![Alt text](rewards_marl.png?raw=true "Title")

## Future work
* DDPG with different actor/critic networks for two agents convergence is very slow and unstable. Implementing Prioritized experience replay and other popular algorithms might boost performance.
* Parallelize the existing code by taking advantage of multicore environments and running multiple agents for faster learning. Follow implementation of MADDPG lab to improve this code.


