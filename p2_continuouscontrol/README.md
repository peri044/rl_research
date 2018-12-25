[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Continuous control 
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.


Clone [DRLND](https://github.com/udacity/deep-reinforcement-learning/) to begin the project. Checkout the p1_navigation folder for more details.

![Trained Agent][image1]
## Files 
* Continuous_Control.ipynb - Jupyter notebook to run the project 
* Continuous_Control.py - Python script to run the project
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

You can visualize the training on Windows by downloading the environment [Reacher env for Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## Instructions to run the project
```
python Continuous_Control.py
```
## Architecture

The architecture of the DDPG agent is as shown 

![Alt text](arch.png?raw=true "Title")

Hyper parameters are as follows:
* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 128        # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR_ACTOR = 1e-3         # learning rate of the actor 
* LR_CRITIC = 2e-4        # learning rate of the critic
* WEIGHT_DECAY = 0        # L2 weight decay



## Results
The game was solved in 216 episodes with the average reward of 30.08. The rewards plot looks as follows

![Alt text](rewards.png?raw=true "Title")

## Future work
* Analyze as to why batchnorm led to significant improvement.
* As mentioned in the project instructions, I would like to implement TRPO, TNPG, D4PG methods and compare all of them.


