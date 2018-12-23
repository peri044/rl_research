[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Navigation 
This project is about collecting as many yellow bananas as possible. A reward of +1 is awarded for a yellow banana and -1 for a blue banana.
State size is 37 (velocity of the agent and ray perception of the objects around it). 
Action size is 4 (left, right, forward, backward). 


Clone [DRLND](https://github.com/udacity/deep-reinforcement-learning/) to begin the project. Checkout the p1_navigation folder for more details.

![Trained Agent][image1]
## Files 
* Navigation.ipynb - Jupyter notebook to run the project 
* Navigation.py - Python script to run the project
* model.py - model file which has the architecture for the Q-network
* dqn_agent.py - dqn_agent class which handles the step updates, experience replay and fixed Q-target techniques for the DQN agent.
* checkpoint.pth - Model checkpoint file which has the weights of the Q-network. 

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

You can visualize the training on Windows by downloading the environment [Banana env for Windows 64-bit](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Instructions to run the project
```
python Navigation.py
```
## Architecture

The DQN agent in this repo consists of three fully connected layers with Relu activations. The architecture is 37 (state_size) -> 64 -> 32 -> 4 (action_size)

Hyper parameters are as follows:
* BUFFER_SIZE = int(1e5)  # replay buffer size
* BATCH_SIZE = 64         # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 1e-3              # for soft update of target parameters
* LR = 5e-4               # learning rate 
* UPDATE_EVERY = 4        # how often to update the network


## Rewards plot
The game was solved in 419 episodes with the average reward of 13.09.

![Alt text](rewards.png?raw=true "Title")

## Future work
* Implement Double DQN and prioritized experience replay techniques.
* Analyze the effect of hyperparameters and architectures on DQN agent learning.


