# Navigation 
This project is about collecting as many yellow bananas as possible. A reward of +1 is awarded for a yellow banana and -1 for a blue banana.
State size is 37 (velocity of the agent and ray perception of the objects around it) 
Action size is 4 (left, right, forward, backward)

## Files 
* Navigation.ipynb - Jupyter notebook to run the project 
* Navigation.py - Python script to run the project
* model.py - model file which has the architecture for the Q-network
* dqn_agent.py - dqn_agent class which handles the step updates, experience replay and fixed Q-target techniques for the DQN agent.

## Instructions to run the project
```
python Navigation.py
```

## Rewards plot
![Alt text](rewards.png?raw=true "Title")

## Future work
* Implement Double DQN and prioritized experience replay techniques.
* Analyze the effect of hyperparameters and architectures on DQN agent learning.


