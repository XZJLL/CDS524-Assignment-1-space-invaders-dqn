CDS524 Assignment 1
1. Game Design
The game utilized in this assignment is Space Invaders from Atari, accessed through the Gym library. The decision to use the Gym library instead of developing the game using the Pygame library was driven by persistent incompatibility issues encountered during the implementation of Q-learning. Specifically, the program faced conflicts among libraries such as TensorFlow, Keras, Seaborn, Pygame, NumPy, Matplotlib, and Pytorch, as well as issues with the CUDA version. Despite multiple attempts to resolve these issues by reinstalling libraries and rewriting the program, the incompatibilities could not be resolved. Consequently, the decision was made to utilize the Space Invaders game available within the Gym library.
1.1 Game Objective and Rules
Objective:
The primary objective in Space Invaders is to defend the Earth from an invading alien fleet. The player controls a laser cannon that moves horizontally at the bottom of the screen. The goal is to shoot and destroy all invading aliens before they reach the bottom of the screen or destroy the player's cannon.
Rules:
	The player/agent can move the cannon left and right to shoot projectiles upward at the alien spaceships.
	The aliens move horizontally and gradually descend toward the bottom of the screen.
	The aliens also shoot projectiles downward toward the player's cannon.
	The player earns points for each alien destroyed.
	The player has a limited number of lives (typically 3). Losing all lives results in a game over.
	The game ends when all aliens are destroyed or when the aliens reach the bottom of the screen.
1.2 State Space and Action Space
State Space:
The state space in Space Invaders is defined by the positions of the player's cannon, the positions of the aliens, and the positions of the projectiles.
	Player Cannon Position: The horizontal position of the player's cannon.
	Alien Positions: The positions of all aliens on the screen.
	Projectile Positions: The positions of all active projectiles (both player and alien).
Action Space:
The action space is discrete and consists of the following possible actions:
	Move Left: Move the cannon to the left.
	Move Right: Move the cannon to the right.
	Shoot Projectile: Fire a projectile upward.
	No Action: The cannon remains in its current position.
1.3 Positive and Negative Rewards
Reward Function:
The reward function in Space Invaders is designed to encourage the player to destroy aliens while avoiding being hit by alien projectiles or allowing the aliens to reach the bottom of the screen.
	Positive Rewards:
Destroying an Alien: The player receives a positive reward (e.g., +10 points) for each alien destroyed.
	Negative Rewards:
Losing a Life: The player receives a large negative reward (e.g., -100 points) if the player's cannon is hit by an alien projectile, resulting in the loss of a life.
Aliens Reaching the Bottom: The player receives a large negative reward (e.g., -100 points) if any alien reaches the bottom of the screen.
	Neutral Rewards:
No Action: The player receives a neutral reward (e.g., 0 points) for time steps where no significant event occurs.

2. Implementation of Q-Learning in the Code
2.1 Initialization
	Environment Initialization: The game environment is created using gym.make('SpaceInvaders-v0').
	Agent Initialization: A DQNAgent object is created in agent.py, initializing both the local Q-network and the target Q-network, and setting hyperparameters 
2.2 Replay Buffer
Used for storing the agent's information (state, action, reward, next state, done) from interactions with the environment.
	A fixed-size circular buffer is implemented using deque.
	After each interaction, the experience (state, action, reward, next_state, done) is stored in the buffer.
	During training, a mini-batch of experiences is randomly sampled from the buffer to update the Q-network.
2.3 Q-Network (DQN)
Approximates the Q-function, taking the state s as input and outputting Q-values for all possible actions.
	A convolutional neural network (CNN) is used to process image-based states (game frames).
	The network architecture includes convolutional layers followed by fully connected layers, ultimately outputting Q-values for each action.
	The local network (qnetwork_local) is used for action selection and computing current Q-values.
	The target network (qnetwork_target) is used to compute target Q-values, stabilizing training.
	Action Selection (Epsilon-Greedy Policy)
To Balance exploration (random actions) and exploitation (selecting the optimal action).
	With probability ε, a random action is selected (exploration).
	With probability 1− ε, the action with the highest Q-value is selected (exploitation).
	The exploration rate  ε decays over time, starting from ε_start and decreasing to ε_end.
2.5 Training Process
	Environment Initialization: Reset the environment and obtain the initial state s.
	Action Selection: Select an action a using the epsilon-greedy policy.
	Action Execution: Execute the action a in the environment, observe the reward r, next state s′, and whether the episode is done.
	Store Experience: Store the experience in the replay buffer.
	Sample Experience: Randomly sample a mini-batch of experiences from the buffer.
	Compute Target Q-Values: Compute target Q-value for non-terminal states(  r+〖γmax〗_a' Q_target (s',a')) and terminal states (r).
	Compute Current Q-Values: Use the local network to compute the current Q-values (Q_local (s',a')).
	Update Q-Network: Update the parameters of the local network by minimizing the mean squared error (MSE) between the current Q-values and the target Q-values.
	Update Target Network: Periodically copy the parameters of the local network to the target network.
	Repeat: Repeat the above steps until the maximum timestep is reached or the episode ends.
2.6 Loss Function and Optimization
	Loss Function: The mean squared error (MSE) is used to compute the difference between the current Q-values and the target Q-values:
	Optimizer: The Adam optimizer is used to update the parameters of the local network.
2.7 Training Loop
	Outer Loop: Iterates over each episode.
	Inner Loop: Iterates over each timestep until the episode ends or the maximum timestep is reached.
	Exploration Rate Decay: After each episode, the exploration rate ϵ decays by a factor of ϵdecay.
2.8 Model Saving and Logging
	Model Saving: The parameters of the local network and training data are saved periodically.
	Logging: Scores and completion timesteps for each episode are recorded for subsequent analysis and visualization.
3. Evaluation plot and game UI
 
 
Github link: XZJLL/CDS524-Assignment-1-space-invaders-dqn
