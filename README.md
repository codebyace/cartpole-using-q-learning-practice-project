Cartpole using Q-learning 

	This project rests on the idea of practicing Reinforcement Learning with q-learning techniques. 

Description: 

	The CartPole problem is a classic control task in Reinforcement Learning. The environment consists of a cart that can move left or right, with a pole attached to it. The goal is to balance the pole upright for as long as possible by moving the cart.

	This project uses Q-Learning, a model-free RL algorithm, to train an agent to solve the problem. The agent learns a policy (Q-table) that maps states to actions, maximizing the cumulative reward.

 Requirements:
 
 	 Libraries: 
	 - `gym` or `gymnasium` (for the CartPole environment)
	 - `numpy` (for numerical computations)
	 - `matplotlib` (for plotting the results)
  


Learning Outcome: 

	A plot is generated to visualize the agent's learning progress. The x-axis represents the episodes, and the y-axis represents the total reward per episode. The plot shows a smooth upward trend, indicating that the agent is learning effectively.

 Expected Output:

	Episode 0: Total Reward = 15.0
	Episode 50: Total Reward = 46.0
	Episode 100: Total Reward = 26.0
	Episode 150: Total Reward = 18.0
	Episode 200: Total Reward = 42.0
	Episode 250: Total Reward = 54.0
	Episode 300: Total Reward = 43.0
	Episode 350: Total Reward = 35.0
	Episode 400: Total Reward = 53.0
	Episode 450: Total Reward = 32.0


 

