import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    Update Q-table using Q-learning rule.
    """
    best_next_action = np.max(Q[sprime])  
    Q[s, a] += alpha * (r + gamma * best_next_action - Q[s, a])  
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    Choose an action using epsilon-greedy policy.
    """
    if np.random.rand() < epsilone:  # Explore
        return np.random.randint(Q.shape[1])
    else:  # Exploit
        return np.argmax(Q[s])


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")  

    Q = np.zeros([env.observation_space.n, env.action_space.n])  

 
    alpha = 0.1  
    gamma = 0.8  
    epsilon = 0.1  
    n_epochs = 1500  
    max_itr_per_epoch = 500 
    rewards = []

    
    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()  

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)  
            Sprime, R, done, _, _ = env.step(A)  
            r += R 

            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)

            S = Sprime  

            if done:  
                break

        print(f"Episode #{e}: Total Reward = {r}")
        rewards.append(r)

    print("Average Reward:", np.mean(rewards))


    for i in range(4): 
        env.reset()
        print(env.render()) 
        time.sleep(2)

    # Close the environment
    env.close()

    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.title("Q-Learning Training Progress")
    plt.show()
