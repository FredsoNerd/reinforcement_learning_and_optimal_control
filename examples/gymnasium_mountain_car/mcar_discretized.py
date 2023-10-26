# reference: https://www.youtube.com/watch?v=_SWnNhM5w-g

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pickle

def run(episodes=100, train = True, render = True, epsilon_0=0.1, alpha=0.9, gamma=0.9, savefile="q_table.pkl"):
    # initializes the environment
    env = gym.make('MountainCar-v0', render_mode = "human" if render else None)

    # space discretization
    space_pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    space_vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
    
    # initializes Q table
    if train:
        q_table = np.zeros((len(space_pos), len(space_vel), env.action_space.n))
    else:
        with open(savefile, "br") as f:
            q_table = pickle.load(f)

    # saves total rewards
    rewards = np.zeros((episodes,))

    # simulates the episodes
    for episode in tqdm.tqdm(range(episodes)):
        epsilon = epsilon_0/(episode + 1)
        # resets environment
        observation, _ = env.reset()
        state_pos = np.digitize(observation[0], space_pos)
        state_vel = np.digitize(observation[1], space_vel)
        action = np.argmax(q_table[state_pos, state_vel, :])

        # single epsode simulation
        while True:
            # epsilon-greedy action
            if np.random.random() < epsilon and train:
                action = env.action_space.sample()
            
            # runs single action step
            observation, reward, terminated, truncated, _ = env.step(action)
            new_state_pos = np.digitize(observation[0], space_pos)
            new_state_vel = np.digitize(observation[1], space_vel)
            new_action = np.argmax(q_table[new_state_pos, new_state_vel, :])
            
            # updates Q table values
            Q_t0 = q_table[state_pos, state_vel, action]
            Q_t1 = q_table[new_state_pos, new_state_vel, new_action]
            q_table[state_pos, state_vel, action] += alpha*(reward + gamma*Q_t1 - Q_t0)

            # updates sistem states
            action = new_action
            state_pos = new_state_pos
            state_vel = new_state_vel
            rewards[episode] += reward

            # ends if done simulation
            if terminated:
                break

    # saves some usefull info 
    if train:
        # plot rewards
        plt.plot(rewards)
        plt.savefig("mcar_discretized_rewards")
        # plot Q table
        x,y = np.meshgrid(space_pos, space_vel)
        data = np.zeros((len(space_pos), len(space_vel)))
        for pos in range(len(space_pos)):
            for vel in range(len(space_vel)):
                data[pos,vel] = np.argmax(q_table[pos,vel,:])
        # _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.plot_surface(x, y, data, linewidth=0, antialiased=False)
        plt.imshow(data)
        plt.savefig("mcar_discretized_q_table")

        # saves trained table
        with open(savefile, "wb") as f:
            pickle.dump(q_table, f)

run(episodes=1000, train=True, render=False)
run(episodes=1, train=False, render=True)