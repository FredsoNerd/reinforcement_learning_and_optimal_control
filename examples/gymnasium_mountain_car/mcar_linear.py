import gymnasium as gym
import numpy as np
import tqdm

def run(n, w = np.array([0,0,0,0]), epsilon=0.1, alpha=0.9, gamma=0.9):
    env = gym.make('MountainCar-v0', render_mode="human",max_episode_steps=n)
    #env = gym.make('MountainCar-v0', max_episode_steps=n)
    action_t = 1
    observation_t, _ = env.reset()

    for _ in tqdm.tqdm(range(n)):
        observation_t1, reward_t1, terminated, truncated, _ = env.step(action_t)
        action_t1 = epsilon_greed_action(env, observation_t1, w, epsilon)

        if terminated or truncated:
            print("Terminated game." if terminated else "Truncated game.")
            observation_t, _ = env.reset()
            action_t = 1

        # updates w under policy
        print(w)
        w = update(observation_t, action_t, observation_t1, action_t1, reward_t1, w, alpha, gamma)

        # updates old states
        action_t = action_t1
        observation_t = observation_t1

    env.close()
    
def epsilon_greed_action(env, observation, w, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    
    action_values = []
    for action in [0,1,2]:
        action_values.append(value(observation, action, w))
    return np.argmax(action_values)

def value(observation, action, w):
    return np.inner(w, np.array([1, *observation, action]))

def update(observation_t, action_t, observation_t1, action_t1, reward_t1, w, alpha, gamma):
    x_t = np.array([1, *observation_t, action_t])
    x_t1 = np.array([1, *observation_t1, action_t1])
    return w + alpha * (reward_t1 + gamma*np.inner(w,x_t1) - np.inner(w,x_t)) * x_t1
    
run(9000)
