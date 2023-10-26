import gymnasium as gym

def run():
    env = gym.make('MountainCar-v0', render_mode = "human")
    observation, _ = env.reset()

    for _ in range(1000):
        #action = env.action_space.sample()
        action = 2 if observation[1] > 0 else 0
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()

run()
