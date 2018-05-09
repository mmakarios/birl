"""
FrozenLake using Q Learning algorithm.
https://gym.openai.com/envs/FrozenLake-v0/
"""

import numpy as np
import gym
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)


def setup_q_matrix(state_size, action_size):
    """ Initialize Q Learning matrix with zeros. """
    return np.zeros([state_size, action_size])


def main():
    """ Q learning """

    # env = gym.make('FrozenLakeNotSlippery-v0')
    env = gym.make('FrozenLake-v0')

    q_table = setup_q_matrix(env.observation_space.n, env.action_space.n)

    # Learning parameters
    learning_rate = 0.8
    discount = 0.95
    num_episodes = 2000

    # Create list to contain total rewards and steps per episodes
    r_list = []

    for i in range(num_episodes):
        epsilon = 1./((i//100)+1)  # decay e-greedy

        state = env.reset()
        r_all = 0
        done = False

        while not done:

            if np.random.rand(1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

            state_new, reward, done, _ = env.step(action)
            # env.render()
            new_value = learning_rate * \
                (reward + discount *
                 np.max(q_table[state_new, :]) - q_table[state, action])

            q_table[state, action] += new_value

            r_all += reward
            state = state_new

        r_list.append(r_all)

    # env.render()
    print("Success rate: " + str(sum(r_list)/num_episodes))

    print("Q:")
    print(q_table)


if __name__ == "__main__":
    main()
