import numpy as np
import gym
import random


env = gym.make("FrozenLake-v0")

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size,action_size))

print(qtable)

total_episodes = 15000
learning_rate = 0.8
max_steps = 99
gamma = .95

#Exploration

epsilon = 1.00
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005  #exponential decay rate for exploration


#Q learning algorithm implementation

#List of rewards

rewards = []

#Keeps training for life or until its stopped

for episode in range(total_episodes):
    #Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        #pick an action 'a' in the current state 's'
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number is greater than epsilon --> exploit(take the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        #Else, do a random choice aka exploration
        else:
            action = env.action_space.sample()

        #Take the action (a) and observe the outcome state and rewards

        new_state, reward, done, info = env.step(action)

        #qtable[new_state,:] : all teh actions we can take from new state


        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward


        #new state is state

        state = new_state

        #if done(if we died) : end the episode

        if done == True:
            break

    #reduce epsilon

    epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
    rewards.append(total_rewards)

print('Score over time:', str(sum(rewards)/total_episodes))
print(qtable)


#play and train

# env.reset()
#
# for episode in range(5):
#     state = env.reset()
#     step = 0
#     done = False
#     print('*****************************')
#     print('Episode', episode)
#
#     for step in range(max_steps):
#         #take an action (index) that has the most expected future rewards
#         action = np.argmax(qtable[state,:])
#
#         new_state, reward, done
























