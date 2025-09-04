import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make('CartPole-v1')

bins = [
    np.linspace(-4.8, 4.8, 10),
    np.linspace(-4,4,10),
    np.linspace(-0.418,0.418,10),
    np.linspace(-4,4,10)
]

def discretize_state(state):
    state_indices = []
    for i in range(len(state)):
        state_indices.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_indices)

state_space_size = [10] * 4
action_space_size = env.action_space.n
q_table = np.zeros(state_space_size + [action_space_size])

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 1000
max_steps = 200

rewards = []

for episodes in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = discretize_state(state)
    total_reward = 0
    done = False


    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done,truncated, _ = env.step(action)
        done = done or truncated
        next_state = discretize_state(next_state)
        
        q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)
    
    if (episodes + 1) % 100 == 0:
        avg_reward = np.mean(rewards[-100:])
        print(f'Episode {episodes + 1}, Average Reward: {avg_reward}, Epsilon: {epsilon:.3f}')

env.close()


plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning on CartPole (Training Process)')
plt.show()

np.save('q_table_cartpole.npy', q_table)


env = gym.make("CartPole-v1",render_mode="human")
state = env.reset()
if isinstance(state, tuple):
    state = state[0]
state = discretize_state(state)
done = False
total_reward = 0
while not done:
    action = np.argmax(q_table[state])
    state,reward,done, truncated, _ = env.step(action)
    done = done or truncated
    state = discretize_state(state)
    total_reward += reward
    env.render()

env.close()
print(f"Test Episode Reward: {total_reward}")    
