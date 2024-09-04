import numpy as np
import env

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, 
                 discount_factor=0.9, exploration_rate=1.0, 
                 exploration_decay=0.99):
        self.env = env
        self.q_table = np.zeros((env.size[0] * env.size[1], 4))  # 初始化 Q 表格
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def state_to_index(self, state):
        return state[0] * self.env.size[1] + state[1]
    
    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(4)  # 随机选择动作
        else:
            state_index = self.state_to_index(state)
            return np.argmax(self.q_table[state_index])  # 选择具有最高 Q 值的动作
    
    def update_q_table(self, state, action, reward, next_state):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        
        best_next_action = np.argmax(self.q_table[next_state_index])
        td_target = reward + self.discount_factor * self.q_table[next_state_index, best_next_action]
        td_error = td_target - self.q_table[state_index, action]
        
        self.q_table[state_index, action] += self.learning_rate * td_error
    
    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay
    
    # print table
    def print_table(self):
        print(self.q_table)

if __name__ == '__main__':
    # 初始化迷宫环境和 Q-learning 代理
    env = env.MazeEnv()
    agent = QLearningAgent(env)

    # 训练 Q-learning 代理
    num_episodes = 500

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        # print the table
        agent.print_table()
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
        
        agent.decay_exploration()

    # 测试 Q-learning 代理
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        env.render()