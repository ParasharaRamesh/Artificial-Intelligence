# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
import tensorflow as tf
import tensorflow.feature_column as fc
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os.path



EPISODES = 1000
batch_size = 1#32

class DQNAgent:
    def __init__(self, state_size, action_size,gamma=0.5):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.describe()
        self.model = self._build_model()

    def describe(self):
        print("state_size:",self.state_size,"action_size:",self.action_size)

    # def input_fn():
    #     batch = random.sample(self.memory, batch_size)
    #     x=[x[0] for x in batch]
    #     y=[x[3] for x in batch]
    #     return {"inp":x},y


    def fit(self,estimator,x,y):
        def input_fn():
            print("x and y are of of types-->",type(x),type(y))
            return {"inp":x},y
        estimator.train(input_fn = lambda x,y :input_fn,steps=1)
        return estimator

    def predict(self,estimator,x):
        def input_fn():
            return {"inp":x}
        value = list(estimator.predict(input_fn=input_fn))
        print("generator returns ",value)
        return value
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        feature_cols=[fc.numeric_column("inp", shape=[self.state_size])]
        hidden_units=[48,24,12]
        estimator = tf.estimator.DNNRegressor(feature_columns=feature_cols,hidden_units = hidden_units,activation_fn=tf.nn.relu,model_dir="./model_checkpoints")
        return estimator  



        # model = Sequential()
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse',
        #             optimizer=Adam(lr=self.learning_rate))
        # return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        print("inside act")
        if np.random.rand() <= self.epsilon:
            print("inside")
            return random.randrange(self.action_size)
        act_values = self.predict(self.model,state)
        # print("taking a predited action",act_values,len(act_values))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.predict(self.model,next_state)[0]))
            target_f = self.predict(self.model,state)
            target_f[0][action] = target
            self.fit(self.model,state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    gamma=float(input("enter the gamma value"))
    agent = DQNAgent(state_size, action_size,gamma=gamma)
    # if os.path.exists("./save/cartpole-dqn-"+str(gamma)+".h5"):
    #     agent.load("./save/cartpole-dqn-"+str(gamma)+".h5")
    done = False


    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state)
            # print("action taken is ",action)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn-"+str(gamma)+".h5")
