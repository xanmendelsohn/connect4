from abc import abstractmethod
import os
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.models as Km
import tensorflow.keras.layers as Kl
import numpy as np
import time
import random as rand
import pickle

class QModelClass:

    def __init__(self, model_name, exploration_factor, alpha, configuration, model_architecture, early_stopping = True):
        self.configuration = configuration
        self.model_name = model_name
        self.model_architecture = model_architecture(self.model_name)
        self.model = self.load_model()
        self.epsilon = 0.1
        self.alpha = alpha #0.5
        self.gamma = 1
        self.exp_factor = exploration_factor
        self.early_stopping = early_stopping
        self.past_prev_state = np.zeros((6, 7))
        self.past_prev_move = None
        self.prev_state = np.zeros((6, 7))
        self.prev_move = None
        self.prev_reward = None
        self.prev_reward_opponent = None
        self.state = None
        self.move = None
        self.memory = []
        
    def reset_memory(self):
        self.past_prev_state = np.zeros((6, 7))
        self.past_prev_move = None
        self.prev_state = np.zeros((6, 7))
        self.prev_move = None
        self.state = None
        self.move = None
        self.memory = []
        self.count_memory = 0
        
    def load_model(self):
        s = 'DQNs/' + self.model_name + '.h5'
        model_file = Path(s)

        if model_file.is_file():
            # print('load model')
            model = Km.load_model(s)
            # print('load model: ' + s)
        else:
            model = self.model_architecture.create_model()
        return model

    @abstractmethod
    def create_model(self):
        pass
    
    def all_options(self, board_state):
        moves = np.where(board_state[0, :] == 0)[1]
        return moves
    
    def obs_to_board_state(self, observation):   
        board_state = np.matrix(np.array(observation.board).reshape(self.configuration["rows"],self.configuration["columns"]))
        if observation.mark==1:
            board_state[np.where(board_state == 2)]= -1
        else:
            board_state[np.where(board_state == 1)]= -1
            board_state[np.where(board_state == 2)]= 1

        return board_state
    
    #simulate a move on the board
    def move_to_board_state(self, board_state, move):
        row = np.where(board_state[:, move] == 0)[0][-1]
        new_board_state = board_state
        new_board_state[row, move] = 1
        return new_board_state
    
    #function to look at board from opponents view
    def switch_sides(self, board_state):
        return np.matrix(board_state*(-1))
    
    def make_state_from_move(self, board_state, move):
        idy = np.where(board_state[:, move] == 0)[0][-1]
        new_state = np.array(board_state)
        new_state[idy, move] = 1
        return new_state
    
        #function to look at board from opponents view
    def opponent_move_from_states(self, prev_state, board_state):
        board_delta = (prev_state - board_state)*(-1)
        opp_move = np.where(board_delta == -1)[1][0]
        return opp_move
    
    #make move based on one-step foresight (minmax alogirthm)
    def foresight_optimal_move(self, board_state):
    
        all_options = self.all_options(board_state)
        
        Q_max_array = []
        #iterate for all possible moves
        for i in all_options:
            #simulate next move
            branch = self.move_to_board_state(board_state, i)
            #switch to opponents view after next view
            branch_opponent = self.switch_sides(branch)
            #get Q values for opponents options after next move
            Q_branch_opponent = self.predict_q_values(branch_opponent)
            #choose move which maximizes opponents Q value after next move
            Q_branch_opponent_max = np.max(Q_branch_opponent)
            Q_max_array.append(Q_branch_opponent_max)
        
        #find next move which minimizes opponents maximal Q potential in the following move
        min_id = np.where(Q_max_array == np.min(Q_max_array))
        move = all_options[min_id]
        
        return int(move)
    
    def current_optimal_move(self, board_state):

        all_options = self.all_options(board_state)
        Q = self.predict_q_values(board_state)
        idx = np.where(Q==np.max(Q[all_options]))[0][0]

        return int(idx)


    def board_state_to_tensor(self, board_state):
        
        board_state_array = np.squeeze(np.asarray(board_state))
        tensor = board_state_array.reshape((1, 6, 7, 1))
        return tensor


    #draw should get positive reward to improve learning against optimal player
    def reward(self, status, reward_orig):

        if status == 'DONE' and reward_orig == 1:
            reward = 1
        elif status == 'ACTIVE':
            reward = 0
        elif status == 'DONE' and reward_orig == -1:
            reward = -1
        else:
            reward = 0.5
        return reward

#     #learning on the fly
#     def learn(self, prev_state, prev_move, state, all_options, move, reward):

#         if prev_move != -1:

#             target = self.set_q_value(prev_state, prev_move, state, all_options, reward)
#             # print(target)
#             self.train_model(prev_state, prev_move, target, 1)
            
#     def train_model(self, prev_state, prev_move, target, epochs):

#         tensor = self.board_state_to_tensor(prev_state, prev_move)

#         if target is not None:
#             self.model.fit(tensor, target, epochs=epochs, verbose=0)

    def load_to_memory(self, prev_state, prev_move, board_state, reward):
        self.memory.append([prev_state, prev_move, board_state, reward])
        
    #load the opponents experience as own experiece to memory
    def load_opponents_experience_to_memory(self, past_prev_state, past_prev_move, prev_state, prev_move, reward):
        board_state_1 = self.make_state_from_move(past_prev_state, past_prev_move)
        board_state_2 = self.make_state_from_move(prev_state, prev_move)
        opp_move = self.opponent_move_from_states(board_state_1, board_state_2)
        self.memory.append([self.switch_sides(board_state_1), opp_move, self.switch_sides(board_state_2), reward])
        
    def save_rewards(self, rewards_list):
        is_file_ = True
        count = 1
        s = ''
        while is_file_:
            s = 'data/rewards_list_' + str(self.model_name) + '_' + str(count) + '.pkl'
            if Path(s).is_file():
                is_file_ = True
                count = count + 1
            else:
                is_file_ = False

        with open(s, 'wb') as output:
            pickle.dump(rewards_list, output)

    def save_memory(self):
        is_file_ = True
        count = 1
        s = ''
        while is_file_:
            s = 'data/value_list_' + str(self.model_name) + '_' + str(count) + '.pkl'
            if Path(s).is_file():
                is_file_ = True
                count = count + 1
            else:
                is_file_ = False

        with open(s, 'wb') as output:
            pickle.dump(self.memory, output)

    def predict_q_values(self, board_state):
        tensor = self.board_state_to_tensor(board_state)
        Q = self.model.predict(tensor)
        return Q[0]

    def set_q_values(self, prev_state, prev_move, board_state, reward):

        Q_s = self.predict_q_values(prev_state)
        Q_s_1 = self.predict_q_values(board_state)
        
        #adjust Q values for illegal moves
        #all_options = self.all_options(prev_state)
        #non_option = [x for x in range(7) if x not in all_options]
        #Q_s[non_option] = float(-1) #float('Inf')  
            
        if reward == 0:
            if len(self.all_options(board_state)) > 0:
                gMaxQ = self.gamma * np.max(Q_s_1[self.all_options(board_state)])
            else:
                print('no moves!!!')
                gMaxQ = 0
            Q_s[prev_move] = Q_s[prev_move] + self.alpha * (reward + gMaxQ - Q_s[prev_move])
        else: 
            Q_s[prev_move] = reward
            
        return Q_s
            
    def save_model(self):
        s = 'DQNs/' + self.model_name + '.h5'

        try:
            os.remove(s)
        except:
            pass

        self.model.save(s)

    def learn_batch(self, memory):
        ## delete active memory after saving
        print('start learning player', self.model_name)
        print('data length:', len(memory))

        # build x_train
        ind = 0
        x_train = np.zeros((len(memory), 6, 7, 1))
        for v in memory:
            [prev_state, prev_move, _, _] = v
            sample = self.board_state_to_tensor(prev_state)
            x_train[ind, :, :, :] = sample
            ind += 1

        loss = 20
        count = 0
        
        if self.early_stopping == False:
            y_train = self.create_targets(memory)
            self.model.fit(x_train, y_train, epochs=15, batch_size=256, verbose=0)
        else:
            while loss > 0.02:
                y_train = self.create_targets(memory)
                self.model.fit(x_train, y_train, epochs=5, batch_size=256, verbose=0)
                loss = self.model.evaluate(x_train, y_train, batch_size=256, verbose=0)[0]
                count += 1
                print('planning number:', count, 'loss', loss)

        loss = self.model.evaluate(x_train, y_train, batch_size=256, verbose=0)


    def create_targets(self, memory):
        y_train = np.zeros((len(memory), 7))
        count = 0
        for v in memory:
            [prev_state, prev_move, board_state, reward] = v
            target = self.set_q_values(prev_state, prev_move, board_state, reward)
            y_train[count, :] = target
            count += 1

        return y_train