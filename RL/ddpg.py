# MIT License
# Copyright (c) 2023 saysaysx

import numpy as np
import matplotlib.pyplot as plt


import numpy
import time

from random import choices
import tensorflow as tf
import  gym
import cv2




print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available: ", gpus)

tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, concatenate, BatchNormalization, Dropout, Conv2D, Reshape, Flatten
from tensorflow.keras.layers import MaxPooling2D
import tensorflow as tf
import keras.backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True



sess = tf.compat.v1.Session(config=config)
sess.as_default()




class environment():
    def __init__(self):
        #name = "MountainCarContinuous-v0"
        name = "Pendulum-v1"

        self.env = gym.make(name)

        print(self.env.action_space)

        self.n_action = self.env.action_space.shape[0]

        self.maxa = self.env.action_space.high[0]
        self.mina = self.env.action_space.low[0]

        print(self.maxa)
        print(self.mina)

        self.step = 0

        self.reward = 0.0
        self.index = 0

        self.state_max = self.env.observation_space.high
        self.state_min = self.env.observation_space.low

        print(self.state_max)
        print(self.state_min)

        self.igame  = 0
        self.xnew = []
        self.ynew = []
        self.observ = [[],[],[],[],[]]
        self.ind_obs = 0
        self.n_obs = 1

        self.env_reset()
        self.state()




    def get_state(self, act):
        self.igame = self.igame + 1
        self.step = self.step + 1
        if(self.index == 0):
            self.reward  = 0.0
        self.ind_obs = 0

        next_observation, reward, done, info = self.env.step(act)

        self.field = numpy.array(next_observation)
        reward = reward

        self.reward = self.reward+reward
        self.index = self.index + 1
        #if(self.index>1000):
        #    self.index = 0
        #    done = True

        return self.state(), reward, done

    def env_reset(self):
        self.index = 0
        im = self.env.reset()
        next_observation = numpy.array(im)
        self.field = next_observation



    def get_image(self):
        x = self.env.render()

        return x

    def state(self):
        state = self.field

        return state#(( state - self.state_min) / (self.state_max - self.state_min))

    def get_shape_state(self):
        return self.state().shape
    def get_len_acts(self):
        return self.n_action

    def get_len_state(self):
        return len(self.state())




class ddpg:
    def __init__(self,env):
        self.env = env
        self.index = 0

        self.max_size = self.index
        self.flag = False

        self.shape_state = env.get_shape_state()
        self.len_act = env.get_len_acts()
        self.gamma = 0.99


        self.max_t = 10
        self.start_time = time.time()

        self.T = 512
        self.n_buffer = 100000
        self.buf_index  = 0
        self.flag_buf = False



        self.rews = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.acts = numpy.zeros((self.n_buffer, ),dtype=numpy.float32)
        self.policies = numpy.zeros((self.n_buffer, self.len_act),dtype=numpy.float32)
        self.values = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.float32)
        self.previous_states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.float32)
        self.dones = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.vars = [i for i in range(self.len_act)]



        inp = Input(shape = self.shape_state)
        lay =  Dense(16, activation = 'relu') (inp)
        lay =  Dense(32, activation = 'relu') (lay)


        inp_act = Input(shape = (self.len_act,))
        laya = Dense(32, activation = 'relu') (inp_act)
        

        laya = keras.layers.Concatenate() ([lay,laya])
        laya = Dense(256, activation = 'relu') (laya)
        lay = Dense(256, activation = 'relu') (laya)
        layq = Dense(1, activation="linear") (lay)

        self.modelq = keras.Model(inputs=[inp, inp_act], outputs=[layq])

        inp = Input(shape = self.shape_state)
        lay =  Dense(256, activation = 'relu') (inp)
        layp = Dense(256, activation = 'relu') (lay)
        layp = Dense(1, activation = 'linear') (layp)
        def loutf(x):
            x = tf.clip_by_value(x, self.env.mina,self.env.maxa)
            #x = (x+1.0)*0.5*(self.env.maxa-self.env.mina)+self.env.mina
            return x

        layo = keras.layers.Lambda(loutf) (layp)

        self.modelp = keras.Model(inputs=[inp], outputs=[layo])

        self.targetq = keras.models.clone_model(self.modelq)
        self.targetp = keras.models.clone_model(self.modelp)

        tf.keras.utils.plot_model(self.modelq, to_file='./out/netq.png', show_shapes=True)

        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.0004)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.0004)

        self.cur_reward = 0.0
        self.cur_reward100 = 0.0
        self.cur_reward10 = 0.0
        self.num_games = 0
        self.flag = True


        #self.model.load_weights("./out/name2.h5")


        self.nwin = "Main"
        cv2.namedWindow(self.nwin)
        cv2.setMouseCallback(self.nwin,self.capture_event)
        self.show_on = True

    def capture_event(self,event,x,y,flags,params):
        if event==cv2.EVENT_LBUTTONDBLCLK:
            self.show_on = not self.show_on
            print("St")


    @tf.function
    def get_net_res(self,l_state):
        out = self.modelp(l_state, training = False)
        return out

    @tf.function
    def get_value_res(self,l_state):
        val = self.modelq(l_state, training = False)[1]
        return val

    def get_net_act(self,l_state):

        out = self.get_net_res(numpy.array([l_state]))
        v = numpy.random.randn(1)
        x = out[0]+v*0.1#ou_noise()

        x = numpy.clip(x, self.env.mina ,self.env.maxa)
        return x



    @tf.function
    def train(self, inp, inp_next, actn, rew, dones):
        with tf.GradientTape() as tape1:
            acts = self.targetp(inp_next, training = True)

            qvt = rew+self.gamma*self.targetq([inp_next,acts], training = True)*(1-dones)
            qv = self.modelq([inp,actn], training = True)
            lossq = tf.reduce_mean(tf.math.square(qv - qvt))
            trainable_vars1 = self.modelq.trainable_variables
        grads1 = tape1.gradient(lossq, trainable_vars1)
        self.optimizer1.apply_gradients(zip(grads1, trainable_vars1))
        return lossq

    @tf.function
    def train_actor(self, inp):
        with tf.GradientTape() as tape2:
            acts = self.modelp(inp, training = True)


            qv = self.modelq([inp,acts], training = True)
            lossp = - tf.reduce_mean(qv)
            trainable_vars2 = self.modelp.trainable_variables


        grads2= tape2.gradient(lossp, trainable_vars2)
        self.optimizer2.apply_gradients(zip(grads2, trainable_vars2))
        return lossp



    @tf.function
    def target_train(self):
        tau = 0.002
        target_weights = self.targetp.trainable_variables
        weights = self.modelp.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

        target_weights = self.targetq.trainable_variables
        weights = self.modelq.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

        return




    def learn_all(self):
        if(self.flag_buf):
            max_count = self.n_buffer
        else:
            max_count = self.buf_index

        indices = numpy.random.choice(max_count, self.T)

        inp_next = tf.stop_gradient(tf.cast(self.states[indices] ,tf.float32))
        inp = tf.stop_gradient(tf.cast(self.previous_states[indices] ,tf.float32))
        acts = tf.stop_gradient(tf.cast(self.acts[indices] ,tf.float32))
        rews = tf.stop_gradient(tf.cast(self.rews[indices] ,tf.float32))
        dones = tf.stop_gradient(tf.cast(self.dones[indices] ,tf.float32))
        lossq = self.train(inp,inp_next,acts, rews, dones)

        indices = numpy.random.choice(max_count, self.T)
        inp = tf.stop_gradient(tf.cast(self.previous_states[indices] ,tf.float32))
        lossp = self.train_actor(inp)
        self.target_train()
        return lossq, lossp

    def add(self, reward,done,state,prev_state,act):
        i = self.buf_index
        self.rews[i] = reward
        self.dones[i] = done
        self.states[i] = state
        self.previous_states[i] = prev_state
        self.acts[i] = act
        self.buf_index = self.buf_index+1
        if self.buf_index>=self.n_buffer:
            self.buf_index = 0
            self.flag_buf = True

        return


    def step(self):


        prev_st  = self.env.state()
        act = self.get_net_act(prev_st)


        state, reward, done  = self.env.get_state(act)

        self.add(reward,done,state,prev_st,act)

        self.cur_reward = self.cur_reward*self.gamma*(1-done)+reward

        if done:
            self.env.env_reset()




        if self.flag:
            self.show()

        if self.index>self.T:
            lossq, lossp = self.learn_all()

        self.index = self.index+1
        if(self.index%1000==0):
            print(f"index {self.index} lossq {lossq}  lossp {lossp}  acts {self.acts[self.buf_index-5:self.buf_index]} rew {self.cur_reward}")
            self.cur_reward = 0
            self.show()

        return

    def show(self):
        cv2.imshow(self.nwin,self.env.get_image())
        if cv2.waitKey(1)==27:
            print("27")
            self.flag = not self.flag
        return




env = environment()
ddpg1 = ddpg(env)
for i in range(100000000):
    ddpg1.step()
