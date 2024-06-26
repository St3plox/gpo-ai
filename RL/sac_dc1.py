# MIT License
# Copyright (c) 2023 saysaysx

#import matplotlib.pyplot as plt


import numpy
import time

from random import choices
import tensorflow as tf
import gymnasium as gym
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
        self.env = gym.make("MsPacman-v4", render_mode="rgb_array")

        #self.env = wr.AtariWrapper(self.env,frame_skip=1, terminal_on_life_loss=False)
        #self.env = wr.NoopResetEnv(self.env)
        #self.env = wr.FireResetEnv(self.env)
        self.env = gym.wrappers.ResizeObservation(self.env,shape=(105,80))
        self.env = gym.wrappers.GrayScaleObservation(self.env,keep_dim=True)
        self.env = gym.wrappers.FrameStack(self.env,num_stack=4)
        print(self.env.action_space.n)
        self.n_action = self.env.action_space.n
        print("Максимальная награда за действие:", self.env.reward_range)

        self.step = 0

        self.reward = 0.0
        self.index = 0
        print(self.env.observation_space)
        shx = (self.env.observation_space.high).transpose([1,2,0,3]).shape
        self.state_max = numpy.reshape(self.env.observation_space.high.transpose([1,2,0,3]),[shx[0],shx[1],shx[2]*shx[3]])
        self.state_min = numpy.reshape(self.env.observation_space.low.transpose([1,2,0,3]),[shx[0],shx[1],shx[2]*shx[3]])


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


        next_observation, reward, done, _,_ = self.env.step(act)
        reward = reward / 10.0
        next_observation = numpy.array(next_observation)
        shx = (next_observation).transpose([1,2,0,3]).shape
        self.field = numpy.reshape(next_observation.transpose([1,2,0,3]),[shx[0],shx[1],shx[2]*shx[3]])


        self.reward = self.reward+reward
        self.index = self.index + 1
        #if(self.index>1000):
        #    self.index = 0
        #    done = True

        return self.state(), reward, done

    def env_reset(self):
        self.index = 0
        im = numpy.array(self.env.reset()[0])
        shx = im.transpose([1,2,0,3]).shape
        next_observation = numpy.reshape(im.transpose([1,2,0,3]),[shx[0],shx[1],shx[2]*shx[3]])


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




class sac:
    def __init__(self,env):
        self.env = env
        self.index = 0

        self.max_size = self.index
        self.flag = False

        self.shape_state = env.get_shape_state()
        self.len_act = env.get_len_acts()
        self.gamma = 0.99
        self.alpha = 0.5


        self.max_t = 10
        self.start_time = time.time()

        self.T = 512
        self.n_buffer = 50000
        self.buf_index  = 0
        self.flag_buf = False



        self.rews = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.acts = numpy.zeros((self.n_buffer, 1),dtype=numpy.float32)
        self.policies = numpy.zeros((self.n_buffer, self.len_act),dtype=numpy.float32)
        self.values = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.uint8)
        self.previous_states = numpy.zeros((self.n_buffer, *self.shape_state),dtype=numpy.uint8)
        self.dones = numpy.zeros((self.n_buffer,),dtype=numpy.float32)
        self.vars = [i for i in range(self.len_act)]

        def dconv(filt,size,x, strides=2):
            y = keras.layers.UpSampling2D(size=strides)(x)
            y = keras.layers.Conv2D(filters = filt,kernel_size=size,activation = 'tanh',padding='same') (y)
            return y





        print("Shape")
        print(self.shape_state)

        inp1 = Input(shape = self.shape_state)
        lay = Conv2D(16,(8,8), strides = (4,4),activation= 'relu',padding='same') (inp1)
        lay = Conv2D(32,(4,4), strides = (2,2), activation= 'relu',padding='same') (lay)
        layb = Conv2D(64,(3,3), strides = (1,1), activation= 'relu',padding='same') (lay)
        layb = Flatten() (layb)
        lay1 = Dense(256, activation="relu") (layb)
        layv = Dense(self.len_act, activation = 'linear') (lay1)
        self.modelq1 = keras.Model(inputs=inp1, outputs=[layv])

        inp2 = Input(shape = self.shape_state)
        lay = Conv2D(16,(8,8), strides = (4,4),activation= 'relu',padding='same') (inp2)
        lay = Conv2D(32,(4,4), strides = (2,2), activation= 'relu',padding='same') (lay)
        layb = Conv2D(64,(3,3), strides = (1,1), activation= 'relu',padding='same') (lay)
        layb = Flatten() (layb)
        lay2 = Dense(256, activation="relu") (layb)
        layv = Dense(self.len_act, activation = 'linear') (lay2)
        self.modelq2 = keras.Model(inputs=inp2, outputs=[layv])

        lay = concatenate([lay1, lay2])
        model = keras.Model(inputs=[inp1,inp2], outputs=[lay])
        inpp = Input(shape = (512,))
        lay = Dense(512, activation="linear") (inpp)
        lay = Dense(256, activation="linear") (lay)
        layp = Dense(self.len_act, activation="softmax") (lay)
        self.modelpt = keras.Model(inputs=inpp, outputs=[layp])

        mp = self.modelpt(model.layers[-1].output)
        self.modelp = keras.Model(inputs=[inp1,inp2], outputs=[mp])



        inp1 = Input(shape = self.shape_state)
        lay = Conv2D(16,(8,8), strides = (4,4),activation= 'relu',padding='same') (inp1)
        lay = Conv2D(32,(4,4), strides = (2,2), activation= 'relu',padding='same') (lay)
        layb = Conv2D(64,(3,3), strides = (1,1), activation= 'relu',padding='same') (lay)
        lay1 = Flatten() (layb)
        lay1 = Dense(256, activation="relu") (lay1)
        layv = Dense(self.len_act, activation = 'linear') (lay1)
        self.targetq1 = keras.Model(inputs=inp1, outputs=[layv])


        inp2 = Input(shape = self.shape_state)
        lay = Conv2D(16,(8,8), strides = (4,4),activation= 'relu',padding='same') (inp2)
        lay = Conv2D(32,(4,4), strides = (2,2), activation= 'relu',padding='same') (lay)
        layb = Conv2D(64,(3,3), strides = (1,1), activation= 'relu',padding='same') (lay)
        lay2 = Flatten() (layb)
        lay2 = Dense(256, activation="relu") (lay2)
        layv = Dense(self.len_act, activation = 'linear') (lay2)
        self.targetq2 = keras.Model(inputs=inp2, outputs=[layv])

        tf.keras.utils.plot_model(self.modelq1, to_file='./out/netq.png', show_shapes=True)
        tf.keras.utils.plot_model(self.targetq1, to_file='./out/nettq1.png', show_shapes=True)
        tf.keras.utils.plot_model(self.modelp, to_file='./out/netp.png', show_shapes=True)

        self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=0.00025)
        self.optimizer3 = tf.keras.optimizers.Adam(learning_rate=0.0006)
        self.alphav = tf.Variable(2.0)
        self.max_alpha = 0.01


        self.cur_reward = 0.0
        self.max_reward = 0.0

        self.cur_reward100 = 0.0
        self.cur_reward10 = 0.0
        self.num_games = 0
        self.flag = True


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
        inp = tf.cast(l_state,tf.float32)/255.0
        out = self.modelp([inp,inp] , training = False)
        #out = tf.clip_by_value(out,0.001,0.992)

        return out

    @tf.function
    def get_value_res(self,l_state):

        val = self.modelq1(l_state/255.0, training = False)[1]
        return val

    def get_net_act(self,l_state):


        out = self.get_net_res(numpy.array([l_state]))

        vars = [i for i in range(self.len_act)]
        index = numpy.array(choices(vars, out[0])[0])
        return index, out[0]



    @tf.function
    def train_q(self, inp, inp_next, actn, rew, dones, al):
        with tf.GradientTape(persistent=True) as tape1:
            inp1 = inp / 255.0
            inp_next1 = inp_next / 255.0



            qv1 =  self.modelq1(inp1, training = True)
            qv2 =  self.modelq2(inp1, training = True)

            tqv1 =  self.targetq1(inp1, training = True)
            tqv2 =  self.targetq2(inp1, training = True)


            #acts1 = tf.random.categorical(tf.math.log(y_pi+1e-12), 1)

            targ1 = self.targetq1(inp_next1, training = True)
            targ2 = self.targetq2(inp_next1, training = True)



            #targ_1 = tf.gather_nd(batch_dims=1,params = targ1,indices  = acts1)
            #targ_2 = tf.gather_nd(batch_dims=1,params = targ2,indices  = acts1)
            #api = tf.gather_nd(batch_dims=1,params = y_pi,indices  = acts1)

            targ = (targ1+targ2)*0.5
            y_pi = self.modelp([inp_next1,inp_next1], training = True)
            y_pi = tf.clip_by_value(y_pi,1e-12,0.99999999999)

            logpi = tf.math.log(y_pi)
            q1 =  tf.gather_nd(batch_dims=1,params = qv1,indices  = actn)
            q2 =  tf.gather_nd(batch_dims=1,params = qv2,indices  = actn)

            #qt1 =  tf.gather_nd(batch_dims=1,params = tqv1,indices  = actn)
            #qt2 =  tf.gather_nd(batch_dims=1,params = tqv2,indices  = actn)
            sr = tf.reduce_mean(abs(targ))*0.1

            dift = tf.reduce_sum((targ-tf.stop_gradient(self.alphav)*logpi)*y_pi*sr, axis=-1)

            qvt =  tf.stop_gradient(rew+self.gamma*dift*(1-dones))

            dif1a = tf.math.square(q1 - qvt)
            #dif1b = tf.math.square(qt1+tf.clip_by_value(q1-qt1,-0.5,0.5)-qvt)
            #dif1 = tf.maximum(dif1a,dif1b)

            dif2a = tf.math.square(q2 - qvt)
            #dif2b = tf.math.square(qt2+tf.clip_by_value(q2-qt2,-0.5,0.5)-qvt)
            #dif2 = tf.maximum(dif2a,dif2b)

            lossq1 = tf.reduce_mean(dif1a)
            lossq2 = tf.reduce_mean(dif2a)
            lossq = lossq1+lossq2
            trainable_varsa = self.modelq1.trainable_variables+self.modelq2.trainable_variables
        gradsa = tape1.gradient(lossq, trainable_varsa)

        with tf.GradientTape() as tape2:
            y_pii = self.modelp([inp1,inp1], training = True)
            y_pii = tf.clip_by_value(y_pii,1e-12,0.9999999999999)

            logpi = tf.math.log(y_pii)
            minq = tf.stop_gradient((qv1+qv2)*0.5)

            diflm = tf.reduce_sum(y_pii*(tf.stop_gradient(self.alphav)*logpi*sr - minq),axis=-1)
            dm = tf.reduce_mean(diflm)
            #lossmin = tf.reduce_mean(tf.math.exp(-y_pii/0.0001))

            lossp = dm#+lossmin

            trainable_vars2 = self.modelpt.trainable_variables
        grads2 = tape2.gradient(lossp, trainable_vars2)

        self.optimizer1.apply_gradients(zip(gradsa, trainable_varsa))

        self.optimizer2.apply_gradients(zip(grads2, trainable_vars2))

        #with tf.GradientTape() as tape3:
        #    H = tf.reduce_sum(y_pii*logpi,axis=-1)
        #    valH = H-tf.math.log(0.111111111)*0.995
        #    lossa = - self.alphav*valH + tf.nn.relu(1e-3-self.alphav)*100.0
        #grads3 = tape3.gradient(lossa, [self.alphav])
        #self.optimizer3.apply_gradients(zip(grads3, [self.alphav]))





        return lossq, lossp, tf.reduce_mean(tf.math.abs(qv1-qv2)), tf.reduce_mean(- self.alphav*logpi)





    @tf.function
    def target_train(self):
        tau = 0.01


        target_weights = self.targetq2.trainable_variables
        weights = self.modelq2.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

        target_weights = self.targetq1.trainable_variables
        weights = self.modelq1.trainable_variables
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

        return




    def learn_all(self):
        if(self.flag_buf):
            max_count = self.n_buffer
        else:
            max_count = self.buf_index

        indices = numpy.random.choice(max_count, self.T)

        inp_next = tf.cast(self.states[indices] ,tf.float32)
        inp = tf.cast(self.previous_states[indices],tf.float32)

        acts = tf.cast(self.acts[indices] ,tf.int32)
        rews = tf.cast(self.rews[indices] ,tf.float32)
        dones = tf.cast(self.dones[indices] ,tf.float32)
        al = (tf.cast(numpy.random.random(), tf.float32)-0.5)*0.00001+0.5
        lossq, lossp, qv, qvt = self.train_q(inp,inp_next,acts, rews, dones, al)
        #if(numpy.random.random()>0.999):
        #    self.alphav.assign(self.max_alpha)






        #lossp = self.train_actor(inp)



        self.target_train()
        return lossq, lossp, qv, qvt

    def add(self, reward,done,prev_state,state,act, pol):
        i = self.buf_index
        self.rews[i] = reward
        self.dones[i] = done
        self.states[i] = state
        self.previous_states[i] = prev_state
        self.acts[i] = act
        self.policies[i] = pol
        self.buf_index = self.buf_index+1
        if self.buf_index>=self.n_buffer:
            self.buf_index = 0
            self.flag_buf = True

        return


    def step(self):


        prev_st  = self.env.state()
        act, pol = self.get_net_act(prev_st)


        state, reward, done  = self.env.get_state(act)




        self.add(reward,done,prev_st,state,act, pol)

        self.cur_reward = self.cur_reward+reward

        if done:
            self.env.env_reset()
            self.cur_reward10 = self.cur_reward10 + self.cur_reward
            self.cur_reward = 0
            self.num_games = self.num_games + 1
            if self.num_games>100:
                self.cur_reward100 = self.cur_reward10/100
                self.cur_reward10 = 0
                self.num_games = 0
                if self.cur_reward100 > self.max_reward:
                    self.max_reward = self.cur_reward100
                    self.max_alpha = self.alphav.numpy()





        if self.flag:
            self.show()

        if self.index>self.T*4 and self.index%64==0:
            lossq, lossp, qv, qvt = self.learn_all()


            if(self.index%4000==0 and self.buf_index>self.T):
                print(f"index {self.index} lossq {lossq}  lossp {lossp} qv {qv}  qvt {qvt} acts {self.policies[self.buf_index-1:self.buf_index]} rew {self.cur_reward100} alpha {self.alphav.numpy()}")
                self.cur_reward = 0
                self.show()

        self.index = self.index+1
        return

    def show(self):
        cv2.imshow(self.nwin,self.env.get_image())
        if cv2.waitKey(1)==27:
            print("27")
            self.flag = not self.flag
        return




env = environment()
sac1 = sac(env)
for i in range(100000000):
    sac1.step()
