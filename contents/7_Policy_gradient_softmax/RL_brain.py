"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  #存储每个回合的，所以开始时清空

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        #输入当前状态，输出选择动作的可能性
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        #输出行为的概率
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability


        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            #反向传递的误差   # 最大化 总体 reward (log_p * R) 就是在最小化 -(log_p * R), 而 tf 的功能里只有最小化 loss
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)  #采取行动的地放action为1，没有采取的为0，再和all_act_prob动作可能性相乘
            #axis = 1,对行求和，因为每一行动作，只有一个被采取，通过tf.one_hot构建的0,1矩阵，再对行求和，得到每一次动作的可能性

            #每个动作采取的可能性×采取对应动作回报值 =>reward，loss的目的是最大化这个reward
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss  self.tf_vt: discounted_ep_rs_norm衰减回合的reward乘以可能性概率=选中的可能×（注意上面取负）
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):  #按照概率来选
        #prob_weights [[0.60442364 0.3955764 ]   注意这个概率时根据网络计算出的，而训练的目标是使好的动作概率更大！
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]}) #np.newaxis 将observation作为其中一维输入
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data 清空，回合制存储
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):   #衰减回合的reward
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs) #np.zeros_like 输出相同维度的0矩阵
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))): #从后往前衰减传递reward
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add  #当前累计回报= 历史回报×衰减系数+当前回报 注意是从后往前算的
        print("discounted_ep_rs",discounted_ep_rs)
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
'''
discounted_ep_rs = 
[87.88311836 87.76072562 87.63709659 87.51221877 87.38607957 87.25866623
 87.12996589 86.99996555 86.86865207 86.73601219 86.60203251 86.46669951
 86.3299995  86.19191869 86.05244312 85.91155871 85.76925122 85.62550628
 85.48030938 85.33364584 85.18550085 85.03585944 84.8847065  84.73202677
 84.57780482 84.42202507 84.26467179 84.10572908 83.94518089 83.783011
 83.61920303 83.45374043 83.2866065  83.11778434 82.94725691 82.77500698
 82.60101715 82.42526985 82.24774732 82.06843164 81.88730469 81.70434817
 81.51954361 81.33287233 81.14431548 80.95385402 80.76146871 80.56714011
 80.3708486  80.17257434 79.97229731 79.76999729 79.56565383 79.35924629
 79.15075383 78.94015538 78.72742968 78.51255523 78.29551033 78.07627306
 77.85482128 77.6311326  77.40518445 77.17695399 76.94641817 76.71355371
 76.47833708 76.24074452 76.00075204 75.7583354  75.5134701  75.26613141
 75.01629435 74.76393369 74.50902393 74.25153932 73.99145386 73.72874127
 73.46337503 73.19532831 72.92457405 72.6510849  72.37483323 72.09579114
 71.81393045 71.52922267 71.24163906 70.95115057 70.65772785 70.36134126
 70.06196087 69.75955643 69.45409741 69.14555294 68.83389186 68.51908268
 68.20109362 67.87989254 67.55544701 67.22772426 66.89669117 66.56231431
 66.22455991 65.88339385 65.53878167 65.19068855 64.83907934 64.48391853
 64.12517023 63.76279821 63.39676587 63.02703624 62.65357195 62.27633531
 61.89528819 61.51039211 61.12160819 60.72889716 60.33221936 59.9315347
 59.52680273 59.11798256 58.70503289 58.28791201 57.86657778 57.44098766
 57.01109865 56.57686732 56.13824982 55.69520184 55.24767862 54.79563497
 54.33902523 53.87780326 53.41192248 52.94133584 52.4659958  51.98585434
 51.50086297 51.0109727  50.51613404 50.01629701 49.51141112 49.00142538
 48.48628826 47.96594773 47.44035125 46.9094457  46.37317748 45.8314924
 45.28433576 44.73165228 44.17338615 43.60948095 43.03987975 42.464525
 41.88335859 41.29632181 40.70335536 40.10439935 39.49939329 38.88827605
 38.27098591 37.64746051 37.01763688 36.38145139 35.73883979 35.08973716
 34.43407794 33.7717959  33.10282414 32.42709509 31.7445405  31.05509141
 30.3586782  29.6552305  28.94467727 28.22694674 27.5019664  26.76966303
 26.02996266 25.28279057 24.52807128 23.76572857 22.99568542 22.21786406
 21.43218592 20.63857164 19.83694105 19.02721318 18.20930624 17.38313762
 16.54862385 15.70568066 14.85422289 13.99416454 13.12541872 12.2478977
 11.36151283 10.46617457  9.5617925   8.64827525  7.72553056  6.79346521
  5.85198506  4.90099501  3.940399    2.9701      1.99        1.        ]
'''


