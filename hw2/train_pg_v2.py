import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
from multiprocessing import Process
import shutil


class MyArgument(object):
    def __init__(self,
                 exp_name='vpg',
                 env_name='CartPole-v1',
                 n_iter=100,
                 gamma=.99,
                 min_batch_size=1000,
                 max_path_length=None,
                 learning_rate=1e-3,
                 reward_to_go=True,
                 render=False,
                 normalize_advantage=True,
                 nn_baseline=True,
                 seed=1,
                 n_layers=1,
                 size=32,
                 debug=False):
        self.exp_name = exp_name
        self.env_name = env_name
        self.n_iter = n_iter
        self.gamma = gamma
        self.min_batch_size = min_batch_size
        self.max_path_length = max_path_length
        self.learning_rate = learning_rate
        self.reward_to_go = reward_to_go
        self.render = render
        self.debug = debug

        self.normalize_advantage = normalize_advantage
        self.nn_baseline = nn_baseline
        self.seed = seed
        self.n_layers = n_layers
        self.size = size

        base_dir = '/tmp/pg/%s' % self.env_name

        self.log_dir = 'bl_' if self.nn_baseline else ''
        self.log_dir += 'rtg_' if self.reward_to_go else ''
        self.log_dir += 'norm_' if self.normalize_advantage else ''
        self.log_dir += 'nn_%d_%d_' % (self.n_layers, self.size)
        self.log_dir += 'lr_%6f' % self.learning_rate

        self.log_dir = os.path.join(base_dir, self.log_dir)
        self.log_dir = os.path.join(self.log_dir, 'seed%d' % self.seed)


class PgModel(object):
    def __init__(self, env, n_layers, size, learning_rate, nn_baseline, debug):
        self.observation_dim = env.observation_space.shape[0]
        self.ph_observation = tf.placeholder(shape=[None, self.observation_dim], name="Observation", dtype=tf.float32)
        self.ph_advance = tf.placeholder(shape=[None], name='advance', dtype=tf.float32)
        self.nn_baseline = nn_baseline
        self.ph_q_value = tf.placeholder(shape=[None], name='QValue', dtype=tf.float32)

        self.debug = debug

        if self.debug:
            self.ph_mean_reward = tf.placeholder(name="reward", dtype=tf.float32)
            tf.summary.scalar("MeanReward", self.ph_mean_reward)

    @staticmethod
    def build_mlp(input_placeholder, output_size, scope,
                  n_layers=2, size=64, activation=tf.nn.relu, output_activation=None):
        with tf.variable_scope(scope):
            x = tf.keras.Input(tensor=input_placeholder)
            for i in range(n_layers):
                x = tf.keras.layers.Dense(units=size, activation=activation)(x)
            x = tf.keras.layers.Dense(units=output_size, activation=output_activation)(x)
        return x

    def get_predict_action(self, sess, observation):
        pass

    def update(self, observations, actions, q):
        pass


class PgModelContinuous(PgModel):
    def __init__(self, env, n_layers, size, learning_rate, nn_baseline, debug):
        super().__init__(env, n_layers, size, learning_rate, nn_baseline, debug)
        self.action_dim = env.action_space.shape[0]
        self.ph_action = tf.placeholder(shape=[None, self.action_dim], name="Action", dtype=tf.float32)

        # Define the Actor Model
        with tf.variable_scope("Actor"):
            # N x action dim
            # Output activation ?
            self.action_mean = self.build_mlp(input_placeholder=self.ph_observation,
                                              output_size=self.action_dim, scope="Mean_%d_%d" % (n_layers, size),
                                              size=size, n_layers=n_layers)

            # action dim
            self.action_sigma = tf.get_variable(name='Sigma', shape=[self.action_dim], dtype=tf.float32, trainable=True)

            tf.summary.histogram('Mean', self.action_mean)
            tf.summary.histogram('Std', self.action_sigma)

            # Broadcast expected here
            # Get N x action dim distributions
            self.normal_dist = tf.distributions.Normal(self.action_mean, self.action_sigma, name="PredictDistribution")

            # Expected N* action dis distributions
            self.predict_action = self.normal_dist.sample(name="PredictAction")
            #self.predict_action = tf.clip_by_value(normal_dist.sample(), env.action_space.low, env.action_space.high,
            #                                      name="PredictAction")

            with tf.name_scope("Loss"):
                action_prob = self.normal_dist.log_prob(self.ph_action, name="Prob")
                self.actor_loss = - tf.reduce_mean(action_prob * self.ph_advance)
        with tf.name_scope("Opt/"):
            self.action_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.actor_loss)

        # Define the Critic Model
        if nn_baseline:
            with tf.name_scope("Critic"):
                self.predict_baseline_2d = self.build_mlp(input_placeholder=self.ph_observation,
                                                          output_size=1,
                                                          scope="NN_%d_%d" % (n_layers, size),
                                                          n_layers=n_layers,
                                                          size=size,
                                                          activation=tf.nn.relu)
                self.predict_baseline = tf.squeeze(self.predict_baseline_2d, axis=1, name="PredictBaseline")

                with tf.name_scope("Loss"):
                    self.critic_loss = tf.losses.mean_squared_error(self.ph_q_value, self.predict_baseline)
                    tf.summary.scalar('Critic Loss', self.critic_loss)
            with tf.name_scope("Opt/"):
                self.critic_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.critic_loss)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for w in weights:
            tf.summary.histogram(w.name, w)
        self.merged = tf.summary.merge_all()

    def get_predict_action(self, sess, observation):
        action = sess.run(self.predict_action, feed_dict={self.ph_observation: observation[None]})
        return action

    def update(self, sess, observations, actions, q, normalize_advance, mean_reward):
        if self.nn_baseline:
            # Update Cirtic Network
            sess.run(self.critic_opt, feed_dict={self.ph_observation: observations,
                                                 self.ph_q_value: q})
            baseline = sess.run(self.predict_baseline, feed_dict={self.ph_observation: observations})
            advance = q - baseline
        else:
            advance = q.copy()

        if normalize_advance:
            advance = (advance - np.mean(advance)) / np.std(advance)

        # Update the Actor network
        if self.debug:
            _, summary = sess.run([self.action_opt, self.merged], feed_dict={self.ph_observation: observations,
                                                                             self.ph_action: actions,
                                                                             self.ph_advance: advance,
                                                                             self.ph_q_value: q,
                                                                             self.ph_mean_reward: mean_reward})
        else:
            sess.run(self.action_opt, feed_dict={self.ph_observation: observations,
                                                 self.ph_action: actions,
                                                 self.ph_advance: advance})
            summary = None

        return summary


class PgModelDiscrete(PgModel):
    def __init__(self, env, n_layers, size, learning_rate, nn_baseline, debug):
        super().__init__(env, n_layers, size, learning_rate, nn_baseline, debug)
        self.action_dim = env.action_space.n
        self.ph_action = tf.placeholder(shape=[None], name="Action", dtype=tf.int32)

        # Define the Actor Model
        with tf.name_scope("Actor"):
            self.action_logist = self.build_mlp(input_placeholder=self.ph_observation,
                                                output_size=self.action_dim, scope="NN_%d_%d" % (n_layers, size),
                                                size=size, n_layers=n_layers)
            self.predict_action_2d = tf.multinomial(self.action_logist, 1)
            self.predict_action = tf.squeeze(self.predict_action_2d, axis=1, name="PredictAction")

            self.batch_size = tf.shape(self.ph_observation)[0]
            with tf.name_scope('Loss'):
                indices = tf.stack([tf.range(self.batch_size), self.ph_action], axis=1)
                action_prob = tf.gather_nd(tf.nn.softmax(self.action_logist), indices)
                self.actor_loss = tf.reduce_mean(-tf.log(action_prob) * self.ph_advance)
                tf.summary.scalar('Actor loss', self.actor_loss)

        with tf.name_scope("Opt/"):
            self.action_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.actor_loss)

        # Define the Critic Model
        if nn_baseline:
            with tf.name_scope("Critic"):
                self.predict_baseline_2d = self.build_mlp(input_placeholder=self.ph_observation,
                                                          output_size=1,
                                                          scope="NN_%d_%d" % (n_layers, size),
                                                          n_layers=n_layers,
                                                          size=size,
                                                          activation=tf.nn.relu)
                self.predict_baseline = tf.squeeze(self.predict_baseline_2d, axis=1, name="PredictBaseline")

                with tf.name_scope("Loss"):
                    self.critic_loss = tf.losses.mean_squared_error(self.ph_q_value, self.predict_baseline)
                    tf.summary.scalar('Critic Loss', self.critic_loss)
            with tf.name_scope("Opt/"):
                self.critic_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.critic_loss)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for w in weights:
            tf.summary.histogram(w.name, w)
        self.merged = tf.summary.merge_all()

    def get_predict_action(self, sess, observation):
        action = sess.run(self.predict_action, feed_dict={self.ph_observation: observation[None]})
        return action

    def update(self, sess, observations, actions, q, normalize_advance, mean_reward):
        if self.nn_baseline:
            # Update Cirtic Network
            sess.run(self.critic_opt, feed_dict={self.ph_observation: observations,
                                                 self.ph_q_value: q})
            baseline = sess.run(self.predict_baseline, feed_dict={self.ph_observation: observations})
            advance = q - baseline
        else:
            advance = q.copy()

        if normalize_advance:
            advance = (advance - np.mean(advance)) / np.std(advance)

        # Update the Actor network
        if self.debug:
            _, summary = sess.run([self.action_opt, self.merged], feed_dict={self.ph_observation: observations,
                                                                             self.ph_action: actions,
                                                                             self.ph_advance: advance,
                                                                             self.ph_q_value: q,
                                                                             self.ph_mean_reward: mean_reward})
        else:
            sess.run(self.action_opt, feed_dict={self.ph_observation: observations,
                                                 self.ph_action: actions,
                                                 self.ph_advance: advance})
            summary = None

        return summary


def discount_reward(paths, gamma, reward_to_go):
    if reward_to_go:
        discounted_reward = []
        for path in paths:
            path_len = len(path['reward'])
            discount_factor = [1 * (gamma ** i) for i in range(path_len)]
            for i in range(path_len):
                discounted_reward.append(
                    np.sum(np.array(path['reward'][i:]) * np.array(discount_factor[:path_len - i])))
    else:
        discounted_reward = []
        for path in paths:
            ret_tau = 0
            discount_factor = 1
            for reward in path['reward']:
                ret_tau += reward * discount_factor
                discount_factor *= gamma
            discounted_reward.extend([ret_tau for i in range(len(path['reward']))])

    q_n = np.array(discounted_reward, dtype=np.float32)
    return q_n

def verify_model(sess, model, env):
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]

    observations = np.random.randn(100, observation_dim)
    actions = model.get_predict_action()



def train_pg(args):
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    logz.configure_output_dir(args.log_dir)
    logz.save_params(args.__dict__)

    # Set random seeds
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # Make the gym environment
    env = gym.make(args.env_name)

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    if discrete:
        model = PgModelDiscrete(env, args.n_layers, args.size, args.learning_rate, args.nn_baseline, debug=args.debug)
    else:
        model = PgModelContinuous(env, args.n_layers, args.size, args.learning_rate, args.nn_baseline, debug=args.debug)

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101
    writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    if not discrete:
        verify_model(sess, model, env)


    max_path_length = args.max_path_length or env.spec.max_episode_steps
    for itr in range(args.n_iter):
        print("********** Iteration %i ************" % itr)
        time_steps_this_batch = 0
        paths = []
        while True:
            observation = env.reset()
            obs, acs, rewards = [], [], []

            render_this_episode = (len(paths) == 0 and (itr % 10 == 0) and args.render)
            path_steps = 0
            while True:
                if render_this_episode:
                    env.render()
                    time.sleep(0.0001)

                obs.append(observation)
                action = model.get_predict_action(sess, observation)[0]
                acs.append(action)
                observation, rew, done, _ = env.step(action)
                rewards.append(rew)

                path_steps += 1
                if done or path_steps > max_path_length:
                    break

            path = {"observation": np.array(obs),
                    "reward": np.array(rewards),
                    "action": np.array(acs)}
            paths.append(path)

            time_steps_this_batch += len(obs)
            if time_steps_this_batch > args.min_batch_size:
                break

        path_reward = [sum(path['reward']) for path in paths]
        mean_reward = sum(path_reward) / len(path_reward)
        print(mean_reward)

        ob_batch = np.concatenate([path["observation"] for path in paths])
        ac_batch = np.concatenate([path["action"] for path in paths])
        q_batch = discount_reward(paths, args.gamma, args.reward_to_go)
        summary = model.update(sess, observations=ob_batch, actions=ac_batch,
                               q=q_batch, normalize_advance=args.normalize_advantage, mean_reward=mean_reward)

        if args.debug:
            writer.add_summary(summary, itr)
            logz.pickle_tf_vars()


def main():
    for baseline in [True]:
        for normalize in [True]:
            for reward_to_go in [True]:
                for seed in [8]:
                    env_name = 'CartPole-v0'
                    env_name = 'MountainCar-v0'
                    env_name = 'MountainCarContinuous-v0'
                    args = MyArgument(env_name=env_name, seed=seed, debug=True, n_layers=1,
                                      size=32, reward_to_go=reward_to_go, normalize_advantage=normalize,
                                      nn_baseline=baseline, n_iter=1000)
                    p = Process(target=train_pg, args=(args,))
                    p.start()
                    p.join()


if __name__ == "__main__":
    main()
