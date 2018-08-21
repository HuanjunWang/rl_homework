import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process


def build_mlp(input_placeholder, output_size, scope,
              n_layers=2, size=64, activation=tf.nn.relu, output_activation=None):
    with tf.variable_scope(scope):
        x = tf.keras.Input(tensor=input_placeholder)
        for i in range(n_layers):
            x = tf.keras.layers.Dense(units=size, activation=activation)(x)
        x = tf.keras.layers.Dense(units=output_size, activation=output_activation)(x)
    return x


def path_length(path):
    return len(path["reward"])


def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             min_time_steps_per_batch=6000,
             max_path_length=None,
             learning_rate=5e-2,
             reward_to_go=True,
             animate=True,
             log_dir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             n_layers=1,
             size=32
             ):
    start = time.time()

    # Log
    logz.configure_output_dir(log_dir)
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)

    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # ========================================================================================#
    # Notes on notation:
    #
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    #
    # Prefixes and suffixes:
    # ob - observation
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    #
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    # ========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # ========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    #
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    # ========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="Observation", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="Action", dtype=tf.int32)
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="Action", dtype=tf.float32)

    sy_adv_n = tf.placeholder(shape=[None], name='advance', dtype=tf.float32)

    # ========================================================================================#
    #                           ----------SECTION 4----------
    # Networks
    #
    # Make symbolic operations for
    #   1. Policy network outputs which describe the policy distribution.
    #       a. For the discrete case, just logits for each action.
    #
    #       b. For the continuous case, the mean / log std of a Gaussian distribution over
    #          actions.
    #
    #      Hint: use the 'build_mlp' function you defined in utilities.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ob_no'
    #
    #   2. Producing samples stochastically from the policy distribution.
    #       a. For the discrete case, an op that takes in logits and produces actions.
    #
    #          Should have shape [None]
    #
    #       b. For the continuous case, use the reparameterization trick:
    #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
    #
    #               mu + sigma * z,         z ~ N(0, I)
    #
    #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
    #
    #          Should have shape [None, ac_dim]
    #
    #      Note: these ops should be functions of the policy network output ops.
    #
    #   3. Computing the log probability of a set of actions that were actually taken,
    #      according to the policy.
    #
    #      Note: these ops should be functions of the placeholder 'sy_ac_na', and the
    #      policy network output ops.
    #
    # ========================================================================================#

    if discrete:
        # YOUR_CODE_HERE
        # N x ac_dim
        sy_logits_na = build_mlp(input_placeholder=sy_ob_no, output_size=ac_dim, scope="Policy", size=size,
                                 n_layers=n_layers)
        # N
        sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, 1, name="Predict"),
                                   axis=1)  # Hint: Use the tf.multinomial op
        with tf.name_scope('Loss'):
            num = tf.range(tf.shape(sy_ac_na)[0])  # rows
            indices = tf.stack([num, sy_ac_na], axis=1)
            sy_prob_n = tf.gather_nd(tf.nn.softmax(sy_logits_na), indices)
            sy_logprob_n = tf.log(sy_prob_n)

    else:
        # YOUR_CODE_HERE
        sy_mean = build_mlp(input_placeholder=sy_ob_no, output_size=ac_dim, scope="Policy", size=size,
                            n_layers=n_layers, activation=tf.nn.tanh, output_activation=tf.nn.tanh)

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Policy')

        for w in weights:
            tf.summary.histogram(w.name, w)

        sy_logstd = tf.get_variable(name="Sigma", dtype=tf.float32, shape=[ac_dim],
                                    initializer=tf.constant_initializer(
                                        .1))  # logstd should just be a trainable variable, not a network output.

        tf.summary.histogram('Mean', sy_mean)
        tf.summary.histogram('Std', sy_logstd)

        normal_dist = tf.distributions.Normal(sy_mean, sy_logstd, name="PredictDistribution")

        sy_sampled_ac = tf.clip_by_value(normal_dist.sample(), env.action_space.low, env.action_space.high,
                                         name="PredictAction")

        with tf.name_scope("Loss"):
            sy_logprob_n = normal_dist.log_prob(
                sy_ac_na)  # Hint: Use the log probability under a multivariate gaussian.

    # ========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    # ========================================================================================#
    with tf.name_scope('Loss/'):
        loss = tf.reduce_mean(
            -sy_logprob_n * sy_adv_n)  # Loss function that we'll differentiate to get the policy gradient.
        if not discrete:
            loss -= tf.reduce_mean(normal_dist.entropy()) * .1

        tf.summary.scalar('Actor loss', loss)

    with tf.name_scope("Opt"):
        update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # ========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    # ========================================================================================#

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
            sy_ob_no,
            1,
            "Critic",
            n_layers=n_layers,
            size=size, activation=tf.nn.relu))

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')

        for w in weights:
            tf.summary.histogram(w.name, w)

        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        # YOUR_CODE_HERE
        sy_RewardToGo_n = tf.placeholder(shape=[None], name='RewardToGo', dtype=tf.float32)
        with tf.name_scope("CriticLoss"):
            critic_loss = tf.losses.mean_squared_error(sy_RewardToGo_n, baseline_prediction)
            tf.summary.scalar('Critic Loss', critic_loss)
        with tf.name_scope("Opt/"):
            baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(critic_loss)

    # ========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    # ========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    sess = tf.Session(config=tf_config)
    sess.__enter__()  # equivalent to `with sess:`
    merged = tf.summary.merge_all()
    tf.global_variables_initializer().run()  # pylint: disable=E1101
    writer = tf.summary.FileWriter('/tmp/rl/pg/7', sess.graph)

    # ========================================================================================#
    # Training Loop
    # ========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.0001)
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no: ob[None]})
                ac = ac[0]
                acs.append(ac)

                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation": np.array(obs),
                    "reward": np.array(rewards),
                    "action": np.array(acs)}
            paths.append(path)
            timesteps_this_batch += path_length(path)
            if timesteps_this_batch > min_time_steps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        # ====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above).
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t.
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
        #       entire trajectory (regardless of which time step the Q-value should be for).
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above.
        #
        # ====================================================================================#

        # YOUR_CODE_HERE

        if reward_to_go:

            discount_reward = []
            for path in paths:
                path_len = len(path['reward'])
                discount_factor = [1 * (gamma ** i) for i in range(path_len)]
                for i in range(path_len):
                    discount_reward.append(
                        np.sum(np.array(path['reward'][i:]) * np.array(discount_factor[:path_len - i])))
        else:
            discount_reward = []
            for path in paths:
                ret_tau = 0
                discount_factor = 1
                for reward in path['reward']:
                    ret_tau += reward * discount_factor
                    discount_factor *= gamma
                discount_reward.extend([ret_tau for i in range(len(path['reward']))])

        q_n = np.array(discount_reward, dtype=np.float32)

        # ====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        # ====================================================================================#

        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            b_n = sess.run(baseline_prediction, feed_dict={sy_ob_no: ob_no})
            print("Baseline Mean:%f Max:%f Min:%f" % (np.mean(b_n), np.max(b_n), np.min(b_n)))
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        # ====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        # ====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            # YOUR_CODE_HERE
            adv_n = (adv_n - np.mean(adv_n)) / np.std(adv_n)

        # ====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        # ====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next iteration. Use the
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE
            for i in range(10):
                _, closs = sess.run([baseline_update_op, critic_loss],
                                    feed_dict={sy_ob_no: ob_no, sy_RewardToGo_n: q_n})
            print("Critic loss:", closs)

        # ====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        # ====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.

        # YOUR_CODE_HERE

        feed_dic = {sy_ob_no: ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n, sy_RewardToGo_n: q_n}

        _, opLoss, summary = sess.run([update_op, loss, merged], feed_dict=feed_dic)
        writer.add_summary(summary, itr)
        print("Predict Loss:", opLoss)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        stepss = [len(path['reward']) for path in paths]
        ep_lengths = [path_length(path) for path in paths]
        logz.log_tabular("Steps:", stepss)
        if len(stepss) == 10:
            break
        # logz.log_tabular("Time", time.time() - start)
        # logz.log_tabular("Iteration", itr)
        # logz.log_tabular("AverageReturn", np.mean(returns))
        # logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        # logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        # logz.log_tabular("EpLenStd", np.std(ep_lengths))
        # logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        # logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def train_func(args):
    print(args)
    train_PG(
        exp_name=args.exp_name,
        env_name=args.env_name,
        n_iter=args.n_iter,
        gamma=args.discount,
        min_time_steps_per_batch=args.batch_size,
        max_path_length=args.max_path_length,
        learning_rate=args.learning_rate,
        reward_to_go=args.reward_to_go,
        animate=args.render,
        log_dir=os.path.join(args.logdir, 'seed_%d' % args.seed),
        normalize_advantages=args.normalize_advantages,
        nn_baseline=args.nn_baseline,
        seed=args.seed,
        n_layers=args.n_layers,
        size=args.size
    )


class MyArgument(object):
    def __init__(self):
        self.exp_name = None
        self.env_name = None
        self.n_iter = None
        self.discount = None
        self.batch_size = None
        self.max_path_length = None
        self.learning_rate = None
        self.reward_to_go = None
        self.render = None
        self.log_dir = None
        self.normalize_advantages = None
        self.nn_baseline = None
        self.seed = None
        self.n_layers = None
        self.size = None


def hard_code_arg(learning_rate, batch_size):
    fun_args = MyArgument()
    fun_args.exp_name = 'vpg'
    fun_args.env_name = 'InvertedPendulum-v1'
    fun_args.n_iter = 30
    fun_args.discount = .99
    fun_args.batch_size = 1000
    fun_args.max_path_length = batch_size
    fun_args.learning_rate = learning_rate
    fun_args.reward_to_go = True
    fun_args.render = False
    fun_args.normalize_advantages = False
    fun_args.nn_baseline = True
    fun_args.n_layers = 2
    fun_args.size = 64

    fun_args.log_dir = make_log_dir(fun_args.exp_name, fun_args.env_name)
    return fun_args


def make_log_dir(exp_name, env_name):
    base_dir = '/tmp/pg'
    if not (os.path.exists(base_dir)):
        os.makedirs(base_dir)
    log_dir = exp_name + '_' + env_name + '_' + time.strftime("%d-%m_%H-%M")
    log_dir = os.path.join(base_dir, log_dir)
    if os.path.exists(log_dir):
        os.removedirs(log_dir)
    os.makedirs(log_dir)
    return log_dir


def main():
    args = hard_code_arg(learning_rate=1e-4, batch_size=1000)

    for e in range(args.n_experiments):
        args.seed += 10
        print('Running experiment with seed %d' % args.seed)
        p = Process(target=train_func, args=(args,))
        p.start()
        p.join()


if __name__ == "__main__":
    main()
