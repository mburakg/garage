"""
This script creates a regression test over garage-DQN and baselines-DQN.

It get Atari10M benchmarks from baselines benchmark, and test each task in
its trail times on garage model and baselines model. For each task, there will
be `trail` times with different random seeds. For each trail, there will be two
log directories corresponding to baselines and garage. And there will be a plot
plotting the average return curve from baselines and garage.
"""
import datetime
import os
import os.path as osp
import random
import unittest

from baselines import deepq
from baselines import logger as baselines_logger
from baselines.bench import benchmarks
from baselines.common.atari_wrappers import make_atari
from baselines.common.misc_util import set_global_seeds
from baselines.logger import configure
from mpi4py import MPI
import tensorflow as tf

from garage.envs.wrappers import AtariEnv
from garage.experiment import deterministic
from garage.experiment import LocalRunner
from garage.logger import CsvOutput
from garage.logger import logger as garage_logger
from garage.logger import StdOutput
from garage.logger import TensorBoardOutput
from garage.np.exploration_strategies import EpsilonGreedyStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteCNNQFunction
import tests.helpers as Rh

# Hyperparams for baselines and garage
params = {
    'lr': 1e-4,
    'num_timesteps': int(1e4),
    'n_epochs': 10,
    'n_epoch_cycles': 2,
    'n_rollout_steps': 500,
    'train_freq': 1,
    'discount': 0.99,
    'exploration_fraction': 0.1,
    'exploration_final_eps': 0.01,
    'learning_starts': int(1e2),
    'target_network_update_freq': 2000,
    'dueling': False,
    'buffer_size': int(5e4),
    'batch_size': 32
}


class TestBenchmarkDQN(unittest.TestCase):
    def test_benchmark_dqn(self):
        """
        Compare benchmarks between garage and baselines.

        :return:
        """
        # Load Atari10M tasks, you can check other benchmarks here
        # https://github.com/openai/baselines/blob/master/baselines/bench/benchmarks.py
        atart_envs = benchmarks.get_benchmark('Atari10M')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        benchmark_dir = osp.join(os.getcwd(), 'data', 'local', 'benchmarks',
                                 'dqn', timestamp)
        result_json = {}

        # only run one env for testing purpose
        # *remove later
        atart_envs = {
            'tasks': [
                # {
                #     'desc': 'Seaquest',
                #     'env_id': 'SeaquestNoFrameskip-v4',
                #     'trials': 1
                # },
                # {
                #     'desc': 'Qbert',
                #     'env_id': 'QbertNoFrameskip-v4',
                #     'trials': 1
                # },
                {
                    'desc': 'Pong',
                    'env_id': 'PongNoFrameskip-v4',
                    'trials': 1
                }
            ]
        }

        for task in atart_envs['tasks']:
            env_id = task['env_id']
            env = make_atari(env_id)
            env = deepq.wrap_atari_dqn(env)
            garage_env = AtariEnv(env)

            seeds = random.sample(range(100), task['trials'])

            task_dir = osp.join(benchmark_dir, env_id)
            plt_file = osp.join(benchmark_dir,
                                '{}_benchmark.png'.format(env_id))
            baselines_csvs = []
            garage_csvs = []

            for trial in range(task['trials']):
                env.reset()
                garage_env.reset()
                seed = seeds[trial]

                trial_dir = osp.join(
                    task_dir, 'trial_{}_seed_{}'.format(trial + 1, seed))
                garage_dir = osp.join(trial_dir, 'garage')
                baselines_dir = osp.join(trial_dir, 'baselines')

                with tf.Graph().as_default():
                    # Run garage algorithms
                    garage_csv = run_garage(garage_env, seed, garage_dir)

                    # Run baselines algorithms
                    baselines_csv = run_baselines(env, seed, baselines_dir)

                garage_csvs.append(garage_csv)
                baselines_csvs.append(baselines_csv)

            env.close()

            Rh.plot(
                b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                g_x='Epoch',
                g_y='Episode100RewardMean',
                b_x='steps',
                b_y='mean 100 episode reward',
                trials=task['trials'],
                seeds=seeds,
                plt_file=plt_file,
                env_id=env_id,
                x_label='Epoch',
                y_label='AverageReturn')

            result_json[env_id] = Rh.create_json(
                b_csvs=baselines_csvs,
                g_csvs=garage_csvs,
                seeds=seeds,
                trails=task['trials'],
                g_x='Epoch',
                g_y='Episode100RewardMean',
                b_x='steps',
                b_y='mean 100 episode reward',
                factor_g=params['n_rollout_steps'] * params['n_epoch_cycles'],
                factor_b=1)

        Rh.write_file(result_json, 'DQN')

    test_benchmark_dqn.huge = True


def run_garage(env, seed, log_dir):
    """
    Create garage model and training.

    Replace the ddpg with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return:
    """
    deterministic.set_seed(seed)

    with LocalRunner() as runner:
        env = TfEnv(env)

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec,
            size_in_transitions=params['buffer_size'],
            time_horizon=1)

        qf = DiscreteCNNQFunction(
            env_spec=env.spec,
            filter_dims=(8, 4, 3),
            num_filters=(32, 64, 64),
            strides=(4, 2, 1),
            dueling=params['dueling'])

        policy = DiscreteQfDerivedPolicy(env_spec=env, qf=qf)

        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env.spec,
            total_timesteps=params['num_timesteps'],
            max_epsilon=1.0,
            min_epsilon=params['exploration_final_eps'],
            decay_ratio=params['exploration_fraction'])

        dqn = DQN(
            env_spec=env.spec,
            policy=policy,
            qf=qf,
            exploration_strategy=epilson_greedy_strategy,
            replay_buffer=replay_buffer,
            qf_lr=params['lr'],
            discount=params['discount'],
            grad_norm_clipping=10,
            double_q=True,  # baseline use double_q internally
            min_buffer_size=params['learning_starts'],
            n_train_steps=params['n_rollout_steps'] * params['train_freq'],
            n_epoch_cycles=params['n_epoch_cycles'],
            target_network_update_freq=(
                params['target_network_update_freq']  # yapf: disable
                / params['n_rollout_steps']),
            buffer_batch_size=params['batch_size'])

        # Set up logger since we are not using run_experiment
        tabular_log_file = osp.join(log_dir, 'progress.csv')
        tensorboard_log_dir = osp.join(log_dir, 'progress')
        garage_logger.add_output(StdOutput())
        garage_logger.add_output(CsvOutput(tabular_log_file))
        garage_logger.add_output(TensorBoardOutput(tensorboard_log_dir))

        runner.setup(dqn, env)
        runner.train(
            n_epochs=params['n_epochs'],
            n_epoch_cycles=params['n_epoch_cycles'],
            batch_size=params['n_rollout_steps'])

        garage_logger.remove_all()

    return tabular_log_file


def run_baselines(env, seed, log_dir):
    """
    Create baselines model and training.

    Replace the ddpg and its training with the algorithm you want to run.

    :param env: Environment of the task.
    :param seed: Random seed for the trail.
    :param log_dir: Log dir path.
    :return
    """
    rank = MPI.COMM_WORLD.Get_rank()
    seed = seed + 1000000 * rank
    set_global_seeds(seed)
    env.seed(seed)

    # Set up logger for baselines
    configure(dir=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    baselines_logger.info('rank {}: seed={}, logdir={}'.format(
        rank, seed, baselines_logger.get_dir()))

    # For latest version of baselines

    # deepq.learn(
    #     env,
    #     'conv_only',
    #     convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    #     hiddens=[256],
    #     dueling=params['dueling'],
    #     lr=params['lr'],
    #     total_timesteps=params['num_timesteps'],
    #     buffer_size=params['buffer_size'],
    #     exploration_fraction=params['exploration_fraction'],
    #     exploration_final_eps=params['exploration_final_eps'],
    #     train_freq=params['train_freq'],
    #     learning_starts=params['learning_starts'],
    #     target_network_update_freq=params['target_network_update_freq'],
    #     gamma=params['discount'],
    #     batch_size=params['batch_size'],
    #     print_freq=10)

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=params['dueling'],
    )

    deepq.learn(
        env,
        q_func=model,
        lr=params['lr'],
        max_timesteps=params['num_timesteps'],
        buffer_size=params['buffer_size'],
        exploration_fraction=params['exploration_fraction'],
        exploration_final_eps=params['exploration_final_eps'],
        train_freq=params['train_freq'],
        learning_starts=params['learning_starts'],
        target_network_update_freq=params['target_network_update_freq'],
        gamma=params['discount'],
        prioritized_replay=False,
        print_freq=1)

    return osp.join(log_dir, 'progress.csv')
