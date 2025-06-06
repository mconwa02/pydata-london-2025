from typing import Dict, Any
import torch.nn as nn
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gymnasium as gym
import optuna
import numpy as np

from src.main.utility.enum_types import RLAgorithmType
from src.main.rl_algorithms.hyper_parameter_tuning.base_hyper_parameter_tuning import BaseHyperParameterTuning

class TD3HyperParameterTuning(BaseHyperParameterTuning):
    """
    Class to tune hyperparameters of TD3 algorithm.
    """
    def __init__(
            self,
            env: gym.Env,
            rl_algo_type: RLAgorithmType = RLAgorithmType.td3,
            rl_problem_title: str = "Pendulum-v1",
    ):
        """
        Constructor
        :param env: RL environment
        :param rl_algo_type: RL algorithm type
        :param rl_problem_title: RL problem title
        """
        super(TD3HyperParameterTuning, self).__init__(env, rl_algo_type, rl_problem_title)

    def sampleParams(
            self,
            trial: optuna.Trial
    ) -> Dict[str, Any]:
        """
        Sampler for RL algorithm (TD3) hyperparameters.
        :param trial: Optuna Trial
        :return: Sampled parameters
        """
        return self.sampleTD3Params(trial)

    def sampleTD3Params(
            self,
            trial: optuna.Trial
    ) -> dict[str, Any]:
        """
        Sampler for TD3 hyperparams (from SB3 Zoo).

        :param trial:
        :return: Hyperparameters for TD3 algorithm
        """
        n_actions = self._env.action_space.shape[-1]
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
        # Polyak coeff
        tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])

        gradient_steps = trial.suggest_categorical("gradient_steps", [1, 4, 8, 16, 32, 64, 128, 256, 512])

        noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
        noise_std = trial.suggest_float("noise_std", 0, 1)

        # NOTE: Add "verybig" to net_arch when tuning HER
        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", 'elu', 'leaky_relu'])

        net_arch = TD3HyperParameterTuning.getNetArchHyperParameter()[net_arch_type]

        activation_fn = TD3HyperParameterTuning.getActivationFunctionHyperParameter()[activation_fn_name]

        hyperparams = {
            "gamma": gamma,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "policy_kwargs": dict(
                net_arch=net_arch,
                activation_fn=activation_fn,
            ),
            "tau": tau,
        }

        if noise_type == "normal":
            hyperparams["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions))
        elif noise_type == "ornstein-uhlenbeck":
            hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions)
            )
        return hyperparams

    def sampleTD3ParamsV2(
            self,
            trial: optuna.Trial
    ) -> dict[str, Any]:
        """
        Sampler for TD3 hyperparams (from SB3 Zoo).

        :param trial:
        :return: Hyperparameters for TD3 algorithm
        """
        # n_actions = self._env.action_space.shape[-1]
        gamma = trial.suggest_categorical("gamma", [0.9999, 0.99999, 0.999999])
        learning_rate = trial.suggest_categorical("learning_rate", [5e-6, 5e-5, 5e-4, 5e-3])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
        # # Polyak coeff
        tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
        #
        # train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
        #
        # gradient_steps = trial.suggest_categorical("gradient_steps", [1, 4, 8, 16, 32, 64, 128, 256, 512])
        #
        noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
        noise_std = trial.suggest_float("noise_std", 0, 1)

        # NOTE: Add "verybig" to net_arch when tuning HER
        net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

        activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu", 'elu', 'leaky_relu'])

        net_arch = TD3HyperParameterTuning.getNetArchHyperParameter()[net_arch_type]

        activation_fn = TD3HyperParameterTuning.getActivationFunctionHyperParameter()[activation_fn_name]

        hyperparams = {
            "gamma": gamma,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            # "train_freq": train_freq,
            # "gradient_steps": gradient_steps,
            "policy_kwargs": dict(
                net_arch=net_arch,
                activation_fn=activation_fn,
            ),
            "tau": tau,
        }
        return hyperparams

    @staticmethod
    def getNetArchHyperParameter():
        """
        Gets the network architecture hyperparameter.
        :return: Network architecture hyperparameter
        """
        return {
            "small": [64, 64],
            "medium": [256, 256],
            "big": [400, 300],
    }

    @staticmethod
    def getActivationFunctionHyperParameter():
        """
        Gets the network architecture hyperparameter.
        :return: Network architecture hyperparameter
        """
        return {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
