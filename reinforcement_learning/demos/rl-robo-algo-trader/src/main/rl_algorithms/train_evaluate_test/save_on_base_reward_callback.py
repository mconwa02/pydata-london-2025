import numpy as np
from stable_baselines3 import TD3, DDPG, SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import os

from src.main.utility.logging import Logger
import src.main.configs.global_configs as configs


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contain the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(
        self,
        check_freq: int,
        log_dir: str,
        verbose=1):
        """
        Constructor
        :param check_freq: Check frequency
        :param log_dir: Log directory
        :param verbose: Verbosity level
        """
        super().__init__(verbose)
        self._logger = Logger.getLogger()
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.mean_window = configs.SB3_SMOOTH_MEAN_WINDOW

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            self._logger.info(f"Saving trained RL model to {self.save_path}")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last episodes (set via configs)
                mean_reward = np.mean(y[-self.mean_window:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        self._logger.info(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True
