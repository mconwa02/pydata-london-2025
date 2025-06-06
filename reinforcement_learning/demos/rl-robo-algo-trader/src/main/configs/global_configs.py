from datetime import datetime
import os
import gymnasium as gym

# Project name
PROJECT_ROOT_PATH = "rl-robo-algo-trader"

# Data file paths
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Model file paths
MODEL_FOLDER = "model"
os.makedirs(MODEL_FOLDER, exist_ok=True)
FROZEN_LAKE_QL_MODEL_FILE_PATH = os.path.join(MODEL_FOLDER, "frozen_lake_q_learning_model.pl")
FROZEN_LAKE_SARSA_MODEL_FILE_PATH = os.path.join(MODEL_FOLDER, "frozen_lake_sarsa_model.pl")
FROZEN_LAKE_DQN_MODEL_FILE_PATH = os.path.join(MODEL_FOLDER, "frozen_lake_dqn_model.pl")


# LOGGING Configurations
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - : %(message)s in %(pathname)s:%(lineno)d'
CURRENT_DATE = datetime.today().strftime("%d%b%Y")
LOG_FILE = f"RL_Demo_Pydata2025_{CURRENT_DATE}.log"
LOG_FOLDER = "logs"
LOG_PATH = os.path.join(LOG_FOLDER, LOG_FILE)


# Miscellaneous settings
NEW_LINE = "\n"
LINE_DIVIDER = "==========" * 5

# Frozen-lake environment configurations
RENDER_MODE = "rgb_array"
FROZEN_LAKE_ENV: gym.Env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=RENDER_MODE)
SEED = 100
EPISODE_UPDATE_FREQUENCY = 1000

# Rendering configurations
DPI = 72
INTERVAL = 100 # ms

# Q-Learning configurations
Q_LEARN_ALPHA = 0.1
Q_LEARN_GAMMA = 0.99
Q_LEARN_EPSILON = 1.0
Q_LEARN_EPSILON_DECAY = 0.995
Q_LEARN_MIN_EPSILON = 0.01
Q_LEARN_N_EPISODES = 5000
Q_LEARN_MAX_STEPS = 100

# DQN configurations
DQN_N_EPISODES = 1000
DQN_MAX_STEPS = 100

# DDPG (Machin) configurations
DDPG_N_EPISODES = 1000
DDPG_MAX_STEPS = 100

# SB3 configurations (for DDPG, TD3, SAC and PPO RL algorithms)
SB3_N_EPISODES = 200  # Hyperparameter tuning => 10 Training => 100 - 200
SB3_N_EVALUATION_EPISODES = 10
SB3_MAX_STEPS = 1000  # Pendulum => 250, Cart-pole => 500, rl-robo-algo-trader => 1000
SB3_TRAIN_TOTAL_TIMESTEPS = 5e4
SB3_N_TUNING_TRAIN_STEPS = SB3_N_EPISODES * SB3_MAX_STEPS
SB3_CHECK_FREQUENCY = 500
SB3_TRAIN_FREQUENCY = 100
SB3_N_EVALUATIONS = 2
SB3_EVAL_FREQ = 5
SB3_N_STARTUP_TRIALS = 5
SB3_N_TRIALS = 5
SB3_TUNING_TIMEOUT = 2 * 60 * 60
SB3_IS_USE_HYPER_PARAMETER_TUNING = False
# SB3_REWARD_THRESHOLD = 200   # Cart-pole problem,
# SB3_REWARD_THRESHOLD = -200   # pendulum problem
SB3_REWARD_THRESHOLD = 10000 # rl-robo-algo-trader
SB3_SMOOTH_MEAN_WINDOW = 20

# SB3 Hyperparameter tuning results path
IS_USE_HYPER_PARAMETER_TUNING = True
HYPER_PARAMETER_RESULT_FOLDER = "model/hyper_parameter"
HYPER_PARAMETER_TENSORBOARD_FOLDER = "model/tensorboard"
os.makedirs(HYPER_PARAMETER_RESULT_FOLDER, exist_ok=True)
os.makedirs(HYPER_PARAMETER_TENSORBOARD_FOLDER, exist_ok=True)
HYPER_PARAMETER_RESULT_PATH = "tuning_results.csv"
HYPER_PARAMETER_HISTORY_PATH = "_tuning_optimization_history.html"
HYPER_PARAMETER_IMPORTANCE_PATH = "tuning_param_importance.html"
HYPER_PARAMETER_REWARD_CURVE_PATH = HYPER_PARAMETER_RESULT_FOLDER + "/{0}_reward_curve.png"
HYPER_PARAMETER_REWARD_CURVE_DATA_PATH = HYPER_PARAMETER_RESULT_FOLDER + "/{0}_reward_curve.csv"
HYPER_PARAMETER_BEST_VALUES = "tuning_best_values.pkl"
HYPER_PARAMETER_BEST_MODEL_PATH = HYPER_PARAMETER_RESULT_FOLDER + "_{0}_best_model"
TUNED_MODEL_PATH = "model/trained-tuned-models/{0}/{1}/"
TUNED_PARAMETER_FILE_NAME = "tuning_best_values.pkl"
TUNED_TEST_USE_CASE = "low_expiry_{0}_{1}"
DEFAULT_MODEL_USE_CASE = "low_expiry"

HYPER_PARAMETER_NOISE_TYPE = "noise_type"
HYPER_PARAMETER_NOISE_STD = "noise_std"
HYPER_PARAMETER_LR_SCHEDULE = "lr_schedule"
HYPER_PARAMETER_LOG_STD_INIT = "log_std_init"
HYPER_PARAMETER_ORTHO_INIT = "ortho_init"
HYPER_PARAMETER_NET_ARCH = "net_arch"
HYPER_PARAMETER_ACTIVATION_FN = "activation_fn"
HYPER_PARAMETER_POLICY_KWARGS = "policy_kwargs"

# rl-robo-algo-trader data configurations
RL_ALGO_TRADER_ENV_NAME = "RL-Algo-Trader"
DATA_PATH = "./data"
S_P_RAW_DATA_PATH = f"{DATA_PATH}/s_and_p_raw_data_with_features.csv"
S_P_SCALED_DATA_PATH = f"{DATA_PATH}/s_and_p_scaled_data_with_features.csv"
HILPISCH_DATA_PATH = f"{DATA_PATH}/aiif_eikon_eod_data.csv"
DATA_START_DATE="2010-01-01"
DATA_END_DATE="2020-01-01"
S_AND_P_YAHOO_TICKER = "^GSPC"
DATA_BAR_TYPE = "Close"
FEATURE_SMA_10 = "SMA_10"
FEATURE_RSI = "RSI"
FEATURE_MACD = "MACD"
S_AND_P_DATA_COLUMNS = [DATA_BAR_TYPE, FEATURE_SMA_10, FEATURE_RSI, FEATURE_MACD]
TRAIN_SPLIT_FACTOR=0.8

# Problem domain space type [Continuous or Discrete]
IS_PROBLEM_SPACE_DISCRETE = False
