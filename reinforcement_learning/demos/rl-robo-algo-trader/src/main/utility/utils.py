import pickle
from typing import List, Dict, Optional, Any
from itertools import islice
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display              # headless display
import matplotlib.pyplot as plt                   # plotting
from matplotlib.animation import FuncAnimation    # animation
from IPython.display import display
import pandas as pd
import random
import torch
from typing import List, Dict, Any, Tuple

from src.main.configs import global_configs as configs
from src.main.utility.data_provider import DataProvider

class Helpers:
    """
    Helper utilities
    """

    @staticmethod
    def createDirectoryIfItDoesNotExist(directory: str):
        """
        Create a directory if it does not exist
        :param directory: The directory to create
        :return:
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def getPojectRootPath(project_name: str = configs.PROJECT_ROOT_PATH) -> str:
        """
        Gets the project root path
        :param project_name: The project name
        :return: Returns the project root path
        """
        current_dir = Path(__file__)
        project_dir = [p for p in current_dir.parents if p.parts[-1] == project_name][0]
        return str(project_dir)

    @staticmethod
    def serialObject(
            data: Any,
            pickle_path: str
    ) -> None:
        """
        Serialize the data to a pickle file
        :param data: Data to serialize
        :param pickle_path: Pickle path
        :return: None
        """
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def deserializeObject(pickle_path: str) -> Any:
        """
        Deserialize the data from a pickle file
        :param pickle_path: Pickle path
        :return: Returns the deserialized data
        """
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def filterDict(
            input_dict: Dict[str, Any],
            filter_list: List[Any]
    ) -> Dict[str, Any]:
        """
        Remove/filter a input dictionary based on a list
        :param input_dict: The input dictionary
        :param filter_list:
        :return: Filtered dictionary
        """
        output_dict = {
            key: input_dict[key]
            for key in input_dict if key not in filter_list
        }
        return output_dict

    @staticmethod
    def getEnumType(
            enum_type: Any,
            enum_type_name: str
    ) -> Any:
        """
        Gets the enum type based on the specified type name
        :param enum_type: Enum type
        :param enum_type_name: Enum name
        :return: Enum value
        """
        enum_dict = enum_type.__dict__
        enum_value = enum_dict.get(enum_type_name.lower())
        if enum_value is None:
            raise ValueError(f"Enum type {enum_type_name} does not exist!")
        return enum_value

    @staticmethod
    def chunklistCollection(
            lst: List[Any],
            chunk_size: int
    ) -> List[List[Any]]:
        """
        Chunks a list collection
        :param lst: The list to chunk
        :param chunk_size:
        :return: Chunked list
        """
        it = iter(lst)  # Create an iterator
        return [
            list(islice(it, chunk_size))
            for _ in range((len(lst) + chunk_size - 1) // chunk_size)
        ]

    @staticmethod
    def chunkArray(
            lst: List[Any],
            chunk_size: int
    ) -> List[Any]:
        """
        Chunks a list collection
        :param lst: The list to chunk
        :param chunk_size:
        :return: Chunked array
        """
        array_list = np.array(lst, dtype=object)
        chunked_array = np.array_split(array_list, chunk_size)
        return chunked_array


    @staticmethod
    def text2Boolean(arg):
        """
        Parses a string to a boolean type
        :param arg: Input argument
        :return: Boolean value
        """
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
            print("Invalid argument specified, had to set it to default value of 'True'")
            return True

    @staticmethod
    def animateEnvironment(images: List[Any]):
        """
        Animates the environment
        :param images: Images
        """
        plt.figure(
            figsize=(images[0].shape[1] / configs.DPI, images[0].shape[0] / configs.DPI),
            dpi=configs.DPI
        )
        patch = plt.imshow(images[0])
        plt.axis = ('off')
        animate = lambda i: patch.set_data(images[i])
        ani = FuncAnimation(
            plt.gcf(),
            animate,
            frames=len(images),
            interval=configs.INTERVAL)
        display.display(display.HTML(ani.to_jshtml()))
        plt.close()

    @staticmethod
    def animationPolicy(agent, env, fps=2):
        """
        Animation policy specification
        :param agent: Input agent
        :param env: Input environment
        :param fps: FramePer Second
        :return: Animation
        """
        # Start virtual display
        display = Display(visible=0, size=(400, 400))
        display.start()

        n_rows, n_cols = agent.Q.shape
        states, actions = [], []
        s, _ = env.reset()
        states.append(s)

        # Rollout under greedy policy
        for _ in range(100):
            a = np.argmax(agent.Q[s])
            s, _, done, _, _ = env.step(a)
            states.append(s)
            if done:
                break

        # Setup plot
        fig, ax = plt.subplots()
        # ax.set_xlim(0, env.desc.shape[1])
        # ax.set_ylim(0, env.desc.shape[0])
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        agent_dot, = ax.plot([], [], 'ro', ms=20)

        def init():
            agent_dot.set_data([], [])
            return agent_dot,

        def update(frame):
            # Convert state index to (row, col)
            row, col = divmod(states[frame], n_cols)
            agent_dot.set_data(col + 0.5, n_rows - row - 0.5)
            return agent_dot,

        anim = FuncAnimation(fig, update, init_func=init,
                             frames=len(states), interval=1000 / fps, repeat=False)

        plt.close(fig)  # prevent static display
        display.stop()
        return anim

    @staticmethod
    def appendTableRow(
            df: pd.DataFrame,
            row: pd.Series):
        """
        :param df: Dataframe to append row to
        :param row: Row to append
        :return: New dataframe with appended row
        """
        return pd.concat([
            df,
            pd.DataFrame([row], columns=row.index)]
        ).reset_index(drop=True)

    @staticmethod
    def createTable(
            columns: List[str]

    ) -> pd.DataFrame:
        """
        Creates a new data table
        :param columns: Columns
        """
        df = pd.DataFrame(
            columns=columns
        )
        return df

    @staticmethod
    def displayTable(
            df: pd.DataFrame,
            n_rows: int,
            n_columns: int
    ) -> None:
        """
        Displays sample rows of a data table
        :param df: Data table
        :param n_rows: Number of rows
        """
        with pd.option_context("display.max_rows", n_rows, "display.max_columns", n_columns,
                               "max_colwidth", 100):
            display(df[:n_rows])

    @staticmethod
    def setSeeds(seed: int = configs.SEED):
        """
        Sets the seed value for the computation to maintain reproducibility
        :param seed: Seed value
        :return: None
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def getAssetData(is_hilpisch_data: bool = False) -> pd.DataFrame:
        """
        Gets raw price data
        :param is_hilpisch_data: Whether the data is hilpisch
        :return: Data
        """
        if is_hilpisch_data:
            raw_data_df = pd.read_csv(configs.HILPISCH_DATA_PATH, index_col=0,
                                      parse_dates=True).dropna()
            return raw_data_df
        else:
            data_provider = DataProvider()
            close_data_raw_df, close_data_scaled, close_data_train, close_data_test = data_provider.getData()
            return close_data_raw_df

    @staticmethod
    def getDataPartitionWindows(
            split_fraction: float = configs.TRAIN_SPLIT_FACTOR) -> Dict[str, Tuple[int, int]]:
        """
        Computes the data partition windows
        :param split_fraction: Split fraction
        :return: Partitions
        """
        data_df = Helpers.getAssetData()
        data_len = len(data_df)
        train_len = int(data_len * split_fraction)
        partitions = {
            "train_window": {
                "start": 0,
                "end": train_len,
            },
            "test_window": {
                "start": train_len,
                "end": data_len,
            }

        }
        return partitions

    @staticmethod
    def formatPrice(price) -> str:
        """
        Price formater
        :param price: Price
        """
        return ("-$" if price < 0 else "$") + "{0:.2f}".format(abs(price))

    @staticmethod
    def plotRlBehavior(
            data_input: List[float],
            states_buy: List[int],
            states_sell: List[int],
            profit: float
    ):
        """
        Plot the behaviour of the RL agent
        :param data_input_df: Data input
        :param states_buy: List of the buy signals during the RL episode
        :param states_sell: List of sell signals during the RL episode
        :param profit: RL reward
        """
        fig = plt.figure(figsize=(15, 5))
        plt.plot(data_input, color='r', lw=2.)
        plt.plot(data_input, '^', markersize=10, color='m', label='Buying signal', markevery=states_buy)
        plt.plot(data_input, 'v', markersize=10, color='y', label='Selling signal', markevery=states_sell)
        plt.title(f"Total gains: {profit: .4f}")
        plt.legend()
        # plt.savefig('output/'+name+'.png')
        plt.show()





