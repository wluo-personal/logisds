import pandas as pd
import numpy as np
import os

from logisds.utils import get_console_logger

logger = get_console_logger("Data-Processor")

class DataLoader:
    NUM_FEATURES = 34

    def __init__(self, N_steps: int, seed:int=1, val_ratio:float=0.4, data_path=None):
        """
        This class provide methods is responsible for loading data,
        feature engineering, provide training, validation data
        :param N_steps: int. The steps of time to use.
        :param seed: int, defalt to 1. Random seed.
        :param val_ratio: float. Portion of validation data.
        :param data_path: str, default to None. The data path.

        Methods:
        reverse_scale_data
        -----------------------------------------
        """
        self.N_steps = N_steps
        self.seed = seed
        self.val_ratio = val_ratio
        self.data_path = self.__get_data_path(data_path)
        self.data_ori = self._load_data()
        self.train_index, self.val_index = self._get_train_val_index()
        self._need_scaling = True


    def __get_data_path(self, data_path=None):
        """if data_path is None. Use th data/data.csv as data path"""
        if data_path is None:
            path = os.path.join(os.path.dirname(__file__),
                                "../data/data.csv")
        else:
            path = data_path
        return path

    def _load_data(self):
        return pd.read_csv(self.data_path)

    def _get_train_val_index(self):
        """
        This method will get the index of validation

        """
        val_len = int(len(self.data_ori) * self.val_ratio)
        # The last data can not be used for training
        integers = np.array(range(self.N_steps, len(self.data_ori) - 1))
        np.random.seed(self.seed)
        vals = np.random.choice(integers, val_len, replace=False)
        trains = set(integers) - set(vals)
        trains = np.array(sorted(list(trains)))
        vals = sorted(vals)
        return trains, vals

    def __scale_x(self):
        """ z normalization"""
        self._mean_x = self.data_ori["x"].mean()
        self._std_x = self.data_ori["x"].std()
        self.data_ori["x"] = (self.data_ori["x"] - self._mean_x) / self._std_x

    def __scale_y(self):
        """ z normalization"""
        self._mean_y = self.data_ori["y"].mean()
        self._std_y = self.data_ori["y"].std()
        self.data_ori["y"] = (self.data_ori["y"] - self._mean_y) / self._std_y

    def __scale_time_delta(self):
        """ min-max normalization"""
        self.data_ori["event_time"] = pd.to_datetime(
            self.data_ori["event_time"])
        self.data_ori["time_delta"] =  self.data_ori["event_time"].diff(
            1).dt.total_seconds()
        self.data_ori["time_delta"] = self.data_ori["time_delta"].fillna(
            self.data_ori["time_delta"].mean())
        # self._min_timedelta = self.data_ori["time_delta"].min()
        # self._max_timedelta = self.data_ori["time_delta"].max()
        # self._gap_timedelta = self._max_timedelta - self._min_timedelta
        # self.data_ori["time_delta"] = (
        #         self.data_ori["time_delta"] - self._min_timedelta
        # ) / self._gap_timedelta
        self._mean_time_delta = self.data_ori["time_delta"].mean()
        self._std_time_delta = self.data_ori["time_delta"].std()
        self._min_scaled_time_delta = - self._mean_time_delta / self._std_time_delta
        self.data_ori["time_delta"] = (self.data_ori["time_delta"] -\
                                       self._mean_time_delta) / \
                                      self._std_time_delta

    def __inverse_scale_x(self, x):
        return x * self._std_x + self._mean_x

    def __inverse_scale_y(self, y):
        return y * self._std_y + self._mean_y

    def __inverse_scale_time_delta(self, time_delta):
        # return time_delta * self._gap_timedelta + self._min_timedelta
        return time_delta * self._std_time_delta + self._mean_time_delta

    def _scale_data(self):
        """This method will normalize input data"""
        self.__scale_x()
        self.__scale_y()
        self.__scale_time_delta()

    def reverse_scale_data(self, time_delta, x, y):
        """
        This method can inverse scale data to original scle
        :param time_delta: scaled time_delta
        :param x: scaled x coordinate
        :param y: scaled y coordinate
        :return: tuple (
            orginal scaled time_delta,
            orginal scaled x,
            orginal scaled y)
        """
        return (self.__inverse_scale_time_delta(time_delta),
                self.__inverse_scale_x(x),
                self.__inverse_scale_y(y))

    def _time_to_features(self, time):
        """
        This method will convert a time to a 1-D array
        :param time:
        :return: 1-D array, length will be 7+24
            The first 7 is the weekday one-hot encoding
            The last 24 is the hour one-hot encoding
        """
        encode = np.zeros(7 + 24)
        hour = time.hour
        day = time.weekday()
        encode[[day, 7 + hour]] = 1
        return encode

    def construct_input_data(self, time, x, y, time_delta):
        """
        This method can construct input data, will return a 1-D np.array
        :param time: timestamp
        :param x: scaled x
        :param y: scaled y
        :param time_delta: float scaled time delta
        :return: 1-D array
        where
            index 0-6 is day one-hot encoding
            index 7-31 is hour one-hot encoding
            index 32 is scaled time_delta
            index 33 is scaled x
            index 34 is scaled y
        """
        array_time = self._time_to_features(time)
        array_remain = np.array([time_delta, x, y])
        array = np.concatenate([array_time, array_remain])
        return array

    def prepare_data(self):
        """
        This method will convert data to NN format
        :return: 2-D array the shape will be (original_length, features)
        """
        if self._need_scaling:
            logger.info("start normalizing data")
            self._scale_data()
            logger.info("Normalizing completes")

        concats = []
        for data in self.data_ori.to_dict(orient="records"):
            time = data["event_time"]
            time_delta = data["time_delta"]
            x = data["x"]
            y = data["y"]
            array = self.construct_input_data(
                time=time, x=x, y=y, time_delta=time_delta)
            concats.append(array)

        self.data_array = np.stack(concats, axis=0)
        return self.data_array

    def data_gen(self, mode="train", batch_size=50):
        """
        This method defines a data generator for NN
        :param mode: str, train or validation
        :param batch_size: int, batch_size
        :return: X,y
            X has shape (batch, N_steps, features)
            y has shape (batch, 3)
        """
        if mode == "train":
            index = self.train_index.copy()
        else:
            index = self.val_index.copy()

        num_loop = len(index) // batch_size

        while True:
            np.random.shuffle(index)
            # in each epoch
            for i in range(num_loop):
                batch_indexes = index[i*batch_size: (i+1)*batch_size]
                x_concats = []
                y_concats = []
                for j in batch_indexes:
                    x,y = self.gen_x_y_by_index(j)
                    x_concats.append(x)
                    y_concats.append(y)
                X = np.stack(x_concats, axis=0)
                y = np.stack(y_concats, axis=0)
                yield X,y

    def gen_x_y_by_index(self, index, return_label=True):
        """
        This method can generate X,y, given index
        :param index: interger
        :return: (X,y)
            X is a 1-D array with length 34
            y is the label array with length 3
        """
        X = self.data_array[index-self.N_steps:index]
        if return_label:
            y = self.data_array[index+1, -3:]
        else:
            y = None
        return X,y








# c = DataLoader(100)
# c.prepare_data()
# d = 1

