import os

import numpy as np
import pandas as pd

from logisds.data import DataLoader
from logisds.model import TSModel
from logisds.utils import get_console_logger

logger = get_console_logger("Inference")


class Inference:

    def __init__(self,
                 N_steps,
                 model_weights_path=None,
                 detail_output_path=None,
                 halfhour_output_path=None):
        self.N_steps = N_steps
        self.model_weights_path = self._get_model_weights_path(
            model_weights_path)
        self.output_path = self._get_inference_output_path(detail_output_path)
        self.halfhour_output_path = self._get_halfhour_output_path(
            halfhour_output_path)

        self.model = self.load_model()
        logger.info("Loading model complete.")
        self.data_loader = DataLoader(N_steps=self.N_steps)
        self.data_loader.prepare_data()

    def _get_model_weights_path(self, model_weights_path=None):
        """This method will resolve correct model weight path"""
        if model_weights_path is None:
            path = os.path.join(os.path.dirname(__file__),
                                "../data/model")
        else:
            path = model_weights_path
        return path

    def _get_inference_output_path(self, output_path=None):
        """This method will resolve correct model weight path"""
        if output_path is None:
            path = os.path.join(os.path.dirname(__file__),
                                "../data/inference.csv")
        else:
            path = output_path
        return path

    def _get_halfhour_output_path(self, output_path=None):
        """This method will resolve correct model weight path"""
        if output_path is None:
            path = os.path.join(os.path.dirname(__file__),
                                "../data/inference_halfhour.csv")
        else:
            path = output_path
        return path

    def load_model(self):
        tsmodel = TSModel(N_steps=self.N_steps,
                          N_features=DataLoader.NUM_FEATURES,
                          lr=0.001)
        model = tsmodel.get_model()
        model.load_weights(self.model_weights_path)
        return model

    def inference(self):
        logger.info("Start inferencing")
        latest_time = self.data_loader.data_ori["event_time"].max()
        end = pd.Timestamp("2017-07-07")
        outputs = []
        times = []
        time_delta_min = - self.data_loader._min_scaled_time_delta
        while latest_time < end:
            logger.info(f"Current timestamp: {latest_time}")
            nn_input_data = self.data_loader.gen_x_y_by_index(
                len(self.data_loader.data_array), return_label=False)[0]
            nn_input_data = np.expand_dims(nn_input_data, axis=0)
            nn_output = self.model.predict(nn_input_data)[0]


            # time
            time_delta, x, y = nn_output
            time_delta = max(time_delta, time_delta_min)
            time_delta_scaled, x_scaled, y_scaled = self.data_loader.reverse_scale_data(
                time_delta, x, y)
            time_delta_scaled = max(0, time_delta_scaled)
            latest_time = latest_time + pd.Timedelta(time_delta_scaled, unit="s")
            times.append(latest_time)
            outputs.append({"event_time": latest_time,
                            "x": x_scaled,
                            "y":y_scaled})

            array = self.data_loader.construct_input_data(
                time=latest_time,
                x=x,
                y=y,
                time_delta=time_delta)

            array = np.expand_dims(array, axis=0)
            self.data_loader.data_array = np.concatenate(
                [self.data_loader.data_array, array], axis=0)

        df = pd.DataFrame(outputs)
        df.to_csv(self.output_path, index=False)
        return df

    def generate_halfhour_result(self):
        result = pd.read_csv(self.output_path)
        result["event_time"] = pd.to_datetime(result["event_time"])
        from_ = result["event_time"].min().floor("d")
        end_ = pd.Timestamp("2017-07-07")
        cur = from_
        data = []
        while cur < end_:
            to_ = cur + pd.Timedelta(1800, unit="s")
            res = self.query_result(cur, to_)[0]
            data_ = {"time_beginning": cur.strftime("%Y-%m-%d %H:%M:%S"),
                         "number_of_event": res}
            data.append(data_)
            logger.info(data_)
            cur = to_

        data = pd.DataFrame(data)
        data.to_csv(self.halfhour_output_path, index=False)
        return data

    def query_result(self, from_datetime:str, to_datetime:str):
        """
        This method will query inference data, and return the data that falls
        into the inverval
        :param from_datetime: str (YYYY-MM-DD HH:MM:SS),
            the beginning time. Inclusive.
        :param to_datetime: str (YYYY-MM-DD HH:MM:SS),
            the beginning time. Exclusive.
        :return: number of occurence, details
        """
        result = pd.read_csv(self.output_path)
        result["event_time"] = pd.to_datetime(result["event_time"])
        if isinstance(from_datetime, str):
            from_datetime = pd.Timestamp(from_datetime)
        if isinstance(to_datetime, str):
            to_datetime = pd.Timestamp(to_datetime)
        mask = (result["event_time"] < to_datetime) & (
                result["event_time"] >= from_datetime)
        result = result.loc[mask]
        return len(result), result


# if __name__ == "__main__":
#     infr = Inference(N_steps=300, model_weights_path=None)
#     # generate inference.csv
#     infr.inference()
#     # generate inference_halfhour.csv
#     infr.generate_halfhour_result()