import os
import tensorflow as tf
from logisds.model import TSModel
from logisds.data import DataLoader
from logisds.utils import get_console_logger

logger = get_console_logger("Train-RNN")

class Trainer:
    def __init__(self,
                 batch_size=100,
                 N_steps=150,
                 lr=0.001,
                 random_seed=1,
                 val_ratio=0.4,
                 early_stopping_round=5,
                 model_weights_path=None):
        self.batch_size = batch_size
        self.N_steps = N_steps
        self.val_ratio = val_ratio
        self.lr = lr
        self.early_stopping_round = early_stopping_round
        self.model_weights_path = self._get_model_weights_path(
            model_weights_path)
        self._init_data(random_seed)


    def _get_model_weights_path(self, model_weights_path=None):
        """This method will resolve correct model weight path"""
        if model_weights_path is None:
            path = os.path.join(os.path.dirname(__file__),
                                "../data/model")
        else:
            path = model_weights_path
        return path

    def _init_data(self, seed):
        """ This method will initialize data loader"""
        self.data_loader = DataLoader(
            seed=seed,
            N_steps=self.N_steps,
            val_ratio=self.val_ratio)
        self.data_loader.prepare_data()

    def train(self):
        tsmodel = TSModel(
            N_steps=self.N_steps,
            N_features=34,
            lr=self.lr)
        model = tsmodel.get_model()
        model.fit_generator(
            self.data_loader.data_gen(mode="train"),
            steps_per_epoch=len(
                self.data_loader.train_index) // self.batch_size,
            verbose=1,
            epochs=100,
            validation_data=self.data_loader.data_gen(
                mode="val"),
            validation_steps=len(
                self.data_loader.val_index) // self.batch_size,
            callbacks=tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_round,
                restore_best_weights=True),
        )
        model.save_weights(self.model_weights_path)
        logger.info(f"Model weights have been saved to: {self.model_weights_path}")
        return model


# if __name__ == "__main__":
#     trainer = Trainer(batch_size=100,
#                       N_steps=300,
#                       lr=0.001,
#                       random_seed=1,
#                       val_ratio=0.4,
#                       early_stopping_round=10,
#                       model_weights_path=None)
#     model = trainer.train()