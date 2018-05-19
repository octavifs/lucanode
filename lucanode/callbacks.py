import pandas as pd
from keras.callbacks import History


class HistoryLog(History):
    """A history callback that will record the loss onto a log each time the batch ends"""
    def __init__(self, logfile):
        super().__init__()
        self.logfile = logfile

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        pd.DataFrame(self.history).to_csv(self.logfile)
