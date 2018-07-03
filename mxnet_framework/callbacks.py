import sys
import numpy as np
import mxnet as mx
import cv2
import logging
import time
from mxnet.model import save_checkpoint

class SpeedometerCustom(object):
    def __init__(self, batch_size, frequent=50, auto_reset=True):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()
                    msg = 'Epoch[%d] Batch[%d] Speed: %.2f samples/sec'
                    msg += ' %s=%f'*len(name_value)
                    logging.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                else:
                    logging.info("Iter[%d] Batch[%d] Speed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()

class BestModelCheckpoint(object):

    def __init__(self, prefix, metric_name, find_max=True):
        self._prefix = prefix
        self._metric_name = metric_name
        self._prev_value = None
        self._find_max = find_max

    def is_best(self, eval_metric, update_value=False):

        name_values = {name: val for name, val in eval_metric.get_name_value()}
        assert self._metric_name in name_values
        value = name_values[self._metric_name]

        if self._prev_value is None:
            # first time
            if update_value:
                self._prev_value = value
            return True

        if self._find_max:
            if value > self._prev_value:
                if update_value:
                    self._prev_value = value
                return True
            return False
        else:
            if value < self._prev_value:
                if update_value:
                    self._prev_value = value
                return True
            return False

    def get_best_value(self):
        return self._prev_value

    def checkpoint_if_only_best(self, eval_metric, sym, arg, aux):
        if self.is_best(eval_metric, update_value=True):
            save_checkpoint(self._prefix, 0, sym, arg, aux)
