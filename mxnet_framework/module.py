from mxnet.module.module import Module
from mxnet.initializer import Uniform
from mxnet import metric
from mxnet.model import BatchEndParam
from mxnet.base import _as_list
from mxnet.module.base_module import _check_input_names
import mxnet.context as ctx
import mxnet as mx
import copy

import time
import logging

class CustomModule(Module):

    def __init__(self, *args, **kwargs):
        super(CustomModule, self).__init__(*args, **kwargs)

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            best_model_callbacks=None,
            eval_interval=None,
            validation_metric=None, monitor=None):

        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)

        if monitor is not None:
            self.install_monitor(monitor)

        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)

        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if not isinstance(eval_metric, metric.EvalMetric):
            eval_metric = metric.create(eval_metric)

        if validation_metric is None:
            validation_metric = copy.deepcopy(eval_metric)

        epoch_metric = copy.deepcopy(eval_metric)

        swa_arg_params = None
        swa_aux_params = None
        swa_cnt = 0

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic_epoch = time.time()
            eval_metric.reset()

            nbatch = 0
            end_of_batch = False
            data_iter = iter(train_data)
            next_data_batch = next(data_iter)
            name_values = []

            while not end_of_batch:
                data_batch = next_data_batch

                if monitor is not None:
                    monitor.tic()

                self.forward_backward(data_batch)
                self.update()

                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True

                self.update_metric(eval_metric, data_batch.label)
                if end_of_batch:
                    name_values = eval_metric.get_name_value()

                if monitor is not None:
                    monitor.toc_print()

                nbatch += 1

                if batch_end_callback is not None:
                    batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                     eval_metric=eval_metric,
                                                     locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)

                    eval_metric.reset()

                # ----------------------------------------
                # evaluation on validation set
                to_go = eval_interval is not None and nbatch % eval_interval == 0
                if to_go and eval_data:
                    res = self.score(eval_data, validation_metric,
                                     score_end_callback=eval_end_callback,
                                     batch_end_callback=eval_batch_end_callback, epoch=epoch)
                    for name, val in res:
                        self.logger.info('Epoch[%d] Batch[%d] Validation-%s=%f', epoch, nbatch, name, val)

                    if best_model_callbacks is not None:
                        for callback in _as_list(best_model_callbacks):
                            if callback.is_best(validation_metric):
                                # sync aux params across devices
                                arg_params, aux_params = self.get_params()
                                sync_made = True
                                callback.checkpoint_if_only_best(validation_metric, self.symbol, arg_params, aux_params)
                                break

            # one epoch of training is finished
            for name, val in name_values:
                self.logger.info('Epoch[%d] Train-%s=%f', epoch + 1, name, val)
            toc_epoch = time.time()
            elapsed = (toc_epoch - tic_epoch)
            avg_speed = float(len(train_data)) / (toc_epoch - tic_epoch)
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch + 1, elapsed)
            self.logger.info('Epoch[%d] Average speed=%.3f samples/sec', epoch + 1, avg_speed)

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            # evaluation on validation set
            if eval_data:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch + 1)
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch + 1, name, val)

                if best_model_callbacks is not None:
                    for callback in _as_list(best_model_callbacks):
                        callback.checkpoint_if_only_best(validation_metric, self.symbol, arg_params, aux_params)

            # end of epoch, reset the data-iter for another epoch
            train_data.reset()