import sys
import numpy as np
import mxnet as mx
import mxnet.metric as mxmetric
import cv2
import itertools
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score

def check_label_shapes(labels, preds, wrap=False, shape=False):
    """Helper function for checking shape of label and prediction
    Parameters
    ----------
    labels : list of `NDArray`
        The labels of the data.
    preds : list of `NDArray`
        Predicted values.
    wrap : boolean
        If True, wrap labels/preds in a list if they are single NDArray
    shape : boolean
        If True, check the shape of labels and preds;
        Otherwise only check their length.
    """
    if not shape:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

    if wrap:
        if isinstance(labels, mx.ndarray.NDArray):
            labels = [labels]
        if isinstance(preds, mx.ndarray.NDArray):
            preds = [preds]

    return labels, preds

def find_prediction(preds, max_labels, min_labels):

    preds_ch = preds[:-1]
    batch_size = preds_ch[0].shape[0]

    # find max prob for each character
    maxprob_v = np.zeros((batch_size, len(preds_ch)), dtype=np.float32)
    maxprob_arg = np.zeros((batch_size, len(preds_ch)), dtype=np.int32)
    for i in range(len(preds_ch)):
        prob_i = preds_ch[i].asnumpy()
        maxprob_i = np.amax(prob_i, axis=1)
        maxprob_arg_i = np.argmax(prob_i, axis=1)
        maxprob_v[:, i] = maxprob_i
        maxprob_arg[:, i] = maxprob_arg_i
    # compute log prob for this
    eps = 1e-12
    maxlogprob_v = np.log(maxprob_v + eps)

    logprob_values = np.zeros((batch_size, max_labels - min_labels + 1), dtype=np.float32)
    accum_logprob = np.zeros((batch_size,), dtype=np.float32)
    for i in range(min_labels):
        accum_logprob += maxlogprob_v[:, i]

    preds_num = preds[-1].asnumpy()
    logprob_num = np.log(preds_num + eps)

    logprob_values[:, 0] = accum_logprob + logprob_num[:, 0]

    for i in range(max_labels - min_labels):
        accum_logprob += maxlogprob_v[:, i + min_labels]
        logprob_values[:, i+1] = accum_logprob + logprob_num[:, i+1]

    number_of_ch = np.argmax(logprob_values, axis=1).astype(np.int32) + min_labels
    result = []
    for i in range(batch_size):
        result.append(list(maxprob_arg[i,:number_of_ch[i]]))
    return result, number_of_ch

class CaptchaAccuracyVarLen(mx.metric.EvalMetric):
    """Computes segmentation metrics.
    """
    def __init__(self, cfg, output_names=None, label_names=None):
        super(CaptchaAccuracyVarLen, self).__init__(
            'CaptchaAccuracyVarLen', axis=1,
            output_names=output_names, label_names=label_names)
        self._cfg = cfg

    def reset(self):
        self._good = 0
        self._total = 0
        self._good_captchas = 0
        self._total_captchas = 0
        self._good_ch = 0
        self._total_ch = 0

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        cfg = self._cfg
        assert len(labels) == len(preds)

        result, number_of_ch = find_prediction(preds, self._cfg.MAX_LABELS, self._cfg.MIN_LABELS)

        labels_len = labels[-1].asnumpy().astype(np.int32) + self._cfg.MIN_LABELS
        self._good += np.count_nonzero(number_of_ch == labels_len)
        self._total += number_of_ch.shape[0]

        corr = 0
        corr_c = 0
        total_c = 0
        for i in range(len(result)):
            if labels_len[i] != number_of_ch[i]:
                continue
            all_corr = True
            total_c += labels_len[i]
            for j in range(labels_len[i]):
                if labels[j][i] != result[i][j]:
                    all_corr = False
                else:
                    corr_c += 1
            if all_corr:
                corr += 1

        self._good_captchas += corr
        self._total_captchas += len(result)

        self._good_ch += corr_c
        self._total_ch += total_c

        #
        # pred_descr = pred_descr.asnumpy()  # (n, descriptor len)
        # label = label.asnumpy().astype('int32')  # (n, num_attributes)
        #
        # assert(label.shape[0] == pred_descr.shape[0])
        # assert(pred_descr.shape[1] == cfg.DESC_LEN)
        #
        # if self._descr_data is None:
        #     self._descr_data = pred_descr
        #     self._labels = label
        # else:
        #     self._descr_data = np.append(self._descr_data, pred_descr, axis=0)
        #     self._labels = np.append(self._labels, label, axis=0)


    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """

        acc = 0.0
        if self._total_captchas > 0:
            acc = float(self._good_captchas) / self._total_captchas

        acc_c = 0.0
        if self._total_ch > 0:
            acc_c = float(self._good_ch) / self._total_ch

        len_acc = 0.0
        if self._total > 0:
            len_acc = float(self._good) / self._total

        names = ['accuracy', 'character-accuracy', 'length-accuracy']
        values = [acc, acc_c, len_acc]

        return names, values

class CaptchaAccuracy(mx.metric.EvalMetric):
    """Computes segmentation metrics.
    """
    def __init__(self, cfg, output_names=None, label_names=None):
        super(CaptchaAccuracy, self).__init__(
            'CaptchaAccuracy', axis=1,
            output_names=output_names, label_names=label_names)
        self._cfg = cfg

    def reset(self):
        self._good = 0
        self._total = 0
        self._good_captchas = 0
        self._total_captchas = 0
        self._good_ch = 0
        self._total_ch = 0

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        cfg = self._cfg
        assert len(labels) == len(preds)

        preds_ch = preds
        batch_size = preds_ch[0].shape[0]

        # find max prob for each character
        pred_labels = np.zeros((batch_size, len(preds)), dtype=np.int32)
        for i in range(len(preds)):
            prob_i = preds[i].asnumpy()
            maxprob_arg_i = np.argmax(prob_i, axis=1)
            pred_labels[:, i] = maxprob_arg_i.astype(np.int32)

        label_m = np.zeros((batch_size, len(labels)), dtype=np.float32)
        for i in range(len(labels)):
            curr_labels = labels[i].asnumpy()
            label_m[:, i] = curr_labels.astype(np.int32)

        self._good_ch += np.count_nonzero(pred_labels == label_m)
        self._total_ch += pred_labels.size

        mask = np.sum(pred_labels == label_m, axis=1)
        self._good_captchas += np.count_nonzero(mask == pred_labels.shape[1])
        self._total_captchas += mask.shape[0]


    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """

        acc = 0.0
        if self._total_captchas > 0:
            acc = float(self._good_captchas) / self._total_captchas

        acc_c = 0.0
        if self._total_ch > 0:
            acc_c = float(self._good_ch) / self._total_ch

        names = ['accuracy', 'character-accuracy']
        values = [acc, acc_c]

        return names, values
