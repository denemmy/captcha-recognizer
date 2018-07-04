#!/usr/bin/env python3.4

import argparse
import numpy as np
import re
from os.path import basename, dirname, splitext, join

from bokeh.plotting import figure, show, output_file
from bokeh.embed import file_html
from bokeh.resources import CDN
TOOLS = "pan,wheel_zoom,reset,save,box_select"

FORMAT_INT = "[-+]?\d+"
FORMAT_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Plot val-metric curve')
    parser.add_argument('--log', dest='log_path',
                        help='path to train log file', type=str, required=True)
    parser.add_argument('--output', dest='output', type=str, required=True)

    args = parser.parse_args()
    return args

def make_sparse(iters, values, max_points=100):

    num_points = len(values)
    ratio = float(num_points) / max_points
    step = int(round(ratio))
    if step > 1:
        last_it = iters[-1]
        last_val = values[-1]

        iters = iters[:-1:step]
        values = values[:-1:step]

        iters = np.append(iters, last_it)
        values = np.append(values, last_val)

    return iters, values

def moving_average(x, y, n=3):
    yn = np.cumsum(y, dtype=float)
    yn = (yn[n:] - yn[:-n]) / float(n)
    halfn = int(n/2)
    xn = x[halfn:-(n - halfn)]
    return xn, yn

def parse_log(log_path):

    train_data = []
    val_data = []
    checkpoint_data = []

    epoch_size = None
    last_epoch = None
    last_batch = None

    with open(log_path, 'r') as f:
        for line in f.readlines():

            if epoch_size is None:
                m = re.match(r'epoch size: ({})'.format(FORMAT_INT), line)
                if m:
                    epoch_size = int(m.group(1))
                    continue

            frm_str = r'.*Epoch\[({})\] Batch\[({})\] Speed: {} samples/sec (.*)'.format(FORMAT_INT,
                                                                                         FORMAT_INT, FORMAT_FLOAT)
            m = re.match(frm_str, line)
            if m:
                epoch, batch, values_raw = m.group(1, 2, 3)
                epoch = int(epoch)
                batch = int(batch)

                values = {}
                parts_raw = values_raw.split()
                for p_raw in parts_raw:
                    parts = p_raw.split('=')
                    name, value = parts[0], float(parts[1])
                    values[name] = value

                train_data.append((epoch, batch, values))
                last_epoch = epoch
                last_batch = batch
                continue

            frm_str = r'.*Epoch\[({})\] Train-(.*)'.format(FORMAT_INT)
            m = re.match(frm_str, line)
            if m:
                epoch, value_raw = m.group(1, 2)
                epoch = int(epoch)
                parts = value_raw.split('=')
                name, value = parts[0], float(parts[1])
                values = {}
                values[name] = value

                train_data.append((epoch, 0, values))
                last_epoch = epoch
                last_batch = 0
                continue

            frm_str = r'.*Epoch\[({})\] Batch\[({})\] Validation-(.*)'.format(FORMAT_INT, FORMAT_INT)
            m = re.match(frm_str, line)
            if m:
                epoch, batch, value_raw = m.group(1, 2, 3)
                epoch = int(epoch)
                batch = int(batch)

                parts = value_raw.split('=')
                name, value = parts[0], float(parts[1])
                values = {}
                values[name] = value

                val_data.append((epoch, batch, values))
                last_epoch = epoch
                last_batch = batch
                continue

            frm_str = r'.*Epoch\[({})\] Validation-(.*)'.format(FORMAT_INT)
            m = re.match(frm_str, line)
            if m:
                epoch, value_raw = m.group(1, 2)
                epoch = int(epoch)
                parts = value_raw.split('=')
                name, value = parts[0], float(parts[1])
                values = {}
                values[name] = value

                val_data.append((epoch, 0, values))
                last_epoch = epoch
                last_batch = 0
                continue

            frm_str = r'.*Saved checkpoint to "(.*)"'.format(FORMAT_INT)
            m = re.match(frm_str, line)
            if m:
                checkpoint_path = m.group(1)
                checkpoint_name = basename(checkpoint_path)
                checkpoint_data.append((last_epoch, last_batch, checkpoint_name))
                continue

    if epoch_size is None:
        epochs, batches, iters_values = zip(*train_data)
        epoch_size = max(batches)

    if epoch_size == 0:
        epoch_size = 1.0

    train_data_split = {}
    for epoch, batch, values in train_data:
        for name in values:
            if name not in train_data_split:
                train_data_split[name] = []
            value = values[name]
            iter = epoch + float(batch) / epoch_size
            train_data_split[name].append((iter, value))

    val_data_split = {}
    for epoch, batch, values in val_data:
        for name in values:
            if name not in val_data_split:
                val_data_split[name] = []
            value = values[name]
            iter = epoch + float(batch) / epoch_size
            val_data_split[name].append((iter, value))

    checkpoint_data_i = []
    for epoch, batch, name in checkpoint_data:
        iter = epoch + float(batch) / epoch_size
        checkpoint_data_i.append((iter, name))

    return train_data_split, val_data_split, checkpoint_data_i

def plot_data(train_data_split, val_data_split, checkpoint_data_i, output_base):

    union_keys = set(train_data_split.keys()).union(set(val_data_split.keys()))

    for name in union_keys:

        print('making {} plot..'.format(name))

        p1 = figure(width=1200, height=600, title='Test/Train {}'.format(name), tools=TOOLS, output_backend='webgl')

        if name in train_data_split:
            train_iters, train_values = zip(*train_data_split[name])
            train_iters = list(train_iters)
            train_values = list(train_values)

            train_iters, train_values = make_sparse(train_iters, train_values)

            if len(train_iters) < 50:
                p1.line(train_iters, train_values, line_width=2., line_color='#df5050', legend='Train {}'.format(name))
            else:
                p1.line(train_iters, train_values, line_width=1., line_color='#eb9191',
                        legend='Train {}'.format(name))
                train_iters_avg, train_values_avg = moving_average(train_iters, train_values, 10)
                p1.line(train_iters_avg, train_values_avg, line_width=2., line_color='#df5050',
                        legend='Train {} (mean)'.format(name))

        if name in val_data_split:
            val_iters, val_values = zip(*val_data_split[name])
            val_iters = list(val_iters)
            val_values = list(val_values)

            # val_iters, val_values = make_sparse(val_iters, val_values)
            #

            if len(val_iters) < 50:
                p1.line(val_iters, val_values, line_width=2., line_color='#94a259', legend='Validation {}'.format(name))
            else:
                p1.line(val_iters, val_values, line_width=1., line_color='#b4be89',
                        legend='Validation {}'.format(name))
                val_iters_avg, val_values_avg = moving_average(val_iters, val_values, 10)
                p1.line(val_iters_avg, val_values_avg, line_width=2., line_color='#94a259',
                        legend='Validation {} (mean)'.format(name))

        p1.ygrid[0].ticker.desired_num_ticks = 30
        p1.xgrid[0].ticker.desired_num_ticks = 20

        p1.legend.location = 'bottom_right'

        html = file_html(p1, CDN, title=log_path)

        output_file = output_base + '_{}.html'.format(name)
        with open(output_file, 'w') as f:
            f.write(html)

def plot(log_path, output_path):
    train_data_split, val_data_split, checkpoint_data_i = parse_log(log_path)
    output_base, output_ext = splitext(output_path)
    plot_data(train_data_split, val_data_split, checkpoint_data_i, output_base)

if __name__ == '__main__':

    args = parse_args()

    log_path = args.log_path
    output_path = args.output
    plot(log_path, output_path)