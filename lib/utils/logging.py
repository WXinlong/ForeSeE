# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Utilities for logging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import deque
import logging
import numpy as np
import sys
from lib.core.config import cfg
def log_stats(stats, args):
    """Log training statistics to terminal"""
    lines = "[Step %d/%d] [Epoch %d/%d]  [%s]\n" % (
            stats['iter'], cfg.TRAIN.MAX_ITER, stats['epoch'], args.epoch[-1], args.dataset)

    lines += "\t\tloss: %.3f,    time: %.6f,    eta: %s\n" % (
        stats['total_loss'], stats['time'], stats['eta'] )

    for k in stats:
        if 'loss' in k and 'total_loss' not in k:
            lines += "\t\t" + ", ".join("%s: %.3f" % (k, v) for k, v in stats[k].items()) + ", "

    # validate criteria
    lines += "\t\t" + ",       ".join("%s: %.6f" % (k, v) for k, v in stats['val_err'].items()) + ", "
    lines += '\n'

    # lr in different groups
    lines += "\t\t" + ",       ".join("%s: %.6f" % (k, v) for k, v in stats['lr'].items()) + ", "
    lines += '\n'
    print(lines[:-1])  # remove last new line

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def AddValue(self, value, size=1):
        self.deque.append(value)
        self.series.append(value)
        self.count += size
        self.total += value

    def GetMedianValue(self):
        return np.median(self.deque)

    def GetAverageValue(self):
        return np.mean(self.deque)

    def GetGlobalAverageValue(self):
        return self.total / self.count


def setup_logging(name):
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(name)
    return logger

