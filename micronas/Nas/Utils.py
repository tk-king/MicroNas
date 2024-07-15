import numpy as np
from torch.utils.data import DataLoader
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
import torch
from math import ceil, floor
import random

import torch.nn.functional as F


def getPadding(kernel_size):
    return (ceil(kernel_size / 2), floor(kernel_size / 2) - 1)

def calcNumLayers(num, stride, thres):
    ctr = 0
    while ceil(num / stride) >= thres:
        num = ceil(num / stride)
        ctr += 1
    return ctr

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt



def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def weight_softmax(weights, eps, hard=False, gumbel=True):
  if gumbel:
    res = F.gumbel_softmax(weights, eps, hard=hard)
    return res
  else:
    res = F.softmax(weights / eps)
    return res


# def random_one_hot_like(input):
#     res = torch.zeros_like(input)
#     r_int = random.randint(0, len(input) - 1)
#     res[r_int] = 1
#     return res

def random_one_hot_like(input):
    if input.dim() == 1:
        res = torch.zeros_like(input)
        r_int = random.randint(0, len(input) - 1)
        res[r_int] = 1
    else:
        res = torch.zeros_like(input)
        for i in range(input.shape[0]):
            r_int = random.randint(0, input.shape[1] - 1)
            res[i, r_int] = 1
    return res