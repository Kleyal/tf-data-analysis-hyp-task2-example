import pandas as pd
import numpy as np
from hyppo.ksample import MMD

chat_id = 405993924 # Ваш chat ID, не меняйте название переменной

def solution(x, y):
    res = MMD(compute_kernel='rbf', gamma=1.0).test(x, y)
    return res[1] < 0.09
