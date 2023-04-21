import pandas as pd
import numpy as np
from hyppo.ksample import MMD

chat_id = 405993924 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    significance_level = 0.09
    pvalue = MMD(compute_kernel='rbf', gamma=1.0).test(x, y).pvalue
    return pvalue < significance_level
