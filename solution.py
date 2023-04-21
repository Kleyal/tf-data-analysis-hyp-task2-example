import pandas as pd
import numpy as np
from hyppo.ksample import KSample

chat_id = 405993924 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    significance_level = 0.09
    pvalue = KSample(indep_test='Hsic', compute_distkern='rbf', gamma=0.5).test(x, y).pvalue
    if pvalue < significance_level:
        return True
    else:
        return False
