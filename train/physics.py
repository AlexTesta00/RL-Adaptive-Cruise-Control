import numpy as np

def kmh_to_ms(speed):
    return int(speed // 3.6)


def map_sigmoid(x, k=0.1):
    return 1 / (1 + np.exp(-k * x))