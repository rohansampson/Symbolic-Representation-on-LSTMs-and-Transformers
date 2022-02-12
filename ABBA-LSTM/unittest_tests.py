# Code originally by https://github.com/nla-group
# My contribution is refine, fix errors and to attempt symbolic representation

import unittest
from forecaster import forecaster
from batchless_VanillaLSTM_pytorch import batchless_VanillaLSTM_pytorch
# from batchless_VanillaLSTM_keras import batchless_VanillaLSTM_keras
# from VanillaLSTM_keras import VanillaLSTM_keras
from ABBA import ABBA as ABBA
import numpy as np
import time

class test_LSTM(unittest.TestCase):

    ##################################################
    # batchless_VanillaLSTM_pytorch
    ##################################################
    def test_VanillaLSTM_stateful_numeric_pytorch(self):
        first_start = time.perf_counter()
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_pytorch(stateful=True), abba=None)
        f.train(max_epoch=500, patience=50)
        first_stop = time.perf_counter()
        elapsed_time = first_start - first_stop
        elapsed_time *= -1
        print("test_VanillaLSTM_stateful_numeric_pytorch")
        print("Training time: ")
        print(elapsed_time)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_VanillaLSTM_stateless_numeric_pytorch(self):
        second_start = time.perf_counter()
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_pytorch(stateful=False), abba=None)
        f.train(max_epoch=500, patience=50)
        second_stop = time.perf_counter()
        elapsed_time = second_start - second_stop
        elapsed_time *= -1
        print("test_VanillaLSTM_stateless_numeric_pytorch")
        print("Training time: ")
        print(elapsed_time)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_VanillaLSTM_stateful_symbolic_pytorch(self):
        third_start = time.perf_counter()
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_pytorch(stateful=True), abba=ABBA(max_len=2, verbose=0))
        f.train(max_epoch=500, patience=50)
        third_stop = time.perf_counter()
        elapsed_time = third_start - third_stop
        elapsed_time *= -1
        print("test_VanillaLSTM_stateful_symbolic_pytorch")
        print("Training time: ")
        print(elapsed_time)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_VanillaLSTM_stateless_symbolic_pytorch(self):
        fourth_start = time.perf_counter()
        time_series = [1, 2, 3, 2]*100 + [1]
        k = 10
        f = forecaster(time_series, model=batchless_VanillaLSTM_pytorch(stateful=False), abba=ABBA(max_len=2, verbose=0))
        f.train(max_epoch=500, patience=50)
        fourth_stop = time.perf_counter()
        elapsed_time = fourth_start - elapsed_time
        elapsed_time *= -1
        print("test_VanillaLSTM_stateless_symbolic_pytorch")
        print("Training time: ")
        print(elapsed_time)
        prediction = f.forecast(k).tolist()
        prediction = [round(p) for p in prediction]
        print(prediction)
        self.assertTrue(prediction == [2, 3, 2, 1, 2, 3, 2, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
