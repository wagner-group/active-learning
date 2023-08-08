#! /usr/bin/env python

import xgboost as xgb
import numpy as np
from scipy import sparse

class xgboost_wrapper():
    def __init__(self, model, binary=False):
        self.model = model
        self.binary = binary
        print('binary classification: ',self.binary)

    def maybe_flat(self, input_data):
        if not isinstance(input_data,np.ndarray):
            input_data = np.copy(input_data.numpy())
        shape = input_data.shape
        if len(input_data.shape) == 1:
            input_data = np.copy(input_data[np.newaxis,:])
        if len(input_data.shape) >= 3:
            input_data = np.copy(input_data.reshape(shape[0],np.prod(shape[1:])))
        return input_data, shape

    def predict(self, input_data):
        input_data, _ = self.maybe_flat(input_data)
        ori_input = np.copy(input_data)
        input_data = xgb.DMatrix(sparse.csr_matrix(input_data))
        ori_input = xgb.DMatrix(sparse.csr_matrix(ori_input))
        test_predict = np.array(self.model.predict(input_data))
        if self.binary:
            test_predict = (test_predict > 0.5).astype(int)
        else:
            test_predict = test_predict.astype(int)
        return test_predict

    def predict_proba(self, input_data):
        input_data, _ = self.maybe_flat(input_data)
        input_back = np.copy(input_data)
        input_data = sparse.csr_matrix(input_data)
        input_data = xgb.DMatrix(input_data)
        test_predict = np.array(self.model.predict(input_data))
        return test_predict

    def predict_label(self, input_data):
        return self.predict(input_data)