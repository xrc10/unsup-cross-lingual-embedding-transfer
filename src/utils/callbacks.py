import math
from collections import defaultdict

class EarlyStopper():
    def __init__(self, patience, min_delta=1e-5):
        self.hist_loss = defaultdict(float)
        self.patience_cnt = 0
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')

    def check_early_stop(self, cur_loss, epoch):
        '''check if we should early stop'''
        self.hist_loss[epoch] = cur_loss
        # if epoch > 0 and self.hist_loss[epoch-1] - cur_loss > self.min_delta:
        if epoch > 0 and cur_loss < self.best_loss - self.min_delta:
            self.patience_cnt = 0
            self.best_loss = cur_loss
        else:
            self.patience_cnt += 1
        if self.patience_cnt > self.patience:
            print("early stopping...")
            return True
        else:
            return False

class ModelChecker():
    def __init__(self, min_delta=1e-5):
        self.best_loss = float('inf')
        self.epoch_loss = defaultdict(list)
        self.min_delta = min_delta

    def record_loss(self, cur_loss, epoch):
        self.epoch_loss[epoch].append(cur_loss)

    def get_best_loss(self):
        return self.best_loss

    def check_for_best(self, cur_loss, epoch):
        cur_loss = sum(self.epoch_loss[epoch]) / float(len(self.epoch_loss[epoch]))
        if epoch > 0 and cur_loss < self.best_loss - self.min_delta:
            self.best_loss = cur_loss
            return True
        else:
            return False

def SGDWR(T_total, T_0, T_mult, lr_max, lr_min): 
    # Use "STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS"
    # return a dictionary records at each epoch the learning rate: {<#epoch>:<lr>}
    lr_dict = {}
    T_cur = 0.0 # epochs since last restart
    T_i = float(T_0)
    for T in range(T_total+1):
        if T_cur > T_i:
            T_cur = 0.0
            T_i = T_i * T_mult
        lr = lr_min + 0.5*(lr_max-lr_min)*(1+math.cos(math.pi*(T_cur/T_i)))
        T_cur += 1
        lr_dict[T] = lr
    return lr_dict