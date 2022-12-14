import torch.optim as optim
from utils.tools import *
from utils.metric_function import *


class Trainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, device, cl=True):
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl

    def train(self, e, train_x, train_x_mark, dec_inp, train_dec_mark, train_y, scaler):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(e, train_x, train_x_mark, dec_inp, train_dec_mark)
        real = train_y
        predict = scaler(output)
        # nzeroindex  = (real<0).nonzero()
        # for index in nzeroindex:
        #     print(index,real[index[0],index[1],index[2],index[3]])

        if self.iter % self.step == 0 and self.task_level <= self.seq_out_len:
            self.task_level += 1
        if self.cl:
            loss = self.loss(predict[:, :self.task_level, :, :], real[:, :self.task_level, :, :], torch.inf)
        else:
            loss = self.loss(predict, real, torch.inf)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        # mae = util.masked_mae(predict,real,0.0).item()
        mape = masked_mape(predict, real, 0.0).item()
        rmse = masked_rmse(predict, real, 0.0).item()
        self.iter += 1
        return loss.item(), mape, rmse

    def eval(self, e, train_x, train_x_mark, dec_inp, train_dec_mark, train_y, scaler):
        self.model.eval()
        output = self.model(e, train_x, train_x_mark, dec_inp, train_dec_mark)
        real = train_y
        predict = scaler(output)
        loss = self.loss(predict, real, torch.nan)
        mape = masked_mape(predict, real, 0.0).item()
        rmse = masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse


class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        # for param in self.params:
        #     grad_norm += math.pow(param.grad.data.norm(), 2)
        #
        # grad_norm = math.sqrt(grad_norm)
        # if grad_norm > 0:
        #     shrinkage = self.max_grad_norm / grad_norm
        # else:
        #     shrinkage = 1.
        #
        # for param in self.params:
        #     if shrinkage < 1:
        #         param.grad.data.mul_(shrinkage)
        self.optimizer.step()
        return grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        # only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()
