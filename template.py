import torch
import os
import os.path as osp
import uuid

uuid = str(uuid.uuid1())[0:8]


class TemplateModel():

    def __init__(self):

        self.writer = None
        self.train_logger = None  # not neccessary
        self.eval_logger = None  # not neccessary
        self.args = None  # not neccessary
        self.accumulation_steps = 1

        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')
        self.best_accu = float('-Inf')

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.metric = None

        self.train_loader = None
        self.eval_loader = None

        self.device = None

        self.ckpt_dir = None
        self.display_freq = None
        self.scheduler = None
        self.mode = None
        # self.eval_per_epoch = None

    def check_init(self):
        assert self.model
        assert self.optimizer
        assert self.criterion
        assert self.metric
        assert self.train_loader
        assert self.eval_loader
        assert self.device
        assert self.ckpt_dir
        assert self.display_freq
        assert self.scheduler

        if not osp.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

    def load_state(self, fname, optim=True, map_location=None):
        state = torch.load(fname, map_location=map_location)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state['model'])
        else:
            self.model.load_state_dict(state['model'])

        if optim and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        self.step = state['step']
        self.epoch = state['epoch']
        self.best_error = state['best_error']
        print('load model from {}'.format(fname))

    def save_state(self, fname, optim=True):
        state = {}

        if isinstance(self.model, torch.nn.DataParallel):
            state['model'] = self.model.module.state_dict()
        else:
            state['model'] = self.model.state_dict()

        if optim:
            state['optimizer'] = self.optimizer.state_dict()
        state['step'] = self.step
        state['epoch'] = self.epoch
        state['best_error'] = self.best_error
        torch.save(state, fname)
        print('save model at {}'.format(fname))

    def train(self):
        self.model.train()
        self.epoch += 1
        for i, batch in enumerate(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss, others = self.train_loss(batch)
            loss.backward()
            if ((i + 1) % self.accumulation_steps) == 0:
                self.optimizer.step()
            if self.step % self.display_freq == 0:
                self.writer.add_scalar('loss_train_%s' % uuid, loss.item(), self.step)
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss))
                if self.train_logger:
                    self.train_logger(self.writer, others)
        torch.cuda.empty_cache()

    def train_loss(self, batch):
        x, y = batch['image'], batch['labels']

        pred = self.model(x)
        loss = self.criterion(pred, y)

        return loss, None

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            error, others = self.eval_error()

        if os.path.exists(self.ckpt_dir) is False:
            os.makedirs(self.ckpt_dir)

        if error < self.best_error:
            self.best_error = error
            self.save_state(osp.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(osp.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar('error_val%s' % uuid, error, self.epoch)
        print('epoch {}\terror {:.3}\tbest_error {:.3}'.format(self.epoch, error, self.best_error))

        if self.eval_logger:
            self.eval_logger(self.writer, others)

        torch.cuda.empty_cache()
        return error

    def eval_error(self):
        error = 0
        iter = 0
        for i,batch in enumerate(self.eval_loader):
            iter += 1
            x, y = batch
            pred = self.model(x)
            error += self.metric(pred, y).item()

        error /= iter

        return error, None

    def num_parameters(self):
        return sum([p.data.nelement() for p in self.model.parameters()])


class F1Accuracy(torch.nn.CrossEntropyLoss):
    def __init__(self,
                 weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'
                 ):
        super(F1Accuracy, self).__init__(weight, size_average, reduce, reduction)
        self.TP = 0.0
        self.TN = 0.0
        self.FP = 0.0
        self.FN = 0.0
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.F1 = 0.0

    def calc_accuracy(self, input, target):
        predict = torch.argmax(input, dim=1, keepdim=False)
        labels = torch.argmax(target, dim=1, keepdim=False)
        for i in range(input.shape[1]):
            self.TP += ((predict == i) * (labels == i)).sum().tolist()
            self.TN += ((predict != i) * (labels != i)).sum().tolist()
            self.FP += ((predict == i) * (labels != i)).sum().tolist()
            self.FN += ((predict != i) * (labels == i)).sum().tolist()
        # self.accuracy = (self.TP + self.TN) / \
        #                 (self.TP + self.TN + self.FP + self.FN)
        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)
        self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        return self.F1

    def forward(self, input, target):
        return self.calc_accuracy(input, target)

