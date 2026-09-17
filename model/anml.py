import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

# from model.resnet1d import ResNet1D
from model.anml_base import NeuromodNet1D  # as Learner
from random import shuffle
from torch.autograd import Variable
from model.resnet1d import ResNet1D
import random

# import model.learner as Learner
from utils.training_metrics import macro_recall
from utils.class_weighted_loss import classification_cross_entropy

logger = logging.getLogger("experiment")


@dataclass
class AnmlConfig:
    update_lr: float = 0.1
    meta_lr: float = 1e-3
    inner_steps: int = 10
    replay_batch_size: int = 20
    memories: int = 5120
    rln: int = 7
    grad_clip_norm: Optional[float] = 0.0
    use_old_task_memory: bool = False

    @staticmethod
    def from_args(args: object) -> "AnmlConfig":
        cfg = AnmlConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, n_inputs, n_outputs, n_tasks, args, neuromodulation=True):

        super(Net, self).__init__()
        self.args = args
        self.cfg = AnmlConfig.from_args(args)
        self.update_lr = self.cfg.update_lr
        self.meta_lr = self.cfg.meta_lr
        self.inner_steps = self.cfg.inner_steps

        # self.net = Learner(n_outputs, self.args, neuromodulation=neuromodulation)
        # self.net = Learner.Learner(config, neuromodulation)
        self.nm = NeuromodNet1D(in_ch=2, mask_dim=512, conv_ch=(64, 112, 112))
        self.backbone = ResNet1D(n_outputs, args, in_channels=2)
        self.optimizer = optim.Adam(self.parameters(), lr=self.meta_lr)
        self.meta_iteration = 0
        self.inputNM = True
        self.nodeNM = False
        self.layers_to_fix = []

        self.param_names = [n for n, _ in self.backbone.named_parameters()]
        self.feature_param_names = [
            n for n in self.param_names if not n.startswith("fc")
        ]
        self.classifier_param_names = [
            n for n in self.param_names if n.startswith("fc")
        ]
        self.feature_param_count = len(self.feature_param_names)

        self.epoch = 0
        self.current_task = 0
        self.batchSize = int(self.cfg.replay_batch_size)
        self.M = []
        self.M_new = []
        self.age = 0
        self.memories = self.cfg.memories
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))

    def reset_classifer(self, class_to_reset):
        bias = self.parameters()[-1]
        weight = self.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def reset_layer(self, layer_to_reset):
        if layer_to_reset % 2 == 0:
            weight = self.parameters()[layer_to_reset]  # -2]
            torch.nn.init.kaiming_normal_(weight)
        else:
            bias = self.parameters()[layer_to_reset]
            bias.data = torch.ones(bias.data.size())

    def add_patch_to_images(self, images, task_num):
        boxSize = 8
        if task_num == 1:
            try:
                images[:, :, : boxSize + 1, : boxSize + 1] = torch.min(images)
            except:
                images[:, : boxSize + 1, : boxSize + 1] = torch.min(images)
        elif task_num == 2:
            images[:, :, -(boxSize + 1) :, -(boxSize + 1) :] = torch.min(images)
        elif task_num == 3:
            images[:, :, : boxSize + 1, -(boxSize + 1) :] = torch.min(images)
        elif task_num == 4:
            images[:, :, -(boxSize + 1) :, : boxSize + 1] = torch.min(images)
        return images

    def shuffle_labels(self, targets, batch=False):
        if batch:
            new_target = (targets[0] + 2) % 1000
            for t in range(len(targets)):
                targets[t] = new_target

            return targets

        else:
            new_target = (targets + 2) % 1000
            return new_target

    def inner_update(self, x, fast_weights, y, bn_training):

        logits = self(x, fast_weights)
        loss = classification_cross_entropy(
            logits, y, class_weighted_ce=self.class_weighted_ce
        )

        if fast_weights is None:
            fast_weights = list(self.parameters())

        grads = torch.autograd.grad(
            loss, fast_weights, create_graph=True, allow_unused=True
        )

        new_fast_weights = []
        for g, w in zip(grads, fast_weights):
            g = torch.zeros_like(w) if g is None else g
            if getattr(w, "learn", True):
                nw = w - self.update_lr * g
            else:
                nw = w
            # preserve the custom flag on the new tensor
            nw.learn = getattr(w, "learn", True)
            new_fast_weights.append(nw)

        return new_fast_weights

    def meta_loss(self, x, fast_weights, y):

        logits = self(x, fast_weights)
        loss_q = classification_cross_entropy(
            logits, y, class_weighted_ce=self.class_weighted_ce
        )
        return loss_q, logits

    def eval_accuracy(self, logits, y):
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()
        return correct

    def forward(self, x, fast_weights):
        mask = self.nm(x)

        if fast_weights is None:
            feats = self.backbone.forward(x, ret_feats=True)
            feats = feats * mask
            logits = self.backbone.forward(feats, classify_feats=True)
        else:
            # partition fast weights into feature and classifier slices
            f_count = self.backbone.feature_param_count
            feature_fast = fast_weights[:f_count]
            classifier_fast = fast_weights[f_count:]

            feats = self.backbone.forward(x, vars=feature_fast, ret_feats=True)
            feats = feats * mask
            logits = self.backbone.forward(
                feats, vars=classifier_fast, classify_feats=True
            )
        return logits

    def freeze_layers(self, layers_to_freeze):

        for name, param in self.named_parameters():
            param.learn = True

        for name, param in self.nm.named_parameters():
            param.learn = False

        # for name, param in self.backbone.named_parameters():
        #     param.learn = True

        # frozen_layers = []
        # for temp in range(layers_to_freeze * 2):
        #     frozen_layers.append("net.vars." + str(temp))

        # for name, param in self.named_parameters():
        #     if name in frozen_layers:
        #         logger.info("RLN layer %s", str(name))
        #         param.learn = False

        # list_of_names = list(filter(lambda x: x[1].learn, self.named_parameters()))

        # for a in list_of_names:
        #     logger.info("TLN layer = %s", a[0])

    def observe(self, x, y, t):
        self.freeze_layers(self.cfg.rln)
        self.backbone.train()
        self.nm.train()

        cls_tr_rec = []

        for pass_itr in range(self.inner_steps):
            self.pass_itr = pass_itr

            # shuffle the data (again)
            perm = torch.randperm(x.size(0))
            x = x[perm]
            y = y[perm]

            self.epoch += 1
            self.zero_grads()

            if t != self.current_task:
                self.M = self.M_new
                self.current_task = t
            # create mini-batches from the main batch
            x_spt = self.make_minibatches(x, int(self.inner_steps))
            y_spt = self.make_minibatches(y, int(self.inner_steps))

            steps = min(self.inner_steps, x.size(0) // self.inner_steps)
            for i in range(steps):
                fast_weights = self.inner_update(
                    x_spt[i], None, y_spt[i], t
                )  # Update task-specific fast weights

            batch_sz = x.shape[0]
            # meta_losses = [0 for _ in range(batch_sz)]

            bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)
            fast_weights = None

            self.zero_grads()
            # meta-loss on the entire batch + memory
            for i in range(0, batch_sz):
                batch_x = x[i].unsqueeze(0)
                batch_y = y[i].unsqueeze(0)

                # if(self.real_epoch == 0):
                #     self.push_to_mem(batch_x, batch_y, torch.tensor(t))

                if self.real_epoch == 0:  # always true
                    with torch.no_grad():
                        self.push_to_mem(
                            batch_x.detach().cpu(),
                            batch_y.detach().cpu(),
                            torch.tensor(t),
                        )

                meta_loss, logits = self.meta_loss(
                    bx, fast_weights, by
                )  # loss on the meta batch
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    cls_tr_rec.append(macro_recall(preds, by))
                # meta_losses[i] += meta_loss
                assert meta_loss.requires_grad, "meta_loss has no grad path to alpha"
                (meta_loss / batch_sz).backward()

            # Taking the meta gradient step (will update the learning rates)
            # self.zero_grad()

            # meta_loss = sum(meta_losses)/len(meta_losses)

            # meta_loss.backward()

            if self.cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.cfg.grad_clip_norm
                )
            self.optimizer.step()

            # better zeroing (lower mem)
            self.zero_grad(set_to_none=True)
            # self.backbone.alpha_lr.zero_grad(set_to_none=True)

        avg_cls_tr_rec = sum(cls_tr_rec) / len(cls_tr_rec) if cls_tr_rec else 0.0
        return meta_loss.item(), avg_cls_tr_rec, None

    def push_to_mem(self, batch_x, batch_y, t):
        """
        Reservoir sampling to push subsampled stream
        of data points to replay/memory buffer
        """

        if self.real_epoch > 0 or self.pass_itr > 0:
            return
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M_new) < self.memories:
                self.M_new.append([batch_x[i], batch_y[i], t])
            else:
                p = random.randint(0, self.age)
                if p < self.memories:
                    self.M_new[p] = [batch_x[i], batch_y[i], t]

    def getBatch(self, x, y, t, batch_size=None):
        """
        Given the new data points, create a batch of old + new data,
        where old data is sampled from the memory buffer
        """

        if x is not None:
            mxi = np.array(x)
            myi = np.array(y)
            mti = np.ones(x.shape[0], dtype=int) * t
        else:
            mxi = np.empty(shape=(0, 0))
            myi = np.empty(shape=(0, 0))
            mti = np.empty(shape=(0, 0))

        bxs = []
        bys = []
        bts = []

        if self.cfg.use_old_task_memory and t > 0:
            MEM = self.M
        else:
            MEM = self.M_new

        batch_size = self.batchSize if batch_size is None else batch_size

        # Sample from memory buffer if not empty
        if len(MEM) > 0:
            order = [i for i in range(0, len(MEM))]
            osize = min(batch_size, len(MEM))
            for j in range(0, osize):

                # randomly sample from self.M_new memory buffer
                shuffle(order)
                k = order[j]
                x, y, t = MEM[k]

                xi = np.array(x)
                yi = np.array(y)
                ti = np.array(t)
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

        # add new data points - ratio of old to new becomes up to 1:1
        for j in range(len(myi)):
            bxs.append(mxi[j])
            bys.append(myi[j])
            bts.append(mti[j])

        bxs = Variable(torch.from_numpy(np.array(bxs))).float()
        bys = Variable(torch.from_numpy(np.array(bys))).long().view(-1)
        bts = Variable(torch.from_numpy(np.array(bts))).long().view(-1)

        # handle gpus if specified
        if self.use_cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()

        return bxs, bys, bts

    def zero_grads(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

    def make_minibatches(
        self, batch: torch.Tensor, mb_size: int, drop_last: bool = False
    ):
        """
        Split a batch [B, ...] into minibatches of size mb_size,
        return as a single tensor stacked with leading dim = num_minibatches.
        """
        B = batch.size(0)
        num_full = B // mb_size
        remainder = B % mb_size

        minibatches = [
            batch[i : i + mb_size] for i in range(0, num_full * mb_size, mb_size)
        ]

        if remainder > 0 and not drop_last:
            minibatches.append(batch[num_full * mb_size :])

        # Now stack into one tensor, padding if necessary
        if drop_last or remainder == 0:
            return torch.stack(minibatches)  # shape [num_batches, mb_size, ...]
        else:
            # pad last minibatch to mb_size along dim=0
            last = minibatches[-1]
            pad_size = mb_size - last.size(0)
            pad = last.new_zeros((pad_size, *last.shape[1:]))
            minibatches[-1] = torch.cat([last, pad], dim=0)
            return torch.stack(minibatches)
