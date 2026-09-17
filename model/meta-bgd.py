import random
from random import shuffle
import numpy as np
import ipdb
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch.autograd import Variable

from model.optimizers_lib import optimizers_lib
from ast import literal_eval
from model.resnet1d import ResNet1D
from utils.training_metrics import macro_recall
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy

"""
This baseline/ablation is constructed by merging C-MAML and BGD
By assigning a variance parameter to each NN parameter in the model
and using BGD's bayesian update to update these means (the NN parameters) and variances
(the learning rates in BGD are derived from the variances)

The 'n' bayesian samples in this case are the 'n' cumulative meta-losses sampled when 
C-MAML is run with 'n' different initial theta vectors as the NN means sampled from the 
(means, variances) stored for the model parameters.
The weight update is then carried out using the BGD formula that implicitly 
uses the variances to derive the learning rates for the parameters
"""


@dataclass
class MetaBgdConfig:
    arch: str = "resnet1d"
    n_layers: int = 2
    n_hiddens: int = 100
    alpha_init: float = 1e-3
    cuda: bool = True
    bgd_optimizer: str = "bgd"
    mean_eta: float = 1.0
    std_init: float = 5e-2
    train_mc_iters: int = 5
    optimizer_params: Sequence[str] = field(default_factory=lambda: ["{}"])
    dataset: str = "tinyimagenet"
    inner_steps: int = 1
    memories: int = 5120
    replay_batch_size: int = 20
    use_old_task_memory: bool = False
    grad_clip_norm: Optional[float] = 0.0
    meta_batches: int = 3
    cifar_batches: int = 1

    @staticmethod
    def from_args(args: object) -> "MetaBgdConfig":
        cfg = MetaBgdConfig()
        for field in cfg.__dataclass_fields__:
            if hasattr(args, field):
                setattr(cfg, field, getattr(args, field))
        return cfg


class Net(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Net, self).__init__()
        self.cfg = MetaBgdConfig.from_args(args)
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.incremental_loader_name = getattr(args, "loader", None)

        if self.cfg.arch != "resnet1d":
            raise ValueError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )
        self.net = ResNet1D(n_outputs, args)

        # define the lr params
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)

        self.use_cuda = self.cfg.cuda
        if self.use_cuda:
            self.net = self.net.cuda()

        # optimizer model
        self.bgd_optimizer = self.cfg.bgd_optimizer
        optimizer_model = optimizers_lib.__dict__[self.cfg.bgd_optimizer]
        # params used to instantiate the BGD optimiser
        opt_params_raw = self.cfg.optimizer_params
        if isinstance(opt_params_raw, str):
            opt_params_str = opt_params_raw
        elif isinstance(opt_params_raw, Sequence):
            opt_params_str = " ".join(opt_params_raw)
        else:
            opt_params_str = str(opt_params_raw)
        optimizer_params = dict(
            {  # "logger": logger,
                "mean_eta": self.cfg.mean_eta,
                "std_init": self.cfg.std_init,
                "mc_iters": self.cfg.train_mc_iters,
            },
            **literal_eval(opt_params_str),
        )
        self.optimizer = optimizer_model(self.net, **optimizer_params)

        self.epoch = 0
        # allocate buffer
        self.M = []
        self.M_new = []
        self.age = 0

        self.is_cifar = (self.cfg.dataset == "cifar100") or (
            self.cfg.dataset == "tinyimagenet"
        )
        self.inner_steps = self.cfg.inner_steps
        self.pass_itr = 0
        self.real_epoch = 0

        # setup memories
        self.current_task = 0

        self.memories = self.cfg.memories
        self.batchSize = int(self.cfg.replay_batch_size)

        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        # if self.is_cifar:
        #     self.nc_per_task = n_outputs / n_tasks
        # else:
        #     self.nc_per_task = n_outputs
        self.n_outputs = n_outputs
        self.is_task_incremental = True

        self.obseve_itr = 0

    def take_multitask_loss(self, bt, t, logits, y):
        loss = 0.0

        for i, ti in enumerate(bt):
            offset1, offset2 = self.compute_offsets(ti)
            loss += classification_cross_entropy(
                logits[i, offset1:offset2].unsqueeze(0),
                y[i].unsqueeze(0) - offset1,
                class_weighted_ce=self.class_weighted_ce,
            )
        return loss / len(bt)

    def forward(self, x, t, fast_weights=None, *, cil_all_seen_upto_task=None):
        if self.bgd_optimizer == "sgd":
            self.optimizer.randomize_weights(force_std=0)
        output = self.net.forward(x, vars=fast_weights)
        if self.is_task_incremental and (
            self.is_cifar or cil_all_seen_upto_task is not None
        ):
            output = misc_utils.apply_task_incremental_logit_mask(
                output,
                t,
                self.classes_per_task,
                self.n_outputs,
                cil_all_seen_upto_task=cil_all_seen_upto_task,
                fill_value=-10e10,
                loader=self.incremental_loader_name,
            )
        return output

    def meta_loss(self, x, fast_weights, y, bt, t):
        """
        differentiate the loss through the network updates wrt alpha
        """

        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            logits = self.net.forward(x, fast_weights)[:, :offset2]

            loss_q = self.take_multitask_loss(bt, t, logits, y)
        else:
            logits = self.net.forward(x, fast_weights)
            # Cross Entropy Loss over data
            loss_q = classification_cross_entropy(
                logits, y, class_weighted_ce=self.class_weighted_ce
            )
        return loss_q, logits

    def compute_offsets(self, task):
        if self.is_task_incremental:
            return misc_utils.compute_offsets(task, self.classes_per_task)
        else:
            return 0, self.n_outputs

    def push_to_mem(self, batch_x, batch_y, t):
        """
        Reservoir sampling memory update
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

    def getBatch(self, x, y, t):
        """
        Given the new data points, create a batch of old + new data,
        where old data is part of the memory buffer
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

        if self.cfg.use_old_task_memory:  # and t>0:
            MEM = self.M
        else:
            MEM = self.M_new

        if len(MEM) > 0:
            order = [i for i in range(0, len(MEM))]
            osize = min(self.batchSize, len(MEM))
            for j in range(0, osize):
                shuffle(order)
                k = order[j]
                x, y, t = MEM[k]

                xi = np.array(x)
                yi = np.array(y)
                ti = np.array(t)
                bxs.append(xi)
                bys.append(yi)
                bts.append(ti)

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

    def take_loss(self, t, logits, y):
        offset1, offset2 = self.compute_offsets(t)
        loss = classification_cross_entropy(
            logits[:, offset1:offset2],
            y - offset1,
            class_weighted_ce=self.class_weighted_ce,
        )

        return loss

    def inner_update(self, x, fast_weights, y, t):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            logits = self.net.forward(x, fast_weights)[:, :offset2]

            loss = self.take_loss(t, logits, y)
            # loss = self.loss(logits, y)
        else:
            logits = self.net.forward(x, fast_weights)
            loss = classification_cross_entropy(
                logits, y, class_weighted_ce=self.class_weighted_ce
            )

        if fast_weights is None:
            fast_weights = [p for p in self.net.parameters()]
        else:
            fast_weights = list(fast_weights)

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        graph_required = True
        grads = list(
            torch.autograd.grad(
                loss,
                fast_weights,
                create_graph=graph_required,
                retain_graph=graph_required,
            )
        )

        for i in range(len(grads)):
            if self.cfg.grad_clip_norm:
                grads[i] = torch.clamp(
                    grads[i], min=-self.cfg.grad_clip_norm, max=self.cfg.grad_clip_norm
                )

        # get fast weights vector by taking SGD step on grads
        fast_weights = list(
            map(
                lambda p: p[1][0] - p[0] * p[1][1],
                zip(grads, zip(fast_weights, self.net.alpha_lr)),
            )
        )
        return fast_weights

    def observe(self, x, y, t):
        self.net.train()
        self.obseve_itr += 1

        if self.bgd_optimizer == "bgd":
            num_of_mc_iters = self.optimizer.get_mc_iters()
        else:
            num_of_mc_iters = 1

        train_acc_values = []

        for glance_itr in range(self.inner_steps):

            mc_meta_losses = [0 for _ in range(num_of_mc_iters)]

            # running C-MAML num_of_mc_iters times to get montecarlo samples of meta-loss
            for pass_itr in range(num_of_mc_iters):
                if self.bgd_optimizer == "bgd":
                    self.optimizer.randomize_weights()

                self.pass_itr = pass_itr
                self.epoch += 1
                self.net.zero_grad()

                perm = torch.randperm(x.size(0))
                x = x[perm]
                y = y[perm]

                if pass_itr == 0 and glance_itr == 0 and t != self.current_task:
                    self.M = self.M_new
                    self.current_task = t

                batch_sz = x.shape[0]

                n_batches = self.cfg.cifar_batches
                rough_sz = math.ceil(batch_sz / n_batches)

                # the samples of new task to iterate over in inner update trajectory
                iterate_till = 1  # batch_sz
                meta_losses = [0 for _ in range(n_batches)]
                accuracy_meta_set = [0 for _ in range(n_batches)]

                # put some asserts to make sure replay batch size can accomodate old and new samples
                bx, by = None, None
                bx, by, bt = self.getBatch(x.cpu().numpy(), y.cpu().numpy(), t)

                fast_weights = None
                # inner loop/fast updates where learn on 1-2 samples in each inner step
                for i in range(n_batches):

                    batch_x = x[i * rough_sz : (i + 1) * rough_sz]
                    batch_y = y[i * rough_sz : (i + 1) * rough_sz]
                    fast_weights = self.inner_update(batch_x, fast_weights, batch_y, t)

                    if pass_itr == 0 and glance_itr == 0:
                        self.push_to_mem(batch_x, batch_y, torch.tensor(t))

                    # the meta loss is computed at each inner step
                    # as this is shown to work better in Reptile []
                    meta_loss, logits = self.meta_loss(bx, fast_weights, by, bt, t)
                    with torch.no_grad():
                        if self.is_cifar:
                            preds_list = []
                            target_list = []
                            for sample_idx, task_idx in enumerate(bt):
                                offset1, offset2 = self.compute_offsets(task_idx)
                                preds = torch.argmax(
                                    logits[sample_idx, offset1:offset2], dim=0
                                )
                                target = by[sample_idx] - offset1
                                preds_list.append(preds.detach().cpu())
                                target_list.append(target.detach().cpu())
                            if preds_list:
                                stacked_preds = torch.stack(preds_list).view(-1)
                                stacked_targets = torch.stack(target_list).view(-1)
                                acc = macro_recall(stacked_preds, stacked_targets)
                            else:
                                acc = 0.0
                        else:
                            preds = torch.argmax(logits, dim=1)
                            acc = macro_recall(preds, by)
                    accuracy_meta_set[i] = acc
                    meta_losses[i] += meta_loss

                self.optimizer.zero_grad()
                meta_loss = sum(meta_losses) / len(meta_losses)
                if torch.isnan(meta_loss):
                    ipdb.set_trace()
                meta_loss.backward()
                if self.cfg.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.cfg.grad_clip_norm
                    )
                mc_meta_losses[pass_itr] = meta_loss
                if accuracy_meta_set:
                    train_acc_values.append(
                        sum(accuracy_meta_set) / len(accuracy_meta_set)
                    )
                if self.bgd_optimizer == "bgd":
                    self.optimizer.aggregate_grads(batch_size=batch_sz)

            print_std = False
            if self.obseve_itr % 220 == 0:
                print_std = True
            if self.bgd_optimizer == "bgd":
                self.optimizer.step(print_std=print_std)
            else:
                self.optimizer.step()

        meta_loss_return = sum(mc_meta_losses) / len(mc_meta_losses)
        avg_cls_tr_rec = (
            sum(train_acc_values) / len(train_acc_values) if train_acc_values else 0.0
        )

        return meta_loss_return.item(), avg_cls_tr_rec, None
