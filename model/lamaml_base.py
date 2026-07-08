import random
import torch
from dataclasses import dataclass
from typing import Optional
from model.resnet1d import ResNet1D
from model.detection_replay import noise_label_from_args, unpack_y_to_class_labels
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class LamamlBaseConfig:
    alpha_init: float = 1e-3
    opt_wt: float = 1e-1
    opt_lr: float = 1e-1
    inner_steps: int = 1
    memories: int = 5120
    replay_batch_size: int = 20
    cuda: bool = True
    use_old_task_memory: bool = False
    learn_lr: bool = False
    second_order: bool = False
    sync_update: bool = False
    momentum: float = 0.0
    meta_batches: int = 3
    arch: str = "resnet1d"
    dataset: str = "tinyimagenet"
    grad_clip_norm: Optional[float] = 2.0
    n_layers: int = 2
    n_hiddens: int = 100
    input_channels: int = 1

    @staticmethod
    def from_args(args: object) -> "LamamlBaseConfig":
        """Build config from CLI / merged args.

        La-MAML uses a single ``inner_steps`` count (total observe passes). For
        backward compatibility with YAML that set both ``inner_steps`` and the
        global ``n_meta`` (formerly a separate outer loop), the effective value is
        ``inner_steps * n_meta`` (each factor defaults to 1).
        """
        cfg = LamamlBaseConfig()
        inner = int(getattr(args, "inner_steps", 1) or 1)
        n_meta_legacy = int(getattr(args, "n_meta", 1) or 1)
        merged_inner = max(1, inner * n_meta_legacy)
        for field_name in cfg.__dataclass_fields__:
            if field_name == "inner_steps":
                setattr(cfg, field_name, merged_inner)
                continue
            if hasattr(args, field_name):
                setattr(cfg, field_name, getattr(args, field_name))
        return cfg


class BaseNet(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(BaseNet, self).__init__()

        self.args = args
        self.incremental_loader_name = getattr(args, "loader", None)
        self.class_weighted_ce = bool(getattr(args, "class_weighted_ce", True))
        self.cfg = LamamlBaseConfig.from_args(args)
        if self.cfg.arch != "resnet1d":
            raise ValueError(
                f"Unsupported arch {self.cfg.arch}; only resnet1d is available now."
            )
        self.net = ResNet1D(n_outputs, args)
        self.net.define_task_lr_params(alpha_init=self.cfg.alpha_init)

        self.opt_wt = torch.optim.SGD(
            list(self.net.parameters()), lr=self.cfg.opt_wt, momentum=0.9
        )
        self.opt_lr = torch.optim.SGD(
            list(self.net.alpha_lr.parameters()), lr=self.cfg.opt_lr, momentum=0.9
        )

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

        self.current_task = 0
        self.memories = self.cfg.memories
        self.batchSize = int(self.cfg.replay_batch_size)

        self.use_cuda = self.cfg.cuda
        if self.use_cuda:
            self.net = self.net.cuda()

        self._reset_velocity()

        self.n_outputs = n_outputs
        self.classes_per_task = misc_utils.build_task_class_list(
            n_tasks,
            n_outputs,
            nc_per_task=getattr(args, "nc_per_task_list", "")
            or getattr(args, "nc_per_task", None),
            classes_per_task=getattr(args, "classes_per_task", None),
        )
        self.nc_per_task = misc_utils.max_task_class_count(self.classes_per_task)
        self.noise_label: int | None = noise_label_from_args(args)

    def _classification_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return classification_cross_entropy(
            logits, targets, class_weighted_ce=self.class_weighted_ce
        )

    def push_to_mem(self, batch_x, batch_y, t):
        """
        Reservoir sampling to push subsampled stream
        of data points to replay/memory buffer
        """

        if self.real_epoch > 0 or self.pass_itr > 0:
            return
        batch_x = batch_x.cpu()
        batch_y = unpack_y_to_class_labels(batch_y).long().cpu()
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

        Sampling is uniform with replacement, identical in distribution to the
        original ``shuffle(order); k = order[j]`` draw, but ``random.choices``
        avoids the O(len(MEM) * osize) reshuffling. Replay tensors (already
        tensors in the buffer) are stacked directly and the current batch is
        concatenated on-device, removing the per-element numpy round-trip.
        """

        if self.cfg.use_old_task_memory and t > 0:
            MEM = self.M
        else:
            MEM = self.M_new

        batch_size = self.batchSize if batch_size is None else batch_size

        replay_x = []
        replay_y = []
        replay_t = []
        # Sample from memory buffer if not empty
        if len(MEM) > 0:
            osize = min(batch_size, len(MEM))
            for k in random.choices(range(len(MEM)), k=osize):
                mx, my, mt = MEM[k]
                replay_x.append(torch.as_tensor(mx))
                replay_y.append(int(torch.as_tensor(my).long().flatten()[0].item()))
                replay_t.append(int(torch.as_tensor(mt).long().flatten()[0].item()))

        parts_x, parts_y, parts_t = [], [], []
        if replay_x:
            parts_x.append(torch.stack(replay_x).float())
            parts_y.append(torch.tensor(replay_y, dtype=torch.long))
            parts_t.append(torch.tensor(replay_t, dtype=torch.long))

        # add new data points - ratio of old to new becomes up to 1:1
        if x is not None:
            cur_x = torch.as_tensor(x).float()
            n_cur = cur_x.shape[0]
            parts_x.append(cur_x)
            parts_y.append(torch.as_tensor(y).long().reshape(-1))
            parts_t.append(torch.full((n_cur,), int(t), dtype=torch.long))

        bxs = torch.cat([p.cpu() for p in parts_x], dim=0).float()
        bys = torch.cat([p.cpu() for p in parts_y], dim=0).long().view(-1)
        bts = torch.cat([p.cpu() for p in parts_t], dim=0).long().view(-1)

        # handle gpus if specified
        if self.use_cuda:
            bxs = bxs.cuda()
            bys = bys.cuda()
            bts = bts.cuda()

        return bxs, bys, bts

    def compute_offsets(self, task):
        # mapping from classes [1-100] to their idx within a task
        offset1, offset2 = misc_utils.compute_offsets(task, self.classes_per_task)
        return int(offset1), int(offset2)

    def _reset_velocity(self) -> None:
        n = sum(1 for _ in self.net.parameters())
        self._velocity: list = [None] * n

    def _async_weight_update(self) -> None:
        """Apply the La-MAML per-parameter weight update with optional momentum.

        Each parameter is updated as:
            v_j <- momentum * v_j + g_j
            theta_j <- theta_j - relu(alpha_j) * v_j
        where g_j is the current gradient and v_j is the running velocity.
        When momentum is 0 this reduces to the plain gradient step.
        """
        mu = self.cfg.momentum
        with torch.no_grad():
            for i, p in enumerate(self.net.parameters()):
                g = p.grad
                if g is None:
                    continue
                if mu > 0.0:
                    v = self._velocity[i]
                    self._velocity[i] = g if v is None else v.mul_(mu).add_(g)
                    g = self._velocity[i]
                lr_i = torch.relu(self.net.alpha_lr[i])
                p.add_(-lr_i * g)

    def zero_grads(self):
        if self.cfg.learn_lr:
            self.opt_lr.zero_grad()
        self.opt_wt.zero_grad()
        self.net.zero_grad()
        self.net.alpha_lr.zero_grad()
