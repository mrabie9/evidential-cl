import random
import torch
from dataclasses import dataclass
from typing import Optional
from model.resnet1d import ResNet1D
from model.replay_utils import unpack_y_to_class_labels
from utils import misc_utils
from utils.class_weighted_loss import classification_cross_entropy


@dataclass
class LamamlBaseConfig:
    alpha_init: float = 1e-3
    opt_wt: float = 1e-1
    opt_lr: float = 1e-1
    inner_steps: int = 1
    memories: int = 5120
    use_ring_buffer: bool = False
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
    grad_clip_norm: Optional[float] = 0.0
    n_layers: int = 2
    n_hiddens: int = 100
    input_channels: int = 1
    # Accepted for old launch scripts only. It split the meta-loss forward into
    # separate replay/current passes so BatchNorm does not mix their statistics;
    # meta_loss now always does that, so the flag no longer changes anything.
    cmaml_joint_er: bool = False
    # How the meta loss combines replay and current rows. "split" (default) is
    # eralg4's ``current + memory_loss_lambda * replay`` -- two separately
    # normalized CEs, so replay's share of the loss is pinned at 1:1. "pooled" is
    # the legacy single CE over the concatenated batch, whose inverse-frequency
    # class weights are computed over that pooled batch: old-task classes are rare
    # there, so replay's share escalates with task count (0.34 at task 0 to 0.85
    # by task 9) and starves the current task. "split_norm" divides "split" by
    # ``1 + memory_loss_lambda`` to pin the share without doubling the loss scale.
    # See docs/cmaml_vs_reser_til.md.
    cmaml_replay_loss_mode: str = "split"
    # Weight on the replay term of the meta loss, matching eralg4's
    # ``current_loss + memory_loss_lambda * replay_loss``.
    memory_loss_lambda: float = 1.0

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

        # Replay-buffer storage strategy: reservoir sampling (default) or a
        # per-task ring buffer that overwrites its own oldest slot (FIFO). The
        # ring buffer splits the total ``memories`` budget evenly across tasks.
        self.n_tasks = int(n_tasks)
        self.use_ring_buffer = bool(self.cfg.use_ring_buffer)
        self.mem_per_task = max(1, self.memories // max(1, self.n_tasks))
        self._ring_positions: dict[int, list[int]] = {}
        self._ring_write_ptr: dict[int, int] = {}

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

    def _classification_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return classification_cross_entropy(
            logits, targets, class_weighted_ce=self.class_weighted_ce
        )

    def push_to_mem(self, batch_x, batch_y, t):
        """Push a subsampled stream of data points into the replay buffer.

        Dispatches to reservoir sampling (default) or a per-task ring buffer
        depending on ``use_ring_buffer``. Both strategies write into the flat
        ``self.M_new`` list consumed by :meth:`getBatch`.
        """

        if self.real_epoch > 0 or self.pass_itr > 0:
            return
        batch_x = batch_x.cpu()
        batch_y = unpack_y_to_class_labels(batch_y).long().cpu()
        t = t.cpu()

        if self.use_ring_buffer:
            self._push_ring_buffer(batch_x, batch_y, t)
        else:
            self._push_reservoir(batch_x, batch_y, t)

    def _push_reservoir(
        self, batch_x: torch.Tensor, batch_y: torch.Tensor, t: torch.Tensor
    ) -> None:
        """Reservoir-sample the incoming stream into a single fixed-size buffer.

        Every example seen so far is retained with equal probability, so the
        buffer approximates a uniform sample over the whole task stream.
        """
        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M_new) < self.memories:
                self.M_new.append([batch_x[i], batch_y[i], t])
            else:
                p = random.randint(0, self.age)
                if p < self.memories:
                    self.M_new[p] = [batch_x[i], batch_y[i], t]

    def _push_ring_buffer(
        self, batch_x: torch.Tensor, batch_y: torch.Tensor, t: torch.Tensor
    ) -> None:
        """Write the incoming stream into a per-task FIFO ring buffer.

        Each task owns ``self.mem_per_task`` slots inside the flat ``M_new``
        list. Slots fill sequentially, then the task's own write pointer wraps
        around and overwrites its oldest exemplar, keeping the most recent
        ``mem_per_task`` examples for that task.
        """
        task_key = int(t)
        capacity = self.mem_per_task
        task_positions = self._ring_positions.setdefault(task_key, [])
        for i in range(batch_x.shape[0]):
            self.age += 1
            sample = [batch_x[i], batch_y[i], t]
            if len(task_positions) < capacity:
                task_positions.append(len(self.M_new))
                self.M_new.append(sample)
            else:
                write_slot = self._ring_write_ptr.get(task_key, 0)
                self.M_new[task_positions[write_slot]] = sample
                self._ring_write_ptr[task_key] = (write_slot + 1) % capacity

    def getBatch(self, x, y, t, batch_size=None):
        """
        Given the new data points, create a batch of old + new data,
        where old data is sampled from the memory buffer

        Sampling is uniform with replacement, identical in distribution to the
        original ``shuffle(order); k = order[j]`` draw, but ``random.choices``
        avoids the O(len(MEM) * osize) reshuffling. Replay tensors (already
        tensors in the buffer) are stacked directly and the current batch is
        concatenated on-device, removing the per-element numpy round-trip.

        With ``use_old_task_memory`` the pool is ``self.M``, the snapshot taken
        at the last task boundary, so replay never contains rows from the task
        being trained. ``self.M`` is empty during task 0, which leaves that task
        on its own data alone and makes its accuracy comparable with the gated
        replay baselines (``er_ring``, ``agem``, ``gem``). Without the flag the
        pool is the live ``self.M_new``, which replays the current task's own
        rows from its second batch onwards.
        """

        if self.cfg.use_old_task_memory:
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
