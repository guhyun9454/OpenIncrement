import argparse
import copy
import datetime
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
 
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
 
from continual_datasets.build_incremental_scenario import build_continual_dataloader
from continual_datasets.dataset_utils import RandomSampleWrapper, get_ood_dataset, set_data_config
from losses import SupConLoss
from angle_similar import RKAngle
from util import TwoCropTransform
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
 
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
 
def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.element_size() * t.nelement())
 
 
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Return top-k accuracies in percentage (0~100)."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
 
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
 
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
 
 
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
 
 
def save_accuracy_heatmap(acc_matrix_upper: np.ndarray, task_id: int, args) -> str:
    """Save accuracy matrix heatmap image and return its path."""
    ensure_dir(args.output_dir)
    fig = plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(acc_matrix_upper, cmap="viridis", vmin=0, vmax=100)
    plt.colorbar()
    plt.title(f"Accuracy Matrix (till task {task_id+1})")
    plt.xlabel("Task")
    plt.ylabel("Task")
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, f"acc_heatmap_task{task_id+1}.png")
    plt.savefig(out_path)
    plt.close(fig)
    return out_path
 
 
def save_anomaly_histogram(id_scores: np.ndarray, ood_scores: np.ndarray, args, suffix: str, task_id: Optional[int]) -> str:
    ensure_dir(args.output_dir)
    fig = plt.figure(figsize=(6, 4), dpi=150)
    plt.hist(id_scores, bins=50, alpha=0.6, label="ID")
    plt.hist(ood_scores, bins=50, alpha=0.6, label="OOD")
    plt.legend()
    plt.title(f"Anomaly Score Histogram ({suffix})" + (f" | task {task_id+1}" if task_id is not None else ""))
    plt.tight_layout()
    out_path = os.path.join(
        args.output_dir, f"anomaly_hist_{suffix}" + (f"_task{task_id+1}" if task_id is not None else "") + ".png"
    )
    plt.savefig(out_path)
    plt.close(fig)
    return out_path
 
 
def _is_dataset(obj) -> bool:
    return isinstance(obj, torch.utils.data.Dataset)
 
 
def _set_transform_inplace(ds, transform) -> None:
    """Recursively set `.transform` for nested datasets (Subset/ConcatDataset/UnknownWrapper/etc.)."""
    if isinstance(ds, torch.utils.data.Subset):
        _set_transform_inplace(ds.dataset, transform)
        return
    if isinstance(ds, torch.utils.data.ConcatDataset):
        for child in ds.datasets:
            _set_transform_inplace(child, transform)
        return
    # wrappers in continual_datasets.dataset_utils
    if hasattr(ds, "dataset") and _is_dataset(getattr(ds, "dataset")):
        _set_transform_inplace(ds.dataset, transform)
        return
    if hasattr(ds, "transform"):
        ds.transform = transform
 
 
def _shallow_clone_with_transform(ds, transform):
    """Create a shallow copy of dataset and override its transform without mutating shared parents."""
    if isinstance(ds, torch.utils.data.Subset):
        base = _shallow_clone_with_transform(ds.dataset, transform)
        return torch.utils.data.Subset(base, ds.indices)
    if isinstance(ds, torch.utils.data.ConcatDataset):
        return torch.utils.data.ConcatDataset([_shallow_clone_with_transform(d, transform) for d in ds.datasets])
    if hasattr(ds, "dataset") and _is_dataset(getattr(ds, "dataset")):
        new_ds = copy.copy(ds)
        new_ds.dataset = _shallow_clone_with_transform(ds.dataset, transform)
        return new_ds
    new_ds = copy.copy(ds)
    if hasattr(new_ds, "transform"):
        new_ds.transform = transform
    return new_ds
 
 
# ---------------------------------------------------------------------------
# OpenIncrement model (ViT encoder + projection head + inlier classifier)
# ---------------------------------------------------------------------------
 
 
class OpenIncrementNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        proj_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = getattr(self.encoder, "num_features", None)
        if feat_dim is None:
            # fallback: infer once
            feat_dim = 768
        self.feat_dim = int(feat_dim)
 
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, proj_dim),
        )
        self.classifier = nn.Linear(self.feat_dim, num_classes)
 
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
 
    def forward_projected(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x)
        z = F.normalize(self.projector(feats), dim=1)
        return z, feats
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.classifier(feats)
 
 
# ---------------------------------------------------------------------------
# Relation-based Knowledge Distillation (distance-wise)
# ---------------------------------------------------------------------------
 
 
class RKDistance(nn.Module):
    def forward(self, last: torch.Tensor, current: torch.Tensor, label: torch.Tensor, old_targets: Sequence[int]) -> torch.Tensor:
        """
        last/current: (2*bsz, dim) where first bsz is view1, second bsz is view2
        label: (bsz,)
        """
        device = current.device
        bsz = label.shape[0]
        label_np = label.detach().cpu().numpy()
        old_set = set(old_targets)
        m = [i for i, l in enumerate(label_np) if int(l) in old_set]
        if len(m) <= 1:
            return torch.zeros((), device=device)
 
        f1_c, f2_c = torch.split(current, [bsz, bsz], dim=0)
        f1_t, f2_t = torch.split(last, [bsz, bsz], dim=0)
        t = torch.cat([f1_t[m, :], f2_t[m, :]], dim=0)
        s = torch.cat([f1_c[m, :], f2_c[m, :]], dim=0)
 
        with torch.no_grad():
            d_t = torch.cdist(t, t, p=2).view(-1)
            d_t = d_t / (d_t.mean().clamp_min(1e-6))
        d_s = torch.cdist(s, s, p=2).view(-1)
        d_s = d_s / (d_s.mean().clamp_min(1e-6))
        return F.smooth_l1_loss(d_s, d_t, reduction="mean")
 
 
# ---------------------------------------------------------------------------
# Replay buffer (byte-budgeted)
# ---------------------------------------------------------------------------
 
 
class ByteReplayBuffer:
    def __init__(self, budget_bytes: int, device: torch.device, seed: int = 0):
        self.budget_bytes = int(budget_bytes)
        self.device = device
        self.seed = int(seed)
        self.inputs: Optional[torch.Tensor] = None  # CPU tensor [N,C,H,W]
        self.targets: Optional[torch.Tensor] = None  # CPU tensor [N]
 
    def __len__(self) -> int:
        if self.targets is None:
            return 0
        return int(self.targets.numel())
 
    def bytes_used(self) -> int:
        if self.inputs is None or self.targets is None:
            return 0
        return tensor_bytes(self.inputs) + tensor_bytes(self.targets)
 
    @torch.no_grad()
    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Replay buffer is empty – cannot sample.")
        n = min(int(n_samples), len(self))
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + int(time.time()) % 100000)
        idx = torch.randperm(len(self), generator=g)[:n]
        x = self.inputs[idx].to(self.device, non_blocking=True)
        y = self.targets[idx].to(self.device, non_blocking=True)
        return x, y
 
    def get_all_cpu(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.inputs is None or self.targets is None:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        return self.inputs, self.targets
 
    def rebuild_from_class_to_examples(
        self,
        class_to_examples: Dict[int, List[torch.Tensor]],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        xs: List[torch.Tensor] = []
        ys: List[int] = []
        for c, exs in class_to_examples.items():
            for x in exs:
                xs.append(x.to(dtype=dtype, device="cpu"))
                ys.append(int(c))
        if len(xs) == 0:
            self.inputs = None
            self.targets = None
            return
        self.inputs = torch.stack(xs, dim=0).contiguous()
        self.targets = torch.tensor(ys, dtype=torch.long, device="cpu").contiguous()
 
        # Hard safety: if still exceeds budget, truncate
        while self.bytes_used() > self.budget_bytes and len(self) > 0:
            self.inputs = self.inputs[:-1]
            self.targets = self.targets[:-1]
 
 
# ---------------------------------------------------------------------------
# OOD scoring (used by evaluate_ood). To keep evaluate_ood code identical shape,
# we keep a simple global context container.
# ---------------------------------------------------------------------------
 
 
GLOBAL_OOD_CONTEXT: Dict[str, torch.Tensor] = {}
SUPPORTED_METHODS = ["OPENINCREMENT_KNN"]
 
 
@torch.no_grad()
def compute_ood_scores(method: str, model: OpenIncrementNet, id_loader, ood_loader, device: torch.device):
    method = method.upper()
    model.eval()
 
    if method != "OPENINCREMENT_KNN":
        raise ValueError(f"지원되지 않는 OOD 메소드: {method}. 지원되는 메소드: {SUPPORTED_METHODS}")

    if "exemplar_feats" not in GLOBAL_OOD_CONTEXT or "exemplar_labels" not in GLOBAL_OOD_CONTEXT:
        raise RuntimeError("OPENINCREMENT_KNN requires exemplar bank; call refresh_exemplar_bank() first.")
    ex_feats = GLOBAL_OOD_CONTEXT["exemplar_feats"].to(device, non_blocking=True)
    ex_labels = GLOBAL_OOD_CONTEXT["exemplar_labels"].to(device, non_blocking=True)
    num_classes = int(GLOBAL_OOD_CONTEXT.get("num_classes", torch.tensor(0)).item())
    if num_classes <= 0 and ex_labels.numel() > 0:
        num_classes = int(ex_labels.max().item()) + 1
    k_fixed = int(GLOBAL_OOD_CONTEXT.get("knn_k", torch.tensor(10)).item())
 
    def score_batch(inputs: torch.Tensor) -> torch.Tensor:
        if ex_feats.numel() == 0:
            return torch.zeros(inputs.size(0), device=device)
        feats = model.forward_features(inputs)
        feats = F.normalize(feats, dim=1)
        sims = feats @ ex_feats.t()  # (B, N)
        k = min(k_fixed, sims.size(1))
        topk_sims, topk_idx = sims.topk(k=k, dim=1)
        topk_labels = ex_labels[topk_idx]  # (B, k)
        per_class_sum = torch.zeros((inputs.size(0), num_classes), device=device)
        per_class_sum.scatter_add_(1, topk_labels, topk_sims)
        total = topk_sims.sum(dim=1).clamp_min(1e-6)
        return per_class_sum.max(dim=1).values / total
 
    id_scores_all: List[torch.Tensor] = []
    ood_scores_all: List[torch.Tensor] = []
 
    for (x_id, _), (x_ood, _) in zip(id_loader, ood_loader):
        x_id = x_id.to(device, non_blocking=True)
        x_ood = x_ood.to(device, non_blocking=True)
        id_scores_all.append(score_batch(x_id).detach().cpu())
        ood_scores_all.append(score_batch(x_ood).detach().cpu())
 
    return torch.cat(id_scores_all, dim=0), torch.cat(ood_scores_all, dim=0)
 
 
# ---------------------------------------------------------------------------
# Training / Evaluation (OODVIL evaluation logic shape)
# ---------------------------------------------------------------------------
 
 
@dataclass
class SeenState:
    seen_classes: Set[int]
 
 
class OpenIncrementRunner:
    def __init__(self, model: OpenIncrementNet, device: torch.device, args, eval_transform):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.eval_transform = eval_transform
 
        self.supcon = SupConLoss(temperature=args.temp)
        self.rk_angle = RKAngle()
        self.rk_dist = RKDistance()
 
        # train encoder + projector only
        self.optimizer = torch.optim.AdamW(
            list(self.model.encoder.parameters()) + list(self.model.projector.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.teacher: Optional[OpenIncrementNet] = None
        self.buffer = ByteReplayBuffer(args.replay_buffer_bytes, device=device, seed=args.seed)
 
    def _make_teacher(self) -> OpenIncrementNet:
        teacher = copy.deepcopy(self.model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher
 
    @torch.no_grad()
    def refresh_exemplar_bank(self, num_classes: int) -> None:
        x_cpu, y_cpu = self.buffer.get_all_cpu()
        if x_cpu.numel() == 0:
            GLOBAL_OOD_CONTEXT["exemplar_feats"] = torch.empty(0)
            GLOBAL_OOD_CONTEXT["exemplar_labels"] = torch.empty(0, dtype=torch.long)
            GLOBAL_OOD_CONTEXT["num_classes"] = torch.tensor(int(num_classes))
            GLOBAL_OOD_CONTEXT["knn_k"] = torch.tensor(int(self.args.knn_k))
            return
 
        feats_all: List[torch.Tensor] = []
        bs = max(1, int(self.args.batch_size))
        for i in range(0, x_cpu.size(0), bs):
            xb = x_cpu[i : i + bs].to(self.device, non_blocking=True)
            fb = self.model.forward_features(xb)
            fb = F.normalize(fb, dim=1)
            feats_all.append(fb.detach().cpu())
        feats = torch.cat(feats_all, dim=0)
 
        GLOBAL_OOD_CONTEXT["exemplar_feats"] = feats
        GLOBAL_OOD_CONTEXT["exemplar_labels"] = y_cpu.clone()
        GLOBAL_OOD_CONTEXT["num_classes"] = torch.tensor(int(num_classes))
        GLOBAL_OOD_CONTEXT["knn_k"] = torch.tensor(int(self.args.knn_k))
 
    def train_one_epoch(
        self,
        train_loader,
        epoch: int,
        old_targets: Sequence[int],
    ) -> Tuple[float, float]:
        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()
 
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
 
        old_set = set(int(x) for x in old_targets)
        for batch_idx, (images, targets) in enumerate(train_loader):
            # develop: 1 epoch 1 batch
            if self.args.develop and batch_idx > 0:
                break
 
            if isinstance(images, (list, tuple)) and len(images) == 2:
                x1, x2 = images[0], images[1]
            else:
                x1, x2 = images, images
 
            x1 = x1.to(self.device, non_blocking=True)
            x2 = x2.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
 
            # replay (byte-budgeted buffer, sample count is separate hyperparam)
            if self.args.replay_batch_size > 0 and len(self.buffer) > 0:
                x_rep, y_rep = self.buffer.sample(self.args.replay_batch_size)
                x1 = torch.cat([x1, x_rep], dim=0)
                x2 = torch.cat([x2, x_rep], dim=0)  # 동일 view (간단 구현)
                targets = torch.cat([targets, y_rep], dim=0)
 
            bsz = targets.size(0)
            x_cat = torch.cat([x1, x2], dim=0)
 
            z, feats = self.model.forward_projected(x_cat)
            z1, z2 = torch.split(z, [bsz, bsz], dim=0)
            feats1, _feats2 = torch.split(feats, [bsz, bsz], dim=0)
 
            features = torch.stack([z1, z2], dim=1)  # (bsz, 2, dim)
            loss_supcon = self.supcon(features, targets)
 
            loss_angle = torch.zeros((), device=self.device)
            loss_dist = torch.zeros((), device=self.device)
            if self.teacher is not None and len(old_set) > 0:
                # select old-class samples only, and optionally sub-sample for cubic RKAngle cost
                old_mask = torch.tensor([int(t.item()) in old_set for t in targets], device=self.device, dtype=torch.bool)
                old_idx = old_mask.nonzero(as_tuple=False).view(-1)
                if old_idx.numel() > 0:
                    if old_idx.numel() > self.args.distill_batch_size:
                        perm = torch.randperm(old_idx.numel(), device=self.device)[: self.args.distill_batch_size]
                        old_idx = old_idx[perm]
 
                    # teacher embeddings for selected subset
                    x_sub = torch.cat([x1[old_idx], x2[old_idx]], dim=0)
                    with torch.no_grad():
                        z_t, _ = self.teacher.forward_projected(x_sub)
 
                    # student embeddings for selected subset
                    z_s = torch.cat([z1[old_idx], z2[old_idx]], dim=0)
                    y_sub = targets[old_idx]
 
                    loss_angle = self.rk_angle(z_t, z_s, y_sub, list(old_set))
                    loss_dist = self.rk_dist(z_t, z_s, y_sub, list(old_set))
 
            loss_dis = loss_angle + self.args.lambda_dis * loss_dist
            loss = self.args.alpha * loss_supcon + (1.0 - self.args.alpha) * loss_dis
 
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.encoder.parameters()) + list(self.model.projector.parameters()),
                    max_norm=self.args.grad_clip_norm,
                )
            self.optimizer.step()
 
            # logging stats
            with torch.no_grad():
                logits = self.model.classifier(feats1)
                acc1 = accuracy(logits, targets, topk=(1,))[0]
 
            batch_size = int(targets.size(0))
            total_loss += float(loss.item()) * batch_size
            total_acc += float(acc1.item()) * batch_size
            total_samples += batch_size
 
            if batch_idx % self.args.print_freq == 0:
                running_avg_loss = total_loss / max(1, total_samples)
                running_avg_acc = total_acc / max(1, total_samples)
                print(
                    f"Epoch {epoch+1}, Batch [{batch_idx}/{len(train_loader)}]: "
                    f"Running Avg Loss = {running_avg_loss:.4f}, Running Avg Acc@1 = {running_avg_acc:.2f}"
                )
 
        return (total_loss / max(1, total_samples)), (total_acc / max(1, total_samples))
 
    @torch.no_grad()
    def _compute_bytes_per_sample(self, sample_x: torch.Tensor, sample_y: torch.Tensor) -> int:
        return tensor_bytes(sample_x) + tensor_bytes(sample_y)
 
    @torch.no_grad()
    def update_replay_buffer_isometric(self, train_dataset, seen_classes: Sequence[int]) -> None:
        """
        Isometric sampling: per-class distance-sorted indices, then equally-spaced picks.
        Memory is controlled by args.replay_buffer_bytes (Total Bytes).
        """
        if self.args.replay_buffer_bytes <= 0:
            return
        if len(seen_classes) == 0:
            return
 
        # bytes/sample estimation
        # Use one sample from dataset clone (single view) to avoid TwoCropTransform list outputs.
        single_view_ds = _shallow_clone_with_transform(train_dataset, self.eval_transform)
        probe_x, probe_y = single_view_ds[0]
        if isinstance(probe_x, (list, tuple)):
            probe_x = probe_x[0]
        probe_y = torch.tensor(int(probe_y), dtype=torch.long)
        bytes_per_sample = self._compute_bytes_per_sample(probe_x, probe_y)
 
        per_class_budget = self.args.replay_buffer_bytes // (len(seen_classes) * bytes_per_sample)
        n_per_class = int(per_class_budget)
        if n_per_class <= 0:
            # budget too small to keep balanced exemplars
            self.buffer.inputs = None
            self.buffer.targets = None
            return
 
        # start from existing exemplars (truncate to n_per_class)
        class_to_examples: Dict[int, List[torch.Tensor]] = {int(c): [] for c in seen_classes}
        if len(self.buffer) > 0:
            x_cpu, y_cpu = self.buffer.get_all_cpu()
            for c in seen_classes:
                idx = (y_cpu == int(c)).nonzero(as_tuple=False).view(-1)
                if idx.numel() == 0:
                    continue
                take = min(int(idx.numel()), n_per_class)
                # deterministic subset
                class_to_examples[int(c)].extend([x_cpu[i].clone() for i in idx[:take].tolist()])
 
        # fill missing classes (or top-up) using memory-safe reservoir candidates (store indices + features only)
        needs: List[int] = [int(c) for c in seen_classes if len(class_to_examples[int(c)]) < n_per_class]
        if len(needs) > 0:
            needs_set = set(needs)
 
            class CandidateIndexedDataset(torch.utils.data.Dataset):
                def __init__(self, base):
                    self.base = base
 
                def __len__(self):
                    return len(self.base)
 
                def __getitem__(self, idx):
                    x, y = self.base[idx]
                    return x, y, idx
 
            indexed_ds = CandidateIndexedDataset(single_view_ds)
            loader = torch.utils.data.DataLoader(
                indexed_ds,
                batch_size=self.args.exemplar_batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
            )
 
            max_cands = int(self.args.exemplar_candidates_per_class)
            # reservoir state per class
            cand_idx: Dict[int, List[int]] = {c: [] for c in needs}
            cand_feat: Dict[int, List[torch.Tensor]] = {c: [] for c in needs}
            seen_count: Dict[int, int] = {c: 0 for c in needs}
 
            self.model.eval()
            for batch_idx, (x, y, idx) in enumerate(loader):
                # develop: exemplar 후보도 1 batch만
                if self.args.develop and batch_idx > 0:
                    break
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                feats = self.model.forward_features(x)
                feats = F.normalize(feats, dim=1).detach().cpu()
                y_cpu = y.detach().cpu()
                idx_cpu = idx.detach().cpu()
 
                for i in range(x.size(0)):
                    c = int(y_cpu[i].item())
                    if c not in needs_set:
                        continue
                    seen_count[c] += 1
                    if len(cand_idx[c]) < max_cands:
                        cand_idx[c].append(int(idx_cpu[i].item()))
                        cand_feat[c].append(feats[i])
                    else:
                        j = random.randint(0, seen_count[c] - 1)
                        if j < max_cands:
                            cand_idx[c][j] = int(idx_cpu[i].item())
                            cand_feat[c][j] = feats[i]
 
            # isometric pick on reservoir candidates per class, then reload only picked inputs
            for c in needs:
                need = n_per_class - len(class_to_examples[c])
                if need <= 0:
                    continue
                if len(cand_idx[c]) == 0:
                    continue
 
                feats_c = torch.stack(cand_feat[c], dim=0)  # (M, D)
                center = feats_c.mean(dim=0, keepdim=True)
                dists = torch.norm(feats_c - center, dim=1)
                order = torch.argsort(dists)  # 가까운 순
                if need >= order.numel():
                    pick_pos = order
                else:
                    positions = torch.linspace(0, order.numel() - 1, steps=need).long()
                    pick_pos = order[positions]
 
                for pos in pick_pos.tolist():
                    ds_idx = cand_idx[c][pos]
                    x_sel, _y_sel = single_view_ds[ds_idx]
                    if isinstance(x_sel, (list, tuple)):
                        x_sel = x_sel[0]
                    class_to_examples[c].append(x_sel.detach().cpu())
 
        # rebuild buffer
        self.buffer.rebuild_from_class_to_examples(class_to_examples, dtype=probe_x.dtype)
        if self.args.verbose:
            print(f"Replay buffer bytes used: {self.buffer.bytes_used()} / {self.buffer.budget_bytes}")
            print(f"Replay buffer samples: {len(self.buffer)} (per-class target {n_per_class})")
 
    def train_classifier_from_buffer(self) -> None:
        if len(self.buffer) == 0:
            return
 
        x_cpu, y_cpu = self.buffer.get_all_cpu()
        ds = torch.utils.data.TensorDataset(x_cpu, y_cpu)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.args.classifier_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
 
        # freeze encoder, train classifier only
        self.model.eval()
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        for p in self.model.projector.parameters():
            p.requires_grad = False
        for p in self.model.classifier.parameters():
            p.requires_grad = True
 
        opt = torch.optim.AdamW(self.model.classifier.parameters(), lr=self.args.classifier_lr, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss().to(self.device)
 
        for ep in range(self.args.classifier_epochs):
            self.model.classifier.train()
            total_loss = 0.0
            total_acc = 0.0
            total_n = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                with torch.no_grad():
                    feats = self.model.forward_features(xb)
                logits = self.model.classifier(feats)
                loss = criterion(logits, yb)
 
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
 
                acc1 = accuracy(logits, yb, topk=(1,))[0]
                bs = int(yb.size(0))
                total_loss += float(loss.item()) * bs
                total_acc += float(acc1.item()) * bs
                total_n += bs
 
            if self.args.verbose:
                print(
                    f"[Classifier] epoch {ep+1}/{self.args.classifier_epochs} "
                    f"loss {total_loss/max(1,total_n):.4f} acc@1 {total_acc/max(1,total_n):.2f}"
                )
 
        # unfreeze encoder/projector back for next task training
        for p in self.model.encoder.parameters():
            p.requires_grad = True
        for p in self.model.projector.parameters():
            p.requires_grad = True
 
    # ---- OODVIL evaluation logic (same shape as snippet) ----
 
    def train_and_evaluate(self, data_loader, device, class_mask, args):
        """
        전체 incremental learning 과정을 수행합니다.
        각 task에 대해 지정된 epoch만큼 fine-tuning 후,
        지금까지의 task에 대해 평가합니다.
        """
        args.num_tasks = len(data_loader)
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks), dtype=np.float32)
 
        seen_classes: Set[int] = set()
        for task_id in range(args.num_tasks):
            task_classes = set(int(c) for c in (class_mask[task_id] if class_mask is not None else []))
            old_targets = sorted(list(seen_classes))
            seen_classes |= task_classes
 
            print(f"{f'Training on Task {task_id+1}/{args.num_tasks}':=^60}")
            train_start = time.time()
            epochs = args.epochs_first if (task_id == 0 and args.epochs_first > 0) else args.epochs
            for epoch in range(epochs):
                epoch_start = time.time()
                epoch_avg_loss, epoch_avg_acc = self.train_one_epoch(data_loader[task_id]["train"], epoch, old_targets)
                epoch_duration = time.time() - epoch_start
                print(
                    f"Epoch [{epoch+1}/{epochs}] Completed in {str(datetime.timedelta(seconds=int(epoch_duration)))}: "
                    f"Avg Loss = {epoch_avg_loss:.4f}, Avg Acc@1 = {epoch_avg_acc:.2f}"
                )
            train_duration = time.time() - train_start
            print(f"Task {task_id+1} training completed in {str(datetime.timedelta(seconds=int(train_duration)))}")
 
            # update replay buffer (isometric) + train inlier classifier on exemplars
            self.update_replay_buffer_isometric(data_loader[task_id]["train"].dataset, sorted(list(seen_classes)))
            self.train_classifier_from_buffer()
            self.refresh_exemplar_bank(num_classes=args.num_classes)
 
            # update teacher for next task
            self.teacher = self._make_teacher()
 
            print(f'{f"Testing on Task {task_id+1}/{args.num_tasks}":=^60}')
            eval_start = time.time()
            self.evaluate_till_now(self.model, data_loader, device, task_id, class_mask, acc_matrix, args)
            eval_duration = time.time() - eval_start
            print(f"Task {task_id+1} evaluation completed in {str(datetime.timedelta(seconds=int(eval_duration)))}")
 
            if args.ood_dataset:
                print(f"{f'OOD Evaluation':=^60}")
                ood_start = time.time()
                all_id_datasets = torch.utils.data.ConcatDataset([data_loader[t]["val"].dataset for t in range(task_id + 1)])
                ood_dataset = data_loader[-1]["ood"]
                self.evaluate_ood(self.model, all_id_datasets, ood_dataset, device, args, task_id)
                ood_duration = time.time() - ood_start
                print(f"OOD evaluation after Task {task_id+1} completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")
 
    def evaluate_till_now(self, model, data_loader, device, task_id, class_mask, acc_matrix, args):
        """
        현재까지의 모든 task에 대해 평가하고,
        A_last, A_avg, Forgetting 지표를 계산하여 출력합니다.
        """
        for t in range(task_id + 1):
            acc_matrix[t, task_id] = self.evaluate_task(model, data_loader[t]["val"], device, t, class_mask, args)
 
        A_i = [np.mean(acc_matrix[: i + 1, i]) for i in range(task_id + 1)]
        A_last = A_i[-1]
        A_avg = np.mean(A_i)
 
        result_str = "[Average accuracy till task{}] A_last: {:.2f} A_avg: {:.2f}".format(task_id + 1, A_last, A_avg)
 
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            result_str += " Forgetting: {:.4f}".format(forgetting)
        else:
            forgetting = 0
 
        if args.wandb:
            import wandb
 
            wandb.log({"A_last (↑)": A_last, "A_avg (↑)": A_avg, "Forgetting (↓)": forgetting, "TASK": task_id})
 
        print(result_str)
        if args.verbose or args.wandb:
            sub_matrix = acc_matrix[: task_id + 1, : task_id + 1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            heatmap_path = save_accuracy_heatmap(result, task_id, args)
            if args.wandb:
                import wandb
 
                wandb.log({"Accuracy Heatmap": wandb.Image(heatmap_path)})
 
        return {"Acc@1": A_last}
 
    def evaluate_task(self, model, data_loader, device, task_id, class_mask, args):
        """
        한 task에 대해 evaluation을 수행하며, 매 print_freq 배치마다 중간 결과를 출력합니다.
        """
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.eval()
        total_acc = 0.0
        total_loss = 0.0
        total_samples = 0
 
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                # develop: 1 batch만 평가
                if args.develop and batch_idx > 0:
                    break
 
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
 
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                batch_size = inputs.size(0)
 
                total_acc += acc1.item() * batch_size
                total_loss += loss.item() * batch_size
                total_samples += batch_size
 
                if batch_idx % args.print_freq == 0:
                    running_avg_loss = total_loss / total_samples
                    running_avg_acc = total_acc / total_samples
                    print(
                        f"Task {task_id+1}, Batch [{batch_idx}/{len(data_loader)}]: "
                        f"Running Avg Loss = {running_avg_loss:.2f}, Running Avg Acc@1 = {running_avg_acc:.2f}"
                    )
 
        avg_acc = total_acc / total_samples
        avg_loss = total_loss / total_samples
        print(f"Task {task_id+1}: Final Avg Loss = {avg_loss:.2f} | Final Avg Acc@1 = {avg_acc:.2f}")
        return avg_acc
 
    def evaluate_ood(self, model, id_datasets, ood_dataset, device, args, task_id=None):
        model.eval()
 
        # === New unified OOD evaluation (adapter 기반) ===
        ood_method = args.ood_method.upper()
 
        id_size, ood_size = len(id_datasets), len(ood_dataset)
        min_size = min(id_size, ood_size)
        if args.develop:
            # develop: OOD eval도 매우 작게
            min_size = min(min_size, getattr(args, "develop_samples", 32))
        if args.ood_develop:
            min_size = args.ood_develop
        if args.verbose:
            print(f"ID dataset size: {id_size}, OOD dataset size: {ood_size}. Using {min_size} samples each for evaluation.")
 
        id_dataset_aligned = RandomSampleWrapper(id_datasets, min_size, args.seed) if id_size > min_size else id_datasets
        ood_dataset_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset
 
        id_loader = torch.utils.data.DataLoader(id_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ood_loader = torch.utils.data.DataLoader(ood_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
 
        if ood_method == "ALL":
            methods = SUPPORTED_METHODS
        else:
            methods = [method.strip().upper() for method in ood_method.split(",")]
            unsupported = [m for m in methods if m not in SUPPORTED_METHODS]
            if unsupported:
                raise ValueError(f"지원되지 않는 OOD 메소드: {unsupported}. 지원되는 메소드: {SUPPORTED_METHODS}")
 
        results = {}
 
        for method in methods:
            id_scores, ood_scores = compute_ood_scores(method, model, id_loader, ood_loader, device)
 
            if args.verbose or args.wandb:
                hist_path = save_anomaly_histogram(id_scores.numpy(), ood_scores.numpy(), args, suffix=method.lower(), task_id=task_id)
                if args.wandb:
                    import wandb
 
                    wandb.log({f"Anomaly Histogram TASK {task_id}": wandb.Image(hist_path)})
 
            binary_labels = np.concatenate([np.ones(id_scores.shape[0]), np.zeros(ood_scores.shape[0])])
            all_scores = np.concatenate([id_scores.numpy(), ood_scores.numpy()])
 
            fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
            auroc = metrics.auc(fpr, tpr)
            idx_tpr95 = np.abs(tpr - 0.95).argmin()
            fpr_at_tpr95 = fpr[idx_tpr95]
 
            print(f"[{method}]: AUROC {auroc * 100:.2f}% | FPR@TPR95 {fpr_at_tpr95 * 100:.2f}%")
            if args.wandb:
                import wandb
 
                wandb.log({f"{method}_AUROC (↑)": auroc * 100, f"{method}_FPR@TPR95 (↓)": fpr_at_tpr95 * 100, "TASK": task_id})
 
            results[method] = {"auroc": auroc, "fpr_at_tpr95": fpr_at_tpr95, "scores": all_scores}
 
        return results
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
 
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("OpenIncrement on OOD-VIL scenario (VIL/CIL/DIL)")
 
    # scenario / data
    p.add_argument("--dataset", type=str, default="CLEAR", choices=["CLEAR", "DomainNet", "CORe50", "iDigits"])
    p.add_argument("--data_path", type=str, default="./data")
    p.add_argument("--IL_mode", type=str, default="vil", choices=["vil", "cil", "dil", "joint"])
    p.add_argument("--num_tasks", type=int, default=5)
    p.add_argument("--shuffle", action="store_true", help="shuffle class order (CIL)")
 
    # runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--print_freq", type=int, default=50)
    p.add_argument("--develop", action="store_true", help="quick dev run (limits batches)")
    p.add_argument("--develop_samples", type=int, default=32, help="develop 모드에서 사용 샘플 수(ood/exemplar 등)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--output_dir", type=str, default="./outputs/oodvil_openincrement")
 
    # backbone / openincrement
    p.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--lambda_dis", type=float, default=0.5)
    p.add_argument("--distill_batch_size", type=int, default=16, help="subset size for RKD (controls cubic RKAngle)")
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
 
    # optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--epochs_first", type=int, default=100)
 
    # replay buffer (BYTE budget)
    p.add_argument("--replay_buffer_bytes", type=int, default=0, help="Total replay buffer budget in bytes")
    p.add_argument("--replay_batch_size", type=int, default=0, help="How many replay samples to mix per batch")
    p.add_argument("--exemplar_batch_size", type=int, default=256, help="Batch size for exemplar feature extraction")
    p.add_argument("--exemplar_candidates_per_class", type=int, default=1024, help="Max reservoir candidates per class (memory-safe)")
 
    # inlier classifier training (on exemplars)
    p.add_argument("--classifier_epochs", type=int, default=10)
    p.add_argument("--classifier_lr", type=float, default=1e-3)
    p.add_argument("--classifier_batch_size", type=int, default=256)
 
    # OSR / OOD
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--ood_dataset", type=str, default="", help="Optional OOD dataset name (e.g., CIFAR100, SVHN, TinyImagenet...)")
    p.add_argument("--ood_method", type=str, default="OPENINCREMENT_KNN", help="OpenIncrement only: OPENINCREMENT_KNN (or ALL)")
    p.add_argument("--ood_develop", type=int, default=0, help="limit samples used for OOD eval")
 
    # wandb
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--wandb_run", type=str, default="")
 
    return p
 
 
def main(args) -> None:
    # Keep flags that build_continual_dataloader forcibly overwrites.
    verbose_flag = bool(args.verbose)
 
    args = set_data_config(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # develop: 전체 프로세스를 매우 빠르게 (태스크별 1 epoch 1 batch)
    if args.develop:
        args.epochs_first = 1
        args.epochs = 1
        args.classifier_epochs = 1
        args.print_freq = 1
        # 데이터 로딩 오버헤드 최소화
        args.num_workers = 0
        # exemplar/ood 단계도 최소화
        args.exemplar_candidates_per_class = min(int(args.exemplar_candidates_per_class), int(args.develop_samples))
        args.exemplar_batch_size = min(int(args.exemplar_batch_size), int(args.develop_samples))
        args.classifier_batch_size = min(int(args.classifier_batch_size), int(args.develop_samples))
        if not args.ood_develop:
            args.ood_develop = int(args.develop_samples)
 
    # build scenario loaders
    data_loader, class_mask, domain_list = build_continual_dataloader(args)
    args.verbose = verbose_flag
 
    if args.ood_dataset:
        data_loader[-1]["ood"] = get_ood_dataset(args.ood_dataset, args)
 
    # model + timm transforms
    model = OpenIncrementNet(
        backbone=args.backbone,
        num_classes=args.num_classes,
        proj_dim=args.proj_dim,
        pretrained=bool(args.pretrained),
    )
 
    data_config = resolve_data_config({}, model=model.encoder)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)
    train_transform_2c = TwoCropTransform(train_transform)
 
    # attach transforms to existing datasets
    for t in range(len(data_loader)):
        _set_transform_inplace(data_loader[t]["train"].dataset, train_transform_2c)
        _set_transform_inplace(data_loader[t]["val"].dataset, eval_transform)
    if args.ood_dataset:
        _set_transform_inplace(data_loader[-1]["ood"], eval_transform)
 
    print(args)
    args.wandb = False
    if args.wandb_run and args.wandb_project:
        import getpass
        import wandb
 
        args.wandb = True
        wandb.init(entity="OODVIL", project=args.wandb_project, name=args.wandb_run, config=vars(args))
        wandb.config.update({"username": getpass.getuser()})
 
    runner = OpenIncrementRunner(model=model, device=device, args=args, eval_transform=eval_transform)
    runner.train_and_evaluate(data_loader, device, class_mask, args)
 
 
if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
