from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler
from timm.models import create_model

from continual_datasets.build_incremental_scenario import build_continual_dataloader
from continual_datasets.dataset_utils import set_data_config, get_ood_dataset
from resnet_big import SupCEResNet
from util import set_optimizer, AverageMeter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('VIL Main (IL + OOD per task)')

    parser.add_argument('--dataset', type=str, default='iDigits')
    parser.add_argument('--data_path', type=str, default='/local_datasets')
    parser.add_argument('--IL_mode', type=str, default='vil', choices=['vil'])
    parser.add_argument('--num_tasks', type=int, default=20)
    parser.add_argument('--shuffle', action='store_true')

    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'vit_b16'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=50)

    parser.add_argument('--ood_dataset', type=str, default=None, help='예: EMNIST, NotMNIST, etc.')
    parser.add_argument('--K', type=int, default=10, help='KNN에서 사용할 K')

    parser.add_argument('--fixed_memory', type=int, default=2000, help='전체 고정 메모리 사이즈(피처 개수 기준)')
    parser.add_argument('--memory_size', type=int, default=50, help='태스크당 최대 피처 개수(고정 메모리 미사용 시)')
    parser.add_argument('--mem_per_task', type=int, default=None, help='태스크당 메모리 개수 고정 (우선 적용)')

    parser.add_argument('--save_dir', type=str, default='./save/VIL')

    args = parser.parse_args()
    return args


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eval_loader_from_datasets(datasets: List[torch.utils.data.Dataset], batch_size: int, num_workers: int) -> DataLoader:
    concat = ConcatDataset(datasets)
    sampler = SequentialSampler(concat)
    return DataLoader(concat, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def train_one_task(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, device: torch.device, epochs: int, print_freq: int) -> None:
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % max(1, print_freq) == 0:
                print(f"Train Epoch[{epoch}] Step[{idx+1}/{len(train_loader)}]\tTime {batch_time.val:.3f}({batch_time.avg:.3f})\tLoss {losses.val:.4f}({losses.avg:.4f})")


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    acc = (correct / max(1, total)) * 100.0
    return acc


@torch.no_grad()
def extract_features(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        # encoder -> features (ResNet: model.encoder; ViT: wrapper.encoder)
        features = model.encoder(images)
        features = F.normalize(features, dim=1)
        feats.append(features.detach().cpu().numpy())
        labels.append(targets.numpy())
    if len(feats) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def update_task_memory(new_features: np.ndarray,
                       memory_by_task: List[np.ndarray],
                       memory_limit_per_task: int) -> None:
    if new_features.shape[0] == 0:
        memory_by_task.append(np.empty((0, 0), dtype=np.float32))
        return
    # 샘플링하여 태스크 메모리로 저장
    num_keep = min(memory_limit_per_task, new_features.shape[0])
    keep_idxs = np.random.choice(new_features.shape[0], size=num_keep, replace=False)
    memory_by_task.append(new_features[keep_idxs])


def pack_memory(memory_by_task: List[np.ndarray]) -> np.ndarray:
    xs = [m for m in memory_by_task if m is not None and m.size > 0]
    if not xs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(xs, axis=0)


def knn_avg_similarity_scores(query_feats: np.ndarray, memory_feats: np.ndarray, K: int) -> np.ndarray:
    if memory_feats.shape[0] == 0 or query_feats.shape[0] == 0:
        return np.zeros((query_feats.shape[0],), dtype=np.float32)
    # cosine similarity (features are L2-normalized already)
    sims = np.matmul(query_feats, memory_feats.T)  # [Nq, Nm]
    # top-K average similarity
    K_eff = min(K, sims.shape[1])
    # partition for top-k indices, then take mean
    topk = np.partition(sims, -K_eff, axis=1)[:, -K_eff:]
    return topk.mean(axis=1)


class ViTWrapper(torch.nn.Module):
    def __init__(self, timm_model: torch.nn.Module):
        super().__init__()
        self.backbone = timm_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        # features before classifier head
        feats = self.backbone.forward_features(x)
        try:
            feats = self.backbone.forward_head(feats, pre_logits=True)
        except Exception:
            pass
        return feats


def build_model(args) -> torch.nn.Module:
    if args.model == 'vit_b16':
        timm_model = create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
        model = ViTWrapper(timm_model)
    else:
        model = SupCEResNet(name=args.model, num_classes=args.num_classes)
    return model


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> Tuple[float, float]:
    from sklearn import metrics
    if id_scores.size == 0 or ood_scores.size == 0:
        return float('nan'), float('nan')
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, drop_intermediate=False)
    auroc = metrics.auc(fpr, tpr)
    idx = np.abs(tpr - 0.95).argmin()
    fpr95 = fpr[idx]
    return auroc * 100.0, fpr95 * 100.0


def main():
    args = parse_args()
    set_seed(args.seed)
    args = set_data_config(args)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    data_loader, class_mask, domain_list = build_continual_dataloader(args)

    model = build_model(args).to(device)
    optimizer = set_optimizer(args, model)

    ood_loader = None
    if args.ood_dataset is not None:
        ood_ds = get_ood_dataset(args.ood_dataset, args)
        ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    memory_by_task: List[np.ndarray] = []
    seen_tasks: int = 0

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"{'VIL RUN':=^60}")
    print(f"dataset={args.dataset} | IL_mode={args.IL_mode} | tasks={args.num_tasks}")
    print(f"model={args.model} | epochs={args.epochs} | batch={args.batch_size} | lr={args.learning_rate}")
    print(f"ood_method=KNN | ood_dataset={args.ood_dataset}")
    print(f"save_dir={args.save_dir}")

    all_val_datasets: List[torch.utils.data.Dataset] = []

    for task_id in range(args.num_tasks):
        print(f"\n{' Task %d ' % (task_id + 1):=^60}")
        current_train_loader = data_loader[task_id]['train']
        current_val_loader = data_loader[task_id]['val']
        all_val_datasets.append(current_val_loader.dataset)

        current_classes: List[int] = class_mask[task_id] if class_mask is not None else list(range(args.num_classes))
        seen_tasks += 1

        if args.mem_per_task is not None:
            memory_per_task = max(1, int(args.mem_per_task))
        elif args.fixed_memory == 0:
            memory_per_task = args.memory_size
        else:
            memory_per_task = max(1, args.fixed_memory // max(1, seen_tasks))

        print(f"domain(s)={domain_list[task_id] if domain_list is not None else 'N/A'} | classes={current_classes} | seen_tasks={seen_tasks} | mem_per_task={memory_per_task}")

        train_one_task(model, optimizer, current_train_loader, device, epochs=args.epochs, print_freq=args.print_freq)

        id_eval_loader = build_eval_loader_from_datasets(all_val_datasets, args.batch_size, args.num_workers)
        id_acc = evaluate_classifier(model, id_eval_loader, device)
        print(f"[Classifier] Acc@ID (tasks 1..{task_id+1}) = {id_acc:.2f}%")

        feats_new, _ = extract_features(model, current_val_loader, device)
        update_task_memory(feats_new, memory_by_task, memory_per_task)
        if args.mem_per_task is not None or args.fixed_memory > 0:
            for i in range(len(memory_by_task)):
                if memory_by_task[i].shape[0] > memory_per_task:
                    keep = np.random.choice(memory_by_task[i].shape[0], size=memory_per_task, replace=False)
                    memory_by_task[i] = memory_by_task[i][keep]
        mem_feats = pack_memory(memory_by_task)
        print(f"[Memory] total feature count = {mem_feats.shape[0]} (tasks={len(memory_by_task)})")

        if ood_loader is not None:
            id_feats, _ = extract_features(model, id_eval_loader, device)
            id_scores = knn_avg_similarity_scores(id_feats, mem_feats, K=args.K)
            ood_feats, _ = extract_features(model, ood_loader, device)
            ood_scores = knn_avg_similarity_scores(ood_feats, mem_feats, K=args.K)

            auroc, fpr95 = compute_ood_metrics(id_scores, ood_scores)
            print(f"[OOD-KNN] AUROC {auroc:.2f}% | FPR@TPR95 {fpr95:.2f}%")

        ckpt_dir = os.path.join(args.save_dir, f"{args.dataset}_{args.model}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"task_{task_id+1}_last.pth")
        torch.save({'model': model.state_dict(), 'seen_tasks': seen_tasks}, ckpt_path)
        print(f"[Save] checkpoint -> {ckpt_path}")

    print("\nAll tasks completed.")


if __name__ == '__main__':
    main()


