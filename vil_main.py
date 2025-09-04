from __future__ import annotations

import argparse
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler
from timm.models import create_model
import os
import matplotlib.pyplot as plt
import seaborn as sns

from continual_datasets.build_incremental_scenario import build_continual_dataloader
from continual_datasets.dataset_utils import set_data_config, get_ood_dataset
from resnet_big import SupCEResNet, SupConResNet, LinearClassifier
from util import set_optimizer, AverageMeter, TwoCropTransform
from losses import SupConLoss


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

    # SupCon params
    parser.add_argument('--temp', type=float, default=0.05)

    parser.add_argument('--ood_dataset', type=str, default=None, help='예: EMNIST, NotMNIST, etc.')
    parser.add_argument('--K', type=int, default=10, help='KNN/OSNN에서 사용할 K')
    parser.add_argument('--Ts', type=float, default=0.85, help='OSNN threshold Ts')
    parser.add_argument('--Tr', type=float, default=1.9, help='OSNN threshold Tr')

    parser.add_argument('--fixed_memory', type=int, default=2000, help='전체 고정 메모리 사이즈(피처 개수 기준)')
    parser.add_argument('--memory_size', type=int, default=50, help='태스크당 최대 피처 개수(고정 메모리 미사용 시)')
    parser.add_argument('--mem_per_task', type=int, default=None, help='태스크당 메모리 개수 고정 (우선 적용)')

    parser.add_argument('--linear_epochs', type=int, default=10)
    parser.add_argument('--linear_lr', type=float, default=0.1)
    parser.add_argument('--develop', action='store_true')

    # Logging / W&B
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='OpenIncrement', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team/user)')
    parser.add_argument('--save', type=str, default='outputs', help='Directory to save figures and artifacts')

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


def _wrap_two_crop_transform(dataset) -> None:
    try:
        from torch.utils.data import Subset, ConcatDataset
    except Exception:
        Subset = object  # type: ignore
        ConcatDataset = object  # type: ignore

    # Recursively wrap transform with TwoCropTransform for train datasets
    if hasattr(dataset, 'datasets') and isinstance(dataset.datasets, list):  # ConcatDataset
        for ds in dataset.datasets:
            _wrap_two_crop_transform(ds)
        return
    if hasattr(dataset, 'dataset'):  # Subset
        _wrap_two_crop_transform(dataset.dataset)
        return
    if hasattr(dataset, 'transform') and dataset.transform is not None:
        if not isinstance(dataset.transform, TwoCropTransform):
            dataset.transform = TwoCropTransform(dataset.transform)


def train_one_task_supcon(model: torch.nn.Module,
                          optimizer: torch.optim.Optimizer,
                          train_loader: DataLoader,
                          device: torch.device,
                          epochs: int,
                          print_freq: int,
                          temperature: float,
                          develop: bool = False) -> None:
    criterion = SupConLoss(temperature=temperature).to(device)
    model.train()

    # ensure two-crop transform is applied to underlying dataset
    try:
        _wrap_two_crop_transform(train_loader.dataset)
    except Exception:
        pass

    for epoch in range(1, epochs + 1):
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()

        for idx, (images, targets) in enumerate(train_loader):
            # images: [view1, view2], each shape [B,C,H,W]
            if isinstance(images, (list, tuple)) and len(images) == 2:
                images = torch.cat([images[0], images[1]], dim=0)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bsz = targets.size(0)

            optimizer.zero_grad()
            feats = model(images)
            f1, f2 = torch.split(feats, [bsz, bsz], dim=0)
            feats_pair = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(feats_pair, targets)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % max(1, print_freq) == 0:
                print(f"[SupCon] Epoch[{epoch}] Step[{idx+1}/{len(train_loader)}]\tTime {batch_time.val:.3f}({batch_time.avg:.3f})\tLoss {losses.val:.4f}({losses.avg:.4f})")
            if develop and (idx + 1) >= 2:
                break


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> float:
    model.eval()
    correct = 0
    total = 0
    for b_idx, (images, targets) in enumerate(val_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        if max_batches is not None and (b_idx + 1) >= max_batches:
            break
    acc = (correct / max(1, total)) * 100.0
    return acc


@torch.no_grad()
def extract_encoder_features(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, labels = [], []
    for b_idx, (images, targets) in enumerate(loader):
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device, non_blocking=True)
        features = model.encoder(images)
        feats.append(features.detach().cpu().numpy())
        labels.append(targets.numpy())
        if max_batches is not None and (b_idx + 1) >= max_batches:
            break
    if len(feats) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


@torch.no_grad()
def extract_embedding_features(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return model's embedding (post-projection, normalized for SupConResNet)."""
    model.eval()
    feats, labels = [], []
    for b_idx, (images, targets) in enumerate(loader):
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device, non_blocking=True)
        features = model(images)
        feats.append(features.detach().cpu().numpy())
        labels.append(targets.numpy())
        if max_batches is not None and (b_idx + 1) >= max_batches:
            break
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


class ViTContrastiveWrapper(torch.nn.Module):
    def __init__(self, timm_model: torch.nn.Module, feat_dim: int = 128):
        super().__init__()
        self.backbone = timm_model  # num_classes=0 to use as encoder
        # infer embedding dim
        embed_dim = getattr(timm_model, 'num_features', None) or getattr(timm_model, 'embed_dim', 768)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(embed_dim, feat_dim),
        )

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        try:
            feats = self.backbone.forward_head(feats, pre_logits=True)
        except Exception:
            pass
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.head(z)
        return F.normalize(z, dim=1)


def build_model(args) -> torch.nn.Module:
    if args.model == 'vit_b16':
        timm_model = create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        model = ViTContrastiveWrapper(timm_model)
    else:
        model = SupConResNet(name=args.model)
    return model


def _build_linear_classifier(model: torch.nn.Module, num_classes: int, model_name: str) -> torch.nn.Module:
    # Try to use predefined LinearClassifier for ResNet backbones; otherwise infer input dim
    try:
        classifier = LinearClassifier(name=model_name, num_classes=num_classes)
        return classifier
    except Exception:
        pass
    # Fallback: infer in_features by a forward pass through encoder
    # Use a dummy input with 3x224x224
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        if next(model.parameters()).is_cuda:
            dummy = dummy.cuda()
        feat = model.encoder(dummy)
        in_dim = feat.shape[1]
    return torch.nn.Linear(in_dim, num_classes)


def _build_concat_loader(datasets: List[torch.utils.data.Dataset], batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    concat = ConcatDataset(datasets)
    sampler = torch.utils.data.RandomSampler(concat) if shuffle else SequentialSampler(concat)
    return DataLoader(concat, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


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


def save_accuracy_heatmap(acc_matrix: np.ndarray, task_id: int, args: argparse.Namespace) -> str:
    # 폴더 생성
    save_dir = os.path.join(args.save, 'heatmaps')
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(acc_matrix, annot=True, cmap='YlGnBu', ax=ax, vmin=0, vmax=100)

    ax.set_title(f'Accuracy Heatmap (Task {task_id+1})')
    ax.set_xlabel('After learning task')
    ax.set_ylabel('Tested on task')

    # 저장 경로 설정
    save_path = os.path.join(save_dir, f'heatmap_task{task_id+1}.png')
    plt.savefig(save_path)
    plt.close()

    return save_path


def save_anomaly_histogram(id_scores: np.ndarray,
                           ood_scores: np.ndarray,
                           args: argparse.Namespace,
                           suffix: str = '',
                           task_id: Optional[int] = None) -> str:
    plt.figure(figsize=(10, 6))

    # 폴더 생성
    save_dir = os.path.join(args.save, 'anomaly_histograms')
    os.makedirs(save_dir, exist_ok=True)

    # ID 및 OOD 점수 히스토그램 그리기
    bins = np.linspace(
        min(np.min(id_scores), np.min(ood_scores)),
        max(np.max(id_scores), np.max(ood_scores)),
        100
    )

    plt.hist(id_scores, bins=bins, alpha=0.5, label='ID', density=True)
    plt.hist(ood_scores, bins=bins, alpha=0.5, label='OOD', density=True)

    plt.title(f'Anomaly Score Distribution ({suffix.upper()})')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()

    task_str = f'_task{task_id+1}' if task_id is not None else ''
    save_path = os.path.join(save_dir, f'anomaly_hist_{suffix}{task_str}.png')
    plt.savefig(save_path)
    plt.close()

    return save_path


def _euclidean_exemplars_per_class(embeddings: np.ndarray, labels: np.ndarray, num_classes: int, memory_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    exemplar_features: List[np.ndarray] = []
    exemplar_labels: List[int] = []
    for c in range(num_classes):
        idxs = np.where(labels == c)[0]
        if idxs.size == 0:
            continue
        feats_c = embeddings[idxs]
        center = feats_c.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(feats_c - center, axis=1)
        k = min(memory_per_class, dists.shape[0])
        keep = np.argpartition(dists, kth=k-1)[:k]
        exemplar_features.append(feats_c[keep])
        exemplar_labels.extend([c] * k)
    if len(exemplar_features) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(exemplar_features, axis=0), np.array(exemplar_labels, dtype=np.int64)


def _osnn_predict(test_feature: np.ndarray,
                  exemplar_features: np.ndarray,
                  exemplar_labels: np.ndarray,
                  K: int,
                  Ts: float,
                  Tr: float) -> Tuple[float, int, float]:
    # cosine similarities (features assumed normalized)
    sims = exemplar_features @ test_feature.reshape(-1, 1)
    sims = sims.squeeze(1)
    # top-K indices
    K_eff = min(K, sims.shape[0])
    topk_idx = np.argpartition(sims, -K_eff)[-K_eff:]
    topk_labels = exemplar_labels[topk_idx]
    # average similarity of top-K
    avg_sim = float(sims[topk_idx].mean())

    # if all K belong to single class
    from collections import Counter
    cnt = Counter(topk_labels.tolist())
    if len(cnt) == 1:
        only_class = topk_labels[0]
        if avg_sim > Ts:
            return 10.0, int(only_class), avg_sim
        else:
            return 10.0, 1000, avg_sim  # OOD

    # compute ratio R between best and second-best class sums
    best_two = cnt.most_common(2)
    best_c, second_c = best_two[0][0], best_two[1][0]
    mask_best = (topk_labels == best_c)
    mask_second = (topk_labels == second_c)
    sum_best = float(sims[topk_idx][mask_best].sum())
    sum_second = float(sims[topk_idx][mask_second].sum())
    R = sum_best / max(1e-12, sum_second)
    if R < Tr:
        return R, 1000, avg_sim
    else:
        return R, int(best_c), avg_sim


def train_linear_classifier(encoder_model: torch.nn.Module,
                            train_loader: DataLoader,
                            num_classes: int,
                            device: torch.device,
                            model_name: str,
                            epochs: int,
                            lr: float,
                            print_freq: int,
                            develop: bool = False) -> torch.nn.Module:
    classifier = _build_linear_classifier(encoder_model, num_classes, model_name)
    classifier = classifier.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)

    encoder_model.eval()
    for epoch in range(1, epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        for idx, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                feats = encoder_model.encoder(images)
            logits = classifier(feats.detach())
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            pred = torch.argmax(logits, dim=1)
            acc = (pred == labels).float().mean().item() * 100.0
            losses.update(loss.item(), images.size(0))
            top1.update(acc, images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % max(1, print_freq) == 0:
                print(f"[Linear] Epoch[{epoch}] Step[{idx+1}/{len(train_loader)}]\tTime {batch_time.val:.3f}({batch_time.avg:.3f})\tLoss {losses.val:.4f}({losses.avg:.4f})\tAcc@1 {top1.val:.2f}({top1.avg:.2f})")
            if develop and (idx + 1) >= 2:
                break

    return classifier


@torch.no_grad()
def evaluate_linear_classifier(encoder_model: torch.nn.Module,
                               classifier: torch.nn.Module,
                               val_loader: DataLoader,
                               device: torch.device,
                               max_batches: Optional[int] = None) -> float:
    encoder_model.eval()
    classifier.eval()
    correct, total = 0, 0
    for b_idx, (images, targets) in enumerate(val_loader):
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        feats = encoder_model.encoder(images)
        logits = classifier(feats)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        if max_batches is not None and (b_idx + 1) >= max_batches:
            break
    return (correct / max(1, total)) * 100.0


def evaluate_till_now_linear(encoder_model: torch.nn.Module,
                             classifier: torch.nn.Module,
                             data_loader: List[Dict[str, DataLoader]],
                             device: torch.device,
                             task_id: int,
                             acc_matrix: np.ndarray,
                             args: argparse.Namespace) -> None:

    for t in range(task_id + 1):
        val_loader_t = data_loader[t]['val']
        acc_t = evaluate_linear_classifier(
            encoder_model, classifier, val_loader_t, device,
            max_batches=(2 if getattr(args, 'develop', False) else None)
        )
        acc_matrix[t, task_id] = acc_t

    A_i = [np.mean(acc_matrix[:i+1, i]) for i in range(task_id+1)]
    A_last = A_i[-1]
    A_avg = np.mean(A_i)

    result_str = "[Average accuracy till task{}] A_last: {:.2f} A_avg: {:.2f}".format(task_id+1, A_last, A_avg)
    
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
        result_str += " Forgetting: {:.4f}".format(forgetting)
    else:
        forgetting = 0

    print(result_str)

    # Accuracy heatmap (upper-triangular of seen tasks)
    sub_matrix = acc_matrix[:task_id+1, :task_id+1]
    mask_upper = np.triu(np.ones_like(sub_matrix, dtype=bool))
    vis_matrix = np.where(mask_upper, sub_matrix, np.nan)
    heatmap_path = save_accuracy_heatmap(vis_matrix, task_id, args)

    # W&B logging
    if getattr(args, 'wandb', False):
        try:
            import wandb
            wandb.log({
                "A_last (↑)": A_last,
                "A_avg (↑)": A_avg,
                "A_last": A_last,
                "A_avg": A_avg,
                "Forgetting (↓)": forgetting,
                "Forgetting": forgetting,
                "TASK": task_id + 1,
                "Accuracy Heatmap": wandb.Image(heatmap_path),
            })
        except Exception as e:
            print(f"[W&B] 로그 실패: {e}")

def main():
    args = parse_args()
    set_seed(args.seed)
    args = set_data_config(args)
    args.verbose = True

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Prepare output directory
    os.makedirs(args.save, exist_ok=True)

    # Initialize Weights & Biases
    if getattr(args, 'wandb', False):
        try:
            import wandb
            run_name = f"{args.dataset}-{args.IL_mode}-{args.model}-seed{args.seed}"
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=run_name)
            # apply wandb.config overrides back to argparse args (for sweeps)
            try:
                cfg_dict = wandb.config.as_dict()
                for k, v in cfg_dict.items():
                    if hasattr(args, k):
                        setattr(args, k, v)
            except Exception as e:
                print(f"[W&B] config 동기화 실패: {e}")
            # per-run save directory
            try:
                run_id = getattr(wandb.run, 'id', None)
                if run_id is not None:
                    args.save = os.path.join(args.save, str(run_id))
                    os.makedirs(args.save, exist_ok=True)
            except Exception as e:
                print(f"[W&B] 저장 경로 설정 실패: {e}")
        except Exception as e:
            print(f"[W&B] 초기화 실패: {e}")

    if args.develop:
        print("[DEVELOP] 개발 모드 활성화: 각 루프 2 iteration, epochs=1, linear_epochs=1")
        args.epochs = min(args.epochs, 1)
        args.linear_epochs = min(args.linear_epochs, 1)
        args.print_freq = 1

    data_loader, class_mask, domain_list = build_continual_dataloader(args)

    model = build_model(args).to(device)
    optimizer = set_optimizer(args, model)

    ood_loader = None
    if args.ood_dataset is not None:
        ood_ds = get_ood_dataset(args.ood_dataset, args)
        ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # for exemplar-based OOD
    exemplar_features: Optional[np.ndarray] = None
    exemplar_labels: Optional[np.ndarray] = None
    seen_tasks: int = 0

    

    print(f"{'VIL RUN':=^60}")
    print(f"dataset={args.dataset} | IL_mode={args.IL_mode} | tasks={args.num_tasks}")
    print(f"model={args.model} | epochs={args.epochs} | batch={args.batch_size} | lr={args.learning_rate}")
    print(f"ood_method=OSNN | ood_dataset={args.ood_dataset}")

    all_val_datasets: List[torch.utils.data.Dataset] = []
    all_train_datasets: List[torch.utils.data.Dataset] = []

    acc_matrix = np.zeros((args.num_tasks, args.num_tasks), dtype=np.float32)

    for task_id in range(args.num_tasks):
        print(f"\n{' Task %d ' % (task_id + 1):=^60}")
        current_train_loader = data_loader[task_id]['train']
        current_val_loader = data_loader[task_id]['val']
        all_val_datasets.append(current_val_loader.dataset)
        all_train_datasets.append(current_train_loader.dataset)

        current_classes: List[int] = class_mask[task_id] if class_mask is not None else list(range(args.num_classes))
        seen_tasks += 1

        if args.mem_per_task is not None:
            memory_per_task = max(1, int(args.mem_per_task))
        elif args.fixed_memory == 0:
            memory_per_task = args.memory_size
        else:
            memory_per_task = max(1, args.fixed_memory // max(1, seen_tasks))

        print(f"domain(s)={domain_list[task_id] if domain_list is not None else 'N/A'} | classes={current_classes} | seen_tasks={seen_tasks} | mem_per_task={memory_per_task}")

        # SupCon training for current task
        train_one_task_supcon(model, optimizer, current_train_loader, device, epochs=args.epochs, print_freq=args.print_freq, temperature=args.temp, develop=args.develop)

        id_eval_loader = build_eval_loader_from_datasets(all_val_datasets, args.batch_size, args.num_workers)
        # Train linear classifier on accumulated train data (freeze encoder)
        lin_train_loader = _build_concat_loader(all_train_datasets, args.batch_size, args.num_workers, shuffle=True)
        classifier_head = train_linear_classifier(model, lin_train_loader, args.num_classes, device, args.model, epochs=args.linear_epochs, lr=args.linear_lr, print_freq=args.print_freq, develop=args.develop)
        id_acc = evaluate_linear_classifier(model, classifier_head, id_eval_loader, device, max_batches=(2 if args.develop else None))
        print(f"[Linear-Classifier] Acc@ID (tasks 1..{task_id+1}) = {id_acc:.2f}%")

        evaluate_till_now_linear(model, classifier_head, data_loader, device, task_id, acc_matrix, args)

        # Build/update exemplar features per class using current encoder and accumulated data
        # Determine memory_per_class
        if args.fixed_memory == 0:
            memory_per_class = args.memory_size
        else:
            memory_per_class = max(1, args.fixed_memory // max(1, args.num_classes))

        ex_build_loader = _build_concat_loader(all_train_datasets, args.batch_size, args.num_workers, shuffle=False)
        emb_all, lab_all = extract_embedding_features(model, ex_build_loader, device, max_batches=(2 if args.develop else None))
        # ensure normalized for cosine in OSNN
        if emb_all.size > 0:
            # already normalized from SupConResNet forward; ensure anyway for ViT wrapper
            norm = np.linalg.norm(emb_all, axis=1, keepdims=True) + 1e-12
            emb_all = emb_all / norm
        exemplar_features, exemplar_labels = _euclidean_exemplars_per_class(emb_all, lab_all, args.num_classes, memory_per_class)
        print(f"[Exemplar] per_class={memory_per_class} | total={exemplar_features.shape[0] if exemplar_features is not None else 0}")

        if ood_loader is not None and exemplar_features is not None and exemplar_features.size > 0:
            # ID features
            id_emb, id_lab = extract_embedding_features(model, id_eval_loader, device, max_batches=(2 if args.develop else None))
            if id_emb.size > 0:
                id_emb = id_emb / (np.linalg.norm(id_emb, axis=1, keepdims=True) + 1e-12)
            # OOD features
            ood_emb, ood_lab = extract_embedding_features(model, ood_loader, device, max_batches=(2 if args.develop else None))
            if ood_emb.size > 0:
                ood_emb = ood_emb / (np.linalg.norm(ood_emb, axis=1, keepdims=True) + 1e-12)

            # OSNN scoring (use average similarity as score for AUROC like knn.py)
            from collections import deque
            SIn, SOut = deque(), deque()
            for f in id_emb:
                _, _, sim = _osnn_predict(f, exemplar_features, exemplar_labels, K=args.K, Ts=args.Ts, Tr=args.Tr)
                SIn.append(sim)
            for f in ood_emb:
                _, _, sim = _osnn_predict(f, exemplar_features, exemplar_labels, K=args.K, Ts=args.Ts, Tr=args.Tr)
                SOut.append(sim)
            id_scores = np.array(SIn, dtype=np.float32)
            ood_scores = np.array(SOut, dtype=np.float32)
            auroc, fpr95 = compute_ood_metrics(id_scores, ood_scores)
            print(f"[OOD-OSNN] AUROC {auroc:.2f}% | FPR@TPR95 {fpr95:.2f}%")

            # W&B: OOD metrics + histogram
            if getattr(args, 'wandb', False):
                try:
                    import wandb
                    wandb.log({
                        "OSNN_AUROC (↑)": auroc,
                        "OSNN_FPR@TPR95 (↓)": fpr95,
                        "TASK": task_id + 1,
                    })
                    hist_path = save_anomaly_histogram(id_scores, ood_scores, args, suffix='osnn', task_id=task_id)
                    wandb.log({f"Anomaly Histogram TASK {task_id+1} (OSNN)": wandb.Image(hist_path)})
                except Exception as e:
                    print(f"[W&B] OOD 로그 실패: {e}")

        

    print("\nAll tasks completed.")

    # W&B finalize
    if getattr(args, 'wandb', False):
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()


