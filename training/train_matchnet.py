import os
import sysscipy
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Добавляем корень проекта в путь импорта, чтобы можно было подключить модель.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.matchnet import MatchNet


def encode_energy_pref(pref: str) -> float:
    pref = pref.lower()
    if pref == "high":
        return 1.0
    if pref == "low":
        return 0.0
    return 0.5


def build_features(  # создает признаки для модели
    user_bpm: float,
    track_bpm: float,
    energy: float,
    dance: float,
    pref: str,
) -> list[float]:
    pref_val = encode_energy_pref(pref)
    energy_alignment = 1.0 - abs(float(energy) - pref_val)

    return [
        float(user_bpm),
        float(track_bpm),
        abs(float(user_bpm) - float(track_bpm)),
        float(energy_alignment),
        float(dance),
    ]


def make_label(user_bpm: float, track_bpm: float, energy: float, pref: str) -> float:  # создает целевую метку 0/1
    pref_val = encode_energy_pref(pref)
    bpm_ok = abs(float(user_bpm) - float(track_bpm)) <= 12
    energy_ok = abs(float(energy) - pref_val) <= 0.45
    return 1.0 if (bpm_ok and energy_ok) else 0.0


def load_music(csv_path: str) -> pd.DataFrame:  # загружает и очищает CSV с музыкой
    df = pd.read_csv(csv_path)

    required = ["title", "artist", "genre", "bpm", "energy", "dance"]  # обязательные колонки
    missing = [c for c in required if c not in df.columns]  # ищет, каких колонок не хватает
    if missing:
        raise ValueError(f"Нет обязательных колонок: {missing}")

    df = df.dropna(subset=required).copy()  # удаляет строки, где есть пропуски в обязательных колонках
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df["dance"] = pd.to_numeric(df["dance"], errors="coerce")
    df = df.dropna(subset=["bpm", "energy", "dance"]).reset_index(drop=True)
    return df


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if high >= 1:
        high = 0.99

    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float,
    high: float,
) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs)
    return filtfilt(b, a, signal)


def extract_acc_magnitude(df: pd.DataFrame) -> np.ndarray:
    candidates = [
        ("userAcceleration.x", "userAcceleration.y", "userAcceleration.z"),
        ("Acc_x", "Acc_y", "Acc_z"),
        ("acc_x", "acc_y", "acc_z"),
    ]

    for cols in candidates:
        if all(c in df.columns for c in cols):
            ax = df[cols[0]].to_numpy(dtype=float)
            ay = df[cols[1]].to_numpy(dtype=float)
            az = df[cols[2]].to_numpy(dtype=float)
            return np.sqrt(ax**2 + ay**2 + az**2)

    raise ValueError("Не удалось найти колонки акселерометра (x, y, z) в MotionSense CSV.")


def compute_bpm_from_motion_file(
    path: str,
    sample_rate_hz: float = 50.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 3.5,
) -> float:
    df = pd.read_csv(path)
    mag = extract_acc_magnitude(df)
    mag = mag - np.mean(mag)

    filtered = apply_bandpass_filter(
        mag,
        fs=sample_rate_hz,
        low=bandpass_low,
        high=bandpass_high,
    )

    min_distance = int(sample_rate_hz * 0.3)
    peaks, _ = find_peaks(filtered, distance=min_distance)

    if len(peaks) < 2:
        raise RuntimeError(f"Недостаточно шагов в сигнале для оценки BPM: {path}")

    intervals = np.diff(peaks)
    median_interval = np.median(intervals)

    if median_interval <= 0:
        raise RuntimeError(f"Некорректный интервал между шагами: {path}")

    step_period_sec = median_interval / sample_rate_hz
    return float(60.0 / step_period_sec)


def load_user_bpms(motionsense_root: str) -> list[float]:  # вычисляет BPM по MotionSense CSV
    if not os.path.isdir(motionsense_root):
        raise FileNotFoundError(f"Папка MotionSense не найдена: {motionsense_root}")

    user_bpms: list[float] = []
    skipped_files = 0

    for dirpath, _, filenames in os.walk(motionsense_root):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue

            csv_path = os.path.join(dirpath, filename)
            try:
                bpm = compute_bpm_from_motion_file(csv_path)
            except Exception:
                skipped_files += 1
                continue

            user_bpms.append(bpm)

    if not user_bpms:
        raise RuntimeError("Не удалось вычислить ни одного BPM из MotionSense CSV.")

    print(
        f"Загружено BPM из MotionSense: {len(user_bpms)}. "
        f"Пропущено файлов: {skipped_files}"
    , flush=True)
    return user_bpms


def normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # нормализуем только BPM-признаки, потому что energy и dance уже примерно в диапазоне 0..1
    means = X.mean(axis=0)
    stds = X.std(axis=0)

    normalize_idx = [0, 1, 2]
    stds[normalize_idx] = np.where(stds[normalize_idx] < 1e-8, 1.0, stds[normalize_idx])
    X[:, normalize_idx] = (X[:, normalize_idx] - means[normalize_idx]) / stds[normalize_idx]
    return X, means, stds


def build_sampler(labels: Iterable[float]) -> WeightedRandomSampler:
    labels_array = np.asarray(list(labels), dtype=np.float32)
    class_counts = np.bincount(labels_array.astype(int), minlength=2)
    class_weights = np.zeros(2, dtype=np.float64)

    for class_idx, count in enumerate(class_counts):
        class_weights[class_idx] = 0.0 if count == 0 else 1.0 / float(count)

    sample_weights = class_weights[labels_array.astype(int)]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compress_user_bpms(user_bpms: list[float], max_points: int = 80) -> list[float]:
    # убираем почти одинаковые BPM, чтобы обучение не разрасталось из-за сотен близких MotionSense-файлов
    unique_bpms = np.unique(np.round(np.asarray(user_bpms, dtype=np.float32), 1))
    if len(unique_bpms) <= max_points:
        return unique_bpms.astype(float).tolist()

    sampled_idx = np.linspace(0, len(unique_bpms) - 1, num=max_points, dtype=int)
    return unique_bpms[sampled_idx].astype(float).tolist()


def evaluate_epoch(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    pred_labels = (preds >= 0.5).float()
    accuracy = (pred_labels == targets).float().mean().item()

    positives = (targets == 1).float().sum().item()
    true_positives = ((pred_labels == 1) & (targets == 1)).float().sum().item()
    recall = 0.0 if positives == 0 else true_positives / positives
    return accuracy, recall


def evaluate_model(
    model: MatchNet,
    loader: DataLoader,
    pos_weight_value: float,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = model(batch_x)
            weights = torch.where(
                batch_y == 1,
                torch.full_like(batch_y, pos_weight_value),
                torch.ones_like(batch_y),
            )
            loss = nn.functional.binary_cross_entropy(preds, batch_y, weight=weights)
            total_loss += loss.item()
            all_preds.append(preds)
            all_targets.append(batch_y)

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    accuracy, recall = evaluate_epoch(preds_cat, targets_cat)
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, accuracy, recall


def main():  # точка входа в скрипт
    root_dir = os.path.dirname(os.path.dirname(__file__))  # определяет корень проекта как папку на уровень выше training
    data_path = os.path.join(root_dir, "data", "spotify_tracks_for_app_unique.csv")  # указывает на CSV с данными
    model_path = os.path.join(root_dir, "models", "matchnet.pt")  # указывает на файл модели
    stats_path = os.path.join(root_dir, "models", "matchnet_stats.pt")  # указывает на файл со статистикой нормализации
    motionsense_root = os.path.join(root_dir, "data", "A_DeviceMotion_data", "A_DeviceMotion_data")

    df = load_music(data_path)
    user_bpms = compress_user_bpms(load_user_bpms(motionsense_root))  # берет BPM из MotionSense и сжимает близкие значения
    prefs = ["high", "low", "neutral"]  # варианты предпочтения по энергии

    print(f"После сжатия используем {len(user_bpms)} опорных BPM.", flush=True)

    track_bpms = df["bpm"].to_numpy(dtype=np.float32)
    energies = df["energy"].to_numpy(dtype=np.float32)
    dances = df["dance"].to_numpy(dtype=np.float32)

    X_blocks: list[np.ndarray] = []  # будущие признаки
    y_blocks: list[np.ndarray] = []  # будущие метки

    for user_bpm in user_bpms:
        user_bpm_arr = np.full_like(track_bpms, fill_value=float(user_bpm), dtype=np.float32)
        bpm_diff = np.abs(user_bpm_arr - track_bpms)

        for pref in prefs:
            pref_val = np.float32(encode_energy_pref(pref))
            energy_alignment = 1.0 - np.abs(energies - pref_val)
            labels = ((bpm_diff <= 12.0) & (np.abs(energies - pref_val) <= 0.45)).astype(np.float32)  # формирует целевую метку по правилам

            X_blocks.append(
                np.column_stack(
                    [
                        user_bpm_arr,
                        track_bpms,
                        bpm_diff,
                        energy_alignment.astype(np.float32),
                        dances,
                    ]
                ).astype(np.float32)
            )
            y_blocks.append(labels)

    X_np = np.vstack(X_blocks)  # объединяет все признаки в одну матрицу
    y_np = np.concatenate(y_blocks)  # объединяет все метки в один вектор

    X_np, means, stds = normalize_features(X_np)  # нормализует BPM-признаки

    positives = int((y_np == 1).sum())
    negatives = int((y_np == 0).sum())
    positive_ratio = float(positives / len(y_np))

    print(
        f"Всего примеров: {len(y_np)} | "
        f"Положительных: {positives} | "
        f"Отрицательных: {negatives} | "
        f"positive_ratio: {positive_ratio:.4f}"
    , flush=True)

    indices = np.arange(len(y_np))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y_np.astype(int),
    )

    X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)  # превращает признаки train в тензор
    y_train = torch.tensor(y_np[train_idx].reshape(-1, 1), dtype=torch.float32)  # превращает метки train в тензор
    X_val = torch.tensor(X_np[val_idx], dtype=torch.float32)  # превращает признаки validation в тензор
    y_val = torch.tensor(y_np[val_idx].reshape(-1, 1), dtype=torch.float32)  # превращает метки validation в тензор

    train_positive_ratio = float((y_np[train_idx] == 1).mean())
    val_positive_ratio = float((y_np[val_idx] == 1).mean())
    print(
        f"train_size: {len(train_idx)} | val_size: {len(val_idx)} | "
        f"train_positive_ratio: {train_positive_ratio:.4f} | "
        f"val_positive_ratio: {val_positive_ratio:.4f}",
        flush=True,
    )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    sampler = build_sampler(y_np[train_idx])  # балансирует классы только на train
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    train_positives = int((y_np[train_idx] == 1).sum())
    train_negatives = int((y_np[train_idx] == 0).sum())

    model = MatchNet(input_dim=5)
    pos_weight_value = 1.0 if train_positives == 0 else max(1.0, train_negatives / max(1, train_positives))  # повышает вес положительного класса
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = int(os.environ.get("MATCHNET_EPOCHS", "15"))  # число эпох можно задать через переменную окружения
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            weights = torch.where(
                batch_y == 1,
                torch.full_like(batch_y, pos_weight_value),
                torch.ones_like(batch_y),
            )
            loss = nn.functional.binary_cross_entropy(preds, batch_y, weight=weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_accuracy, val_recall = evaluate_model(model, val_loader, pos_weight_value)

        print(
            f"Эпоха {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_accuracy: {val_accuracy:.4f} | "
            f"val_recall: {val_recall:.4f} | "
            f"train_positive_ratio: {train_positive_ratio:.4f} | "
            f"val_positive_ratio: {val_positive_ratio:.4f}"
        , flush=True)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)  # сохраняет веса модели
    torch.save(
        {
            "feature_indices": [0, 1, 2],
            "mean": means[[0, 1, 2]].tolist(),
            "std": stds[[0, 1, 2]].tolist(),
            "means": means.tolist(),
            "stds": stds.tolist(),
            "normalize_idx": [0, 1, 2],
        },
        stats_path,  # сохраняет статистику нормализации
    )

    print(f"Модель сохранена: {model_path}", flush=True)
    print(f"Статистика нормализации сохранена: {stats_path}", flush=True)


if __name__ == "__main__":
    main()
