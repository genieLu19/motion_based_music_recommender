import json
import os
import sys

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy.signal import butter, filtfilt, find_peaks


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

print("CURRENT_DIR =", CURRENT_DIR)
print("ROOT_DIR =", ROOT_DIR)
print("MODELS_DIR =", MODELS_DIR)
print("models exists =", os.path.isdir(MODELS_DIR))

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from matchnet import MatchNet


@dataclass
class Config:
    motionsense_root: str = os.path.join(ROOT_DIR, "data", "A_DeviceMotion_data")
    music_csv: str = os.path.join(ROOT_DIR, "data", "spotify_tracks_for_app_unique.csv")
    model_path: str = os.path.join(ROOT_DIR, "models", "matchnet.pt")
    stats_path: str = os.path.join(ROOT_DIR, "models", "matchnet_stats.pt")
    preferences_path: str = os.path.join(ROOT_DIR, "app", "user_preferences.json")

    sample_rate_hz: float = 50.0
    bandpass_low: float = 0.5
    bandpass_high: float = 3.5

    bpm_tolerance_max: float = 40.0
    top_n_default: int = 10
    tiny_model_score_threshold: float = 1e-4
    tiny_model_score_share: float = 0.8
    model_weight: float = 0.20
    baseline_weight: float = 0.80


CFG = Config()


def list_activity_files(root: str) -> Dict[str, List[str]]:
    activity_files: Dict[str, List[str]] = {}

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Папка MotionSense не найдена: {root}")

    for dirpath, _, filenames in os.walk(root):
        folder = os.path.basename(dirpath).lower()
        if not filenames:
            continue

        activity_code: Optional[str] = None
        for code in ["wlk", "jog", "run", "ups", "dws"]:
            if code in folder:
                activity_code = code
                break

        if activity_code is None:
            continue

        for f in filenames:
            if f.lower().endswith(".csv"):
                full_path = os.path.join(dirpath, f)
                activity_files.setdefault(activity_code, []).append(full_path)

    return activity_files


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if high >= 1:
        high = 0.99

    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs)
    return filtfilt(b, a, signal)


def load_motion_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV MotionSense не найден: {path}")
    return pd.read_csv(path)


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
            return np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    raise ValueError("Не удалось найти колонки акселерометра (x, y, z) в MotionSense CSV.")


def compute_bpm_from_motion_file(path: str, cfg: Config) -> float:
    df = load_motion_csv(path)
    mag = extract_acc_magnitude(df)
    mag = mag - np.mean(mag)

    filtered = apply_bandpass_filter(
        mag,
        fs=cfg.sample_rate_hz,
        low=cfg.bandpass_low,
        high=cfg.bandpass_high,
    )

    min_distance = int(cfg.sample_rate_hz * 0.3)
    peaks, _ = find_peaks(filtered, distance=min_distance)

    if len(peaks) < 2:
        raise RuntimeError("Недостаточно шагов в сигнале для оценки BPM.")

    intervals = np.diff(peaks)
    median_interval = np.median(intervals)

    if median_interval <= 0:
        raise RuntimeError("Ошибка при оценке интервала между шагами.")

    step_period_sec = median_interval / cfg.sample_rate_hz
    bpm = 60.0 / step_period_sec
    return float(bpm)


def compute_activity_bpm(file_paths: List[str], cfg: Config) -> tuple[float, int, int]:
    bpm_values = []
    skipped_files = 0

    for path in file_paths:
        try:
            bpm_values.append(compute_bpm_from_motion_file(path, cfg))
        except Exception:
            skipped_files += 1

    if not bpm_values:
        raise RuntimeError("Не удалось вычислить BPM ни для одного файла выбранной активности.")

    median_bpm = float(np.median(bpm_values))
    return median_bpm, len(bpm_values), skipped_files


def load_music_database(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Музыкальный каталог не найден: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["title", "artist", "genre", "bpm", "energy", "dance"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"В музыкальной базе отсутствуют обязательные колонки: {missing}")

    df = df.dropna(subset=required_cols).copy()

    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df["dance"] = pd.to_numeric(df["dance"], errors="coerce")
    df = df.dropna(subset=["bpm", "energy", "dance"]).copy()

    df["artist"] = df["artist"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["genre"] = df["genre"].astype(str).str.strip()

    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"], keep="first")
    else:
        df = df.drop_duplicates(subset=["artist", "title"], keep="first")

    return df.reset_index(drop=True)


def filter_by_genre(df: pd.DataFrame, genre_choice: str) -> pd.DataFrame:
    if genre_choice == "Любой жанр":
        return df.copy()

    mask = df["genre"].str.lower() == genre_choice.lower()
    filtered = df[mask]

    if filtered.empty:
        st.warning("Для выбранного жанра не найдено треков. Используются все треки.")
        return df.copy()

    return filtered.reset_index(drop=True)


def filter_by_artists(
    df: pd.DataFrame,
    include_artists: List[str],
    exclude_artists: List[str],
) -> pd.DataFrame:
    result = df.copy()

    if include_artists:
        result = result[result["artist"].isin(include_artists)]

    if exclude_artists:
        result = result[~result["artist"].isin(exclude_artists)]

    return result.reset_index(drop=True)


def load_user_preferences(path: str) -> dict:
    if not os.path.isfile(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    return data if isinstance(data, dict) else {}


def save_user_preferences(path: str, preferences: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(preferences, file, ensure_ascii=False, indent=2)


def sanitize_saved_multiselect(values: List[str], available_options: List[str]) -> List[str]:
    available_set = set(available_options)
    return [value for value in values if value in available_set]


def encode_energy_pref(energy_pref: str) -> float:
    energy_pref = energy_pref.lower()

    if "энергич" in energy_pref:
        return 1.0
    if "спокой" in energy_pref:
        return 0.0
    return 0.5


def build_match_features(
    user_bpm: float,
    track_bpm: float,
    energy: float,
    dance: float,
    energy_pref: str,
) -> List[float]:
    pref_val = encode_energy_pref(energy_pref)
    energy_alignment = 1.0 - abs(float(energy) - pref_val)

    return [
        float(user_bpm),
        float(track_bpm),
        float(abs(user_bpm - track_bpm)),
        float(energy_alignment),
        float(dance),
    ]


def load_normalization_stats(stats_path: str) -> Optional[dict]:
    if not os.path.isfile(stats_path):
        return None

    stats = torch.load(stats_path, map_location="cpu")
    if {"feature_indices", "mean", "std"}.issubset(stats):
        return stats

    # Совместимость с новым форматом из train_matchnet.py.
    if {"normalize_idx", "means", "stds"}.issubset(stats):
        normalize_idx = list(stats["normalize_idx"])
        means = list(stats["means"])
        stds = list(stats["stds"])
        return {
            "feature_indices": normalize_idx,
            "mean": [means[idx] for idx in normalize_idx],
            "std": [stds[idx] for idx in normalize_idx],
        }

    return None


def apply_feature_normalization(features: List[float], stats: Optional[dict]) -> List[float]:
    if not stats:
        return features

    normalized = list(features)
    feature_indices = stats["feature_indices"]
    mean = stats["mean"]
    std = stats["std"]

    for offset, feature_index in enumerate(feature_indices):
        denom = std[offset] if std[offset] != 0 else 1.0
        normalized[feature_index] = (normalized[feature_index] - mean[offset]) / denom

    return normalized


def load_matchnet_model(model_path: str) -> MatchNet:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "Файл модели MatchNet не найден.\n"
            f"Ожидаемый путь: {model_path}\n\n"
            "Сначала обучите модель командой:\n"
            "python training/train_matchnet.py"
        )

    model = MatchNet(input_dim=5)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_match_score(
    model: MatchNet,
    user_bpm: float,
    track_bpm: float,
    energy: float,
    dance: float,
    energy_pref: str,
    norm_stats: Optional[dict],
) -> float:
    features = build_match_features(
        user_bpm=user_bpm,
        track_bpm=track_bpm,
        energy=energy,
        dance=dance,
        energy_pref=energy_pref,
    )
    features = apply_feature_normalization(features, norm_stats)

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        score = model(x).item()

    return float(score)


def blend_match_scores(model_score: float, baseline_score: float) -> float:
    return CFG.model_weight * float(model_score) + CFG.baseline_weight * float(baseline_score)


def get_activity_rules(activity_code: str) -> dict:
    rules = {
        "wlk": {"max_bpm_diff": 8.0, "min_match_score": 0.45},
        "jog": {"max_bpm_diff": 8.0, "min_match_score": 0.40},
        "ups": {"max_bpm_diff": 9.0, "min_match_score": 0.42},
        "dws": {"max_bpm_diff": 9.0, "min_match_score": 0.42},
        "run": {"max_bpm_diff": 10.0, "min_match_score": 0.38},
    }
    return rules.get(activity_code, {"max_bpm_diff": 12.0, "min_match_score": 0.40})


def is_model_degenerate(model_scores: pd.Series, cfg: Config) -> bool:
    if model_scores.empty:
        return False

    tiny_share = float((model_scores.abs() < cfg.tiny_model_score_threshold).mean())
    return tiny_share >= cfg.tiny_model_score_share


def describe_bpm_similarity(bpm_diff: float) -> str:
    bpm_diff = float(bpm_diff)
    if bpm_diff <= 3:
        return "Очень высокая"
    if bpm_diff <= 6:
        return "Высокая"
    if bpm_diff <= 10:
        return "Средняя"
    if bpm_diff <= 15:
        return "Низкая"
    return "Очень низкая"


def format_recommendations_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    display_df["bpm"] = display_df["bpm"].map(lambda value: round(float(value), 1))
    display_df["match_score"] = display_df["match_score"].map(lambda value: round(float(value), 4))
    display_df["Схожесть BPM"] = display_df["bpm_diff"].map(describe_bpm_similarity)
    display_df = display_df.rename(
        columns={
            "artist": "Исполнитель",
            "title": "Трек",
            "genre": "Жанр",
            "bpm": "BPM",
            "match_score": "Оценка совпадения",
        }
    )
    return display_df


def estimate_recommendation_confidence(recs: pd.DataFrame, activity_code: str) -> tuple[str, str]:
    if recs.empty:
        return "Низкая", "Подходящие треки не найдены после фильтрации кандидатов."

    rules = get_activity_rules(activity_code)
    top_score = float(recs.iloc[0]["match_score"])
    mean_diff = float(recs["bpm_diff"].head(min(5, len(recs))).mean())
    min_required = float(rules["min_match_score"])

    if top_score >= max(0.75, min_required + 0.2) and mean_diff <= 6:
        return "Высокая", "Лучшие треки хорошо совпадают с темпом движения и проходят строгий отбор."
    if top_score >= max(0.55, min_required + 0.1) and mean_diff <= 10:
        return "Средняя", "Есть несколько подходящих кандидатов, но совпадение по темпу не идеальное."
    return "Низкая", "Даже лучшие треки совпадают с движением умеренно, поэтому рекомендации менее надёжны."


def compute_match_score_baseline(
    user_bpm: float,
    track_bpm: float,
    energy: float,
    dance: float,
    energy_pref: str,
    cfg: Config,
) -> float:
    bpm_diff = abs(track_bpm - user_bpm)
    bpm_score = 1.0 - min(bpm_diff / cfg.bpm_tolerance_max, 1.0)

    energy_pref = energy_pref.lower()
    if "энергич" in energy_pref:
        energy_score = energy
    elif "спокой" in energy_pref:
        energy_score = 1.0 - energy
    else:
        energy_score = 0.5

    dance_score = dance
    score = 0.75 * bpm_score + 0.15 * energy_score + 0.10 * dance_score
    return float(score)


def recommend_tracks(
    music_df: pd.DataFrame,
    user_bpm: float,
    activity_code: str,
    energy_pref: str,
    top_n: int,
    model: MatchNet,
    norm_stats: Optional[dict],
) -> pd.DataFrame:
    music_df = music_df.copy()
    rules = get_activity_rules(activity_code)

    if "track_id" in music_df.columns:
        music_df = music_df.drop_duplicates(subset=["track_id"], keep="first")
    else:
        music_df = music_df.drop_duplicates(subset=["artist", "title"], keep="first")

    music_df["bpm_diff"] = (music_df["bpm"].astype(float) - float(user_bpm)).abs()
    music_df = music_df[music_df["bpm_diff"] <= float(rules["max_bpm_diff"])].copy()

    if music_df.empty:
        return music_df

    scores = []
    model_scores = []
    baseline_scores = []
    for _, row in music_df.iterrows():
        model_score = predict_match_score(
            model=model,
            user_bpm=user_bpm,
            track_bpm=row["bpm"],
            energy=row["energy"],
            dance=row["dance"],
            energy_pref=energy_pref,
            norm_stats=norm_stats,
        )
        baseline_score = compute_match_score_baseline(
            user_bpm=user_bpm,
            track_bpm=row["bpm"],
            energy=row["energy"],
            dance=row["dance"],
            energy_pref=energy_pref,
            cfg=CFG,
        )
        score = blend_match_scores(model_score, baseline_score)

        model_scores.append(model_score)
        baseline_scores.append(baseline_score)
        scores.append(score)

    music_df["model_score"] = model_scores
    music_df["baseline_score"] = baseline_scores
    music_df["match_score"] = scores
    music_df = music_df[music_df["match_score"] >= float(rules["min_match_score"])].copy()

    if music_df.empty:
        return music_df

    music_df = music_df.sort_values(by="match_score", ascending=False)

    if "track_id" in music_df.columns:
        music_df = music_df.drop_duplicates(subset=["track_id"], keep="first")
    else:
        music_df = music_df.drop_duplicates(subset=["artist", "title"], keep="first")

    return music_df.head(top_n).reset_index(drop=True)


def main():
    st.set_page_config(
        page_title="Система подбора треков по ритму движения",
        layout="wide",
    )

    st.sidebar.title("Настройки пользователя")

    try:
        music_df = load_music_database(CFG.music_csv)
    except Exception as e:
        st.error(f"Ошибка при загрузке музыкальной базы: {e}")
        return

    try:
        matchnet_model = load_matchnet_model(CFG.model_path)
    except Exception as e:
        st.error(f"Ошибка при загрузке MatchNet: {e}")
        st.stop()

    norm_stats = load_normalization_stats(CFG.stats_path)

    try:
        activity_files = list_activity_files(CFG.motionsense_root)
    except Exception as e:
        st.error(f"Ошибка при поиске файлов MotionSense: {e}")
        return

    preferences = load_user_preferences(CFG.preferences_path)
    use_saved_preferences = st.sidebar.checkbox(
        "Использовать предыдущий ввод",
        value=bool(preferences),
    )

    if not activity_files:
        st.error("Не найдены CSV MotionSense ни для одной активности.")
        return

    activity_pairs = [
        ("wlk", "Ходьба"),
        ("jog", "Бег"),
        ("ups", "Подъём по лестнице"),
        ("dws", "Спуск по лестнице"),
    ]

    available_pairs = [(code, label) for code, label in activity_pairs if code in activity_files]

    if not available_pairs:
        st.error("Не найдено ни одной активности с CSV-файлами MotionSense.")
        return

    display_labels = [label for _, label in available_pairs]
    code_by_label = {label: code for code, label in available_pairs}

    saved_activity_code = preferences.get("activity_code") if use_saved_preferences else None
    saved_activity_label = next(
        (label for code, label in available_pairs if code == saved_activity_code),
        display_labels[0],
    )
    activity_index = display_labels.index(saved_activity_label)
    selected_label = st.sidebar.selectbox("Тип активности", display_labels, index=activity_index)
    activity_code = code_by_label[selected_label]

    genre_list = ["Любой жанр"] + sorted(music_df["genre"].dropna().unique())
    saved_genre = preferences.get("genre_choice", "Любой жанр") if use_saved_preferences else "Любой жанр"
    if saved_genre not in genre_list:
        saved_genre = "Любой жанр"
    genre_index = genre_list.index(saved_genre)
    genre_choice = st.sidebar.selectbox("Жанр музыки", genre_list, index=genre_index)

    music_for_artists = filter_by_genre(music_df, genre_choice)
    artist_options = sorted(music_for_artists["artist"].dropna().unique())
    saved_include_artists = sanitize_saved_multiselect(
        preferences.get("include_artists", []) if use_saved_preferences else [],
        artist_options,
    )
    saved_exclude_artists = sanitize_saved_multiselect(
        preferences.get("exclude_artists", []) if use_saved_preferences else [],
        artist_options,
    )

    include_artists = st.sidebar.multiselect(
        "Предпочитаемые исполнители",
        options=artist_options,
        default=saved_include_artists,
    )

    exclude_artists = st.sidebar.multiselect(
        "Исключить исполнителей",
        options=artist_options,
        default=saved_exclude_artists,
    )

    energy_options = [
        "Без предпочтений",
        "Более энергичные и ритмичные",
        "Более спокойные и ненавязчивые",
    ]
    saved_energy_pref = preferences.get("energy_pref", energy_options[0]) if use_saved_preferences else energy_options[0]
    if saved_energy_pref not in energy_options:
        saved_energy_pref = energy_options[0]
    energy_index = energy_options.index(saved_energy_pref)
    energy_pref = st.sidebar.radio(
        "Желаемый стиль треков по интенсивности и танцевальности:",
        energy_options,
        index=energy_index,
    )

    save_user_preferences(
        CFG.preferences_path,
        {
            "activity_code": activity_code,
            "genre_choice": genre_choice,
            "include_artists": include_artists,
            "exclude_artists": exclude_artists,
            "energy_pref": energy_pref,
        },
    )

    st.title("Интеллектуальная система подбора музыкальных треков по ритму движения пользователя")
    st.write(
        "Выберите параметры слева и нажмите кнопку ниже, чтобы система подобрала треки "
        "под ритм вашего движения."
    )

    if st.button("Подобрать треки"):
        try:
            user_bpm, used_files, skipped_files = compute_activity_bpm(
                activity_files[activity_code],
                CFG,
            )
        except Exception as e:
            st.error(f"Ошибка при вычислении BPM: {e}")
            return

        st.info(
            f"BPM рассчитан по нескольким файлам активности: использовано {used_files}, "
            f"пропущено {skipped_files}."
        )
        st.success(f"Рассчитанный BPM движения: {user_bpm:.1f}")

        music_filtered = filter_by_genre(music_df, genre_choice)
        music_filtered = filter_by_artists(music_filtered, include_artists, exclude_artists)

        if music_filtered.empty:
            st.warning("После фильтрации по жанру и исполнителям не осталось ни одного трека.")
            return

        try:
            recs = recommend_tracks(
                music_df=music_filtered,
                user_bpm=user_bpm,
                activity_code=activity_code,
                energy_pref=energy_pref,
                top_n=CFG.top_n_default,
                model=matchnet_model,
                norm_stats=norm_stats,
            )
        except Exception as e:
            st.error(f"Ошибка при формировании рекомендаций: {e}")
            return

        if recs.empty:
            st.warning(
                "Подходящих треков не найдено после отбора кандидатов. "
                "Попробуйте другой жанр, измените список исполнителей или выберите другую активность."
            )
            return

        confidence_label, confidence_message = estimate_recommendation_confidence(recs, activity_code)
        if confidence_label == "Высокая":
            st.success(f"Уверенность рекомендаций: {confidence_label}. {confidence_message}")
        elif confidence_label == "Средняя":
            st.info(f"Уверенность рекомендаций: {confidence_label}. {confidence_message}")
        else:
            st.warning(f"Уверенность рекомендаций: {confidence_label}. {confidence_message}")

        recs_display = format_recommendations_for_display(recs)

        st.subheader("Рекомендованные треки")
        st.dataframe(
            recs_display[
                [
                    "Исполнитель",
                    "Трек",
                    "Жанр",
                    "BPM",
                    "Схожесть BPM",
                    "Оценка совпадения",
                ]
            ],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
