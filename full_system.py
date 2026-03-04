import os
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import streamlit as st


@dataclass
class Config:
    motionsense_root: str = r"D:\документы\уник\диплом\datasets\A_DeviceMotion_data\A_DeviceMotion_data"
    music_csv: str = r"D:\документы\уник\диплом\datasets\spotify_tracks_for_app_unique.csv"

    sample_rate_hz: float = 50.0  # Частота дискретизации акселерометра (Гц)
    bandpass_low: float = 0.5  # Полоса пропускания фильтра (Гц), соответствующая шагам человека
    bandpass_high: float = 3.5

    bpm_tolerance_max: float = 40.0  # Максимально допустимая разница BPM между движением и треком
    top_n_default: int = 10


CFG = Config()


# поиск CSV-файлов MotionSense и группировка по активности
def list_activity_files(root: str) -> Dict[str, List[str]]:
    activity_files: Dict[str, List[str]] = {}

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Папка MotionSense не найдена: {root}")

    for dirpath, dirnames, filenames in os.walk(root):
        folder = os.path.basename(dirpath).lower()
        if not filenames:
            continue

        # определяем код активности по названию папки
        activity_code: Optional[str] = None
        for code in ["wlk", "jog", "run", "ups", "dws", "sit"]:
            if code in folder:
                activity_code = code
                break

        if activity_code is None:
            continue
        # сохраняем все CSV-файлы данной активности
        for f in filenames:
            if f.lower().endswith(".csv"):
                full_path = os.path.join(dirpath, f)
                activity_files.setdefault(activity_code, []).append(full_path)

    return activity_files


# проектирование полосового фильтра Баттерворта
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1:  # Защита от выхода за допустимый диапазон
        high = 0.99
    b, a = butter(order, [low, high], btype="band")
    return b, a


# применение фильтра к сигналу
def apply_bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    b, a = butter_bandpass(low, high, fs)
    filtered = filtfilt(b, a, signal)
    return filtered


def load_motion_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV MotionSense не найден: {path}")
    return pd.read_csv(path)


# вычисление модуля ускорения (|a|) из трёх осей
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
            mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)  # Евклидова норма ускорения
            return mag

    raise ValueError("Не удалось найти колонки акселерометра (x, y, z) в MotionSense CSV.")


# оценка BPM движения пользователя
def compute_bpm_from_motion_file(path: str, cfg: Config) -> float:
    df = load_motion_csv(path)
    mag = extract_acc_magnitude(df)

    mag = mag - np.mean(mag)  # центрирование сигнала (удаление постоянной составляющей)

    # фильтрация сигнала в диапазоне шагов человека
    filtered = apply_bandpass_filter(
        mag,
        fs=cfg.sample_rate_hz,
        low=cfg.bandpass_low,
        high=cfg.bandpass_high,
    )

    # поиск пиков (шагов)
    min_distance = int(cfg.sample_rate_hz * 0.3)
    peaks, _ = find_peaks(filtered, distance=min_distance)

    if len(peaks) < 2:
        raise RuntimeError("Недостаточно шагов в сигнале для оценки BPM.")

    # интервалы между шагами
    intervals = np.diff(peaks)
    median_interval = np.median(intervals)
    if median_interval <= 0:
        raise RuntimeError("Ошибка при оценке интервала между шагами.")

    # перевод в BPM
    step_period_sec = median_interval / cfg.sample_rate_hz
    bpm = 60.0 / step_period_sec
    return float(bpm)


def load_music_database(csv_path: str) -> pd.DataFrame:
    # загружает музыкальный каталог и проводит базовую очистку + удаление дублей
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Музыкальный каталог не найден: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["title", "artist", "genre", "bpm", "energy", "dance"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"В музыкальной базе отсутствуют обязательные колонки: {missing}")

    df = df.dropna(subset=required_cols)

    # приводим типы
    df["bpm"] = pd.to_numeric(df["bpm"], errors="coerce")
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df["dance"] = pd.to_numeric(df["dance"], errors="coerce")
    df = df.dropna(subset=["bpm", "energy", "dance"])

    # нормализуем текст, чтобы BTS и bts считались одним и тем же
    df["artist"] = df["artist"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["genre"] = df["genre"].astype(str).str.strip()

    # удаление дубликатов
    # если в датасете есть track_id — лучше по нему
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"], keep="first")
    else:
        # иначе по паре (исполнитель + название)
        df = df.drop_duplicates(subset=["artist", "title"], keep="first")

    return df.reset_index(drop=True)



def filter_by_genre(df: pd.DataFrame, genre_choice: str) -> pd.DataFrame:
    """Фильтрация по жанру (или возврат всех треков, если выбран «Любой жанр»)."""
    if genre_choice == "Любой жанр":
        return df.copy()
    mask = df["genre"].str.lower() == genre_choice.lower()
    filtered = df[mask]
    if filtered.empty:
        st.warning("Для выбранного жанра не найдено треков. Используются все треки.")
        return df.copy()
    return filtered.reset_index(drop=True)


# Фильтрация по исполнителям
def filter_by_artists(
        df: pd.DataFrame,
        include_artists: List[str],
        exclude_artists: List[str],
) -> pd.DataFrame:
    # Учитывает предпочтительных и исключаемых исполнителей
    result = df.copy()
    if include_artists:
        result = result[result["artist"].isin(include_artists)]
    if exclude_artists:
        result = result[~result["artist"].isin(exclude_artists)]
    return result.reset_index(drop=True)


# Интегральная оценка соответствия трека
def compute_match_score(
        user_bpm: float,
        track_bpm: float,
        energy: float,
        dance: float,
        energy_pref: str,
        cfg: Config,
) -> float:
    # Оценка совпадения BPM
    bpm_diff = abs(track_bpm - user_bpm)
    bpm_score = 1.0 - min(bpm_diff / cfg.bpm_tolerance_max, 1.0)

    # Учет предпочтений по энергетике
    energy_pref = energy_pref.lower()
    if "энергич" in energy_pref:
        energy_score = energy
    elif "спокой" in energy_pref:
        energy_score = 1.0 - energy
    else:
        energy_score = 0.5
    # Танцевальность
    dance_score = dance

    # Взвешенная сумма
    score = 0.6 * bpm_score + 0.25 * energy_score + 0.15 * dance_score
    return float(score)


def recommend_tracks(
    music_df: pd.DataFrame,
    user_bpm: float,
    energy_pref: str,
    top_n: int,
    cfg: Config,
) -> pd.DataFrame:
    music_df = music_df.copy()

    # страховка от дублей ДО скоринга
    if "track_id" in music_df.columns:
        music_df = music_df.drop_duplicates(subset=["track_id"], keep="first")
    else:
        music_df = music_df.drop_duplicates(subset=["artist", "title"], keep="first")

    scores = []
    for _, row in music_df.iterrows():
        scores.append(
            compute_match_score(
                user_bpm=user_bpm,
                track_bpm=row["bpm"],
                energy=row["energy"],
                dance=row["dance"],
                energy_pref=energy_pref,
                cfg=cfg,
            )
        )
    music_df["match_score"] = scores

    music_df = music_df.sort_values(by="match_score", ascending=False)

    # страховка от дублей ПОСЛЕ сортировки (на случай одинаковых ключей/грязных данных)
    if "track_id" in music_df.columns:
        music_df = music_df.drop_duplicates(subset=["track_id"], keep="first")
    else:
        music_df = music_df.drop_duplicates(subset=["artist", "title"], keep="first")

    return music_df.head(top_n).reset_index(drop=True)




def main():
    """
    Главная функция Streamlit-приложения.
    Отвечает за интерфейс пользователя и вызов логики рекомендаций.
    """

    # базовая настройка страницы приложения
    st.set_page_config(
        page_title="Система подбора треков по ритму движения",
        layout="wide",
    )

    # заголовок боковой панели
    st.sidebar.title("Настройки пользователя")

    # загрузка музыкальной базы
    try:
        music_df = load_music_database(CFG.music_csv)
    except Exception as e:
        st.error(f"Ошибка при загрузке музыкальной базы: {e}")
        return

    # поиск файлов MotionSense
    try:
        activity_files = list_activity_files(CFG.motionsense_root)
    except Exception as e:
        st.error(f"Ошибка при поиске файлов MotionSense: {e}")
        return

    if not activity_files:
        st.error("Не найдены CSV MotionSense ни для одной активности.")
        return

    # фиксированный порядок кодов и человекочитаемых названий
    activity_pairs = [
        ("wlk", "Ходьба"),
        ("jog", "Бег"),
        ("sit", "Сидение"),
        ("ups", "Подъём по лестнице"),
        ("dws", "Спуск по лестнице"),
    ]

    # оставляем только те активности, для которых реально есть файлы
    available_pairs = [(code, label) for code, label in activity_pairs if code in activity_files]

    if not available_pairs:
        st.error("Не найдено ни одной активности с CSV-файлами MotionSense.")
        return

    # формирование списков для выбора в интерфейсе
    display_labels = [label for _, label in available_pairs]
    code_by_label = {label: code for code, label in available_pairs}

    # выбор типа активности пользователем
    selected_label = st.sidebar.selectbox("Тип активности", display_labels)
    activity_code = code_by_label[selected_label]

    # выбор жанра
    genre_list = ["Любой жанр"] + sorted(music_df["genre"].dropna().unique())
    genre_choice = st.sidebar.selectbox("Жанр музыки", genre_list)

    # для выбора артистов сначала учитываем выбранный жанр
    music_for_artists = filter_by_genre(music_df, genre_choice)
    artist_options = sorted(music_for_artists["artist"].dropna().unique())

    include_artists = st.sidebar.multiselect(
        "Предпочитаемые исполнители",
        options=artist_options,
    )

    exclude_artists = st.sidebar.multiselect(
        "Исключить исполнителей",
        options=artist_options,
    )
    # предпочтения по интенсивности
    energy_pref = st.sidebar.radio(
        "Желаемый стиль треков по интенсивности и танцевальности:",
        [
            "Без предпочтений",
            "Более энергичные и ритмичные",
            "Более спокойные и ненавязчивые",
        ],
        index=0,
    )

    # количество рекомендаций
    top_n = st.sidebar.number_input(
        "Сколько треков отобразить",
        min_value=1,
        max_value=50,
        value=CFG.top_n_default,
        step=1,
    )

    # основная часть интерфейса
    st.title("Интеллектуальная система подбора музыкальных треков по ритму движения пользователя")
    st.write(
        "Выберите параметры слева и нажмите кнопку ниже, чтобы система подобрала треки "
        "под ритм вашего движения."
    )

    # запуск подбора
    if st.button("Подобрать треки"):
        # выбираем случайный файл движения для выбранной активности
        motion_file = random.choice(activity_files[activity_code])
        st.info(f"Используется файл движения: `{motion_file}`")

        # вычисление BPM с обработкой ошибок
        try:
            user_bpm = compute_bpm_from_motion_file(motion_file, CFG)
        except Exception as e:
            st.error(f"Ошибка при вычислении BPM: {e}")
            return

        st.success(f"Рассчитанный BPM движения: {user_bpm:.1f}")

        # фильтрация по жанру и артистам
        music_filtered = filter_by_genre(music_df, genre_choice)
        music_filtered = filter_by_artists(music_filtered, include_artists, exclude_artists)

        if music_filtered.empty:
            st.warning("После фильтрации по жанру и исполнителям не осталось ни одного трека.")
            return

        # формирование рекомендаций
        try:
            recs = recommend_tracks(
                music_df=music_filtered,
                user_bpm=user_bpm,
                energy_pref=energy_pref,
                top_n=int(top_n),
                cfg=CFG,
            )
        except Exception as e:
            st.error(f"Ошибка при формировании рекомендаций: {e}")
            return
        # вывод результатов
        st.subheader("Рекомендованные треки")
        st.dataframe(
            recs[["artist", "title", "genre", "bpm", "energy", "dance", "match_score"]],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
