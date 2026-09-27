"""
Dashboard Visualization
=======================

Графики для дашборда в стиле печатного отчёта: бумажный фон,
сдержанный сине-оливково-песочный палитра, читаемые подписи.

Файлы results/01..04 — классические графики пайплайна,
results/05 — динамика всех тренировок,
results/06 — счёт побед моделей,
results/07 — кривая обучения.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Палитра «печатного отчёта» — без кислотных цветов
PAPER = "#f4f1e8"      # бумага
INK = "#1d2733"        # текст
STEEL = "#3d5a80"      # синий (сталь)
OLIVE = "#6b7f4e"      # оливковый (успех / лучшая модель)
SAND = "#c9a227"       # песочно-золотой (акцент)
RUST = "#a0522d"       # рыжевато-коричневый (ошибки)
GRID = "#d8d3c4"       # линии сетки

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": "#9a9484",
    "axes.titlecolor": INK,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 130,
})


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def _save(fig, name):
    _ensure_dir()
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _style_axes(ax, title=None, xlabel=None, ylabel=None, legend=True):
    ax.grid(True, alpha=0.5, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------
# 01 — Данные + модель
# --------------------------------------------------------------------------

def plot_data_and_model(X_train, y_train, X_test, y_test, pipeline,
                        feature_col, target_col, out_name="01_data_and_model.png"):
    if feature_col is None:
        return None

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.scatter(X_train[feature_col], y_train, s=34, color=STEEL, alpha=0.75,
               edgecolor=INK, linewidth=0.5, label="Обучающая выборка")
    ax.scatter(X_test[feature_col], y_test, s=34, color=RUST, alpha=0.85,
               marker="s", edgecolor=INK, linewidth=0.5, label="Тестовая выборка")

    x_line = np.linspace(X_train[feature_col].min(), X_train[feature_col].max(), 200)
    line_df = {feature_col: x_line}
    for col in X_train.columns:
        if col != feature_col:
            line_df[col] = 0
    import pandas as pd
    y_line = pipeline.predict(pd.DataFrame(line_df))
    ax.plot(x_line, y_line, color=OLIVE, linewidth=2.4, label="Модель (прогноз)")

    _style_axes(ax, title=f"{target_col} в зависимости от {feature_col}",
                xlabel=feature_col, ylabel=target_col)
    return _save(fig, out_name)


# --------------------------------------------------------------------------
# 02 — Actual vs Predicted
# --------------------------------------------------------------------------

def plot_actual_vs_predicted(y_test, y_pred, task_type,
                             out_name="02_actual_vs_predicted.png"):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.scatter(y_test, y_pred, s=34, color=STEEL, alpha=0.75,
               edgecolor=INK, linewidth=0.5, label="Наблюдения")

    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    pad = (hi - lo) * 0.05 or 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--",
            color=OLIVE, linewidth=1.8, label="Идеальное совпадение (y = x)")

    _style_axes(ax, title="Фактические vs предсказанные значения",
                xlabel="Факт", ylabel="Прогноз")
    ax.set_aspect("equal", adjustable="box")
    return _save(fig, out_name)


# --------------------------------------------------------------------------
# 03 — Сравнение моделей
# --------------------------------------------------------------------------

def plot_model_comparison(model_names, cv_scores, cv_stds, scoring,
                          out_name="03_model_comparison.png"):
    fig, ax = plt.subplots(figsize=(9.4, 5.2))

    colors = [OLIVE if s == max(cv_scores) else STEEL for s in cv_scores]
    y_pos = np.arange(len(model_names))
    ax.barh(y_pos, cv_scores, xerr=cv_stds, height=0.6, color=colors,
            edgecolor=INK, linewidth=0.6, capsize=4,
            error_kw={"ecolor": INK, "linewidth": 0.9})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names)
    ax.invert_yaxis()

    for i, s in enumerate(cv_scores):
        ax.text(s + 0.004, i, f"{s:.4f}", va="center", fontsize=10,
                fontweight="bold", color=INK)

    _style_axes(ax, title=f"Сравнение моделей — кросс-валидация ({scoring.upper()})",
                xlabel=f"Средний {scoring.upper()} по {5} fold", legend=False, ylabel="")
    return _save(fig, out_name)


# --------------------------------------------------------------------------
# 04 — Распределение ошибок
# --------------------------------------------------------------------------

def plot_error_distribution(y_test, y_pred, out_name="04_error_distribution.png"):
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].hist(errors, bins=min(20, max(8, len(errors) // 3)), color=STEEL,
                 edgecolor=INK, linewidth=0.6, alpha=0.9)
    axes[0].axvline(x=0, color=RUST, linestyle="--", linewidth=1.8, label="Нет ошибки")
    _style_axes(axes[0], title="Распределение ошибок", xlabel="Ошибка (факт − прогноз)",
                ylabel="Частота")

    axes[1].bar(range(len(errors)), errors, color=STEEL, edgecolor=INK,
                linewidth=0.5, alpha=0.9)
    axes[1].axhline(y=0, color=RUST, linestyle="--", linewidth=1.6)
    _style_axes(axes[1], title="Ошибка по наблюдениям", xlabel="№ наблюдения",
                ylabel="Ошибка", legend=False)

    fig.suptitle("Анализ ошибок лучшей модели", fontweight="bold", y=1.02)
    return _save(fig, out_name)


# --------------------------------------------------------------------------
# 05 — Динамика тренировок
# --------------------------------------------------------------------------

def plot_history(trend, out_name="05_training_history.png"):
    if not trend:
        return None

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    scores = [t["score"] for t in trend]
    xs = list(range(1, len(scores) + 1))

    ax.plot(xs, scores, color=STEEL, linewidth=1.6, alpha=0.55, label="Показатель прогона")
    ax.scatter(xs, scores, s=26, color=STEEL, edgecolor=INK, linewidth=0.5, zorder=3)

    # скользящее среднее (окно 5)
    if len(scores) >= 5:
        window = min(5, len(scores))
        ma = np.convolve(scores, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(scores) + 1), ma, color=OLIVE, linewidth=2.4,
                label="Скользящее среднее (5)")

    best_i = int(np.argmax(scores)) + 1
    ax.scatter([best_i], [max(scores)], s=110, color=SAND, edgecolor=INK,
               linewidth=0.8, zorder=5, label="Лучший результат")
    ax.annotate(f"{max(scores):.4f}", (best_i, max(scores)),
                textcoords="offset points", xytext=(8, 8),
                fontweight="bold", color=INK)

    _style_axes(ax, title="Динамика качества по прогонам",
                xlabel="№ тренировки", ylabel="CV-показатель")
    return _save(fig, out_name)


# --------------------------------------------------------------------------
# 06 — Счёт побед моделей
# --------------------------------------------------------------------------

def plot_model_wins(winners, out_name="06_model_wins.png"):
    if not winners:
        return None
    names = sorted(winners, key=winners.get, reverse=True)
    values = [winners[n] for n in names]

    fig, ax = plt.subplots(figsize=(9.4, max(3.6, 0.55 * len(names) + 1.4)))
    colors = [OLIVE if i == 0 else STEEL for i in range(len(names))]
    ax.barh(range(len(names)), values, color=colors, edgecolor=INK,
            linewidth=0.6, height=0.62)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), va="center",
                fontweight="bold", color=INK)
    _style_axes(ax, title="Сколько раз модель становилась лучшей",
                xlabel="Количество побед", legend=False, ylabel="")
    return _save(fig, out_name)


# --------------------------------------------------------------------------
# 07 — Кривая обучения
# --------------------------------------------------------------------------

def plot_learning_curve(curve, scoring, out_name="07_learning_curve.png"):
    """
    curve: [{"model": name, "sizes": [...], "scores": [...]}, ...]
    """
    if not curve:
        return None
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    palette = [STEEL, OLIVE, SAND, RUST, "#5c6b8a"]

    for i, series in enumerate(curve):
        ax.plot(series["sizes"], series["scores"],
                marker="o", markersize=5, linewidth=1.8,
                color=palette[i % len(palette)], label=series["model"])

    _style_axes(ax, title=f"Кривая обучения — влияние объёма данных ({scoring.upper()})",
                xlabel="Размер обучающей выборки", ylabel=f"{scoring.upper()}")
    return _save(fig, out_name)


# --------------------------------------------------------------------------
# Сборный конвейер графиков одного прогона
# --------------------------------------------------------------------------

def charts_for_run(data, results, best_name, metrics, y_test, y_pred,
                   feature_col, target, task_type):
    """Строит графики 01–04 и возвращает список путей."""
    paths = []
    _ = metrics

    if task_type == "regression" and feature_col is not None:
        p = plot_data_and_model(
            data["X_train"], data["y_train"], data["X_test"], data["y_test"],
            results[best_name]["pipeline"], feature_col, target)
        if p:
            paths.append(p)

    if y_test is not None and y_pred is not None:
        p = plot_actual_vs_predicted(y_test, y_pred, task_type)
        if p:
            paths.append(p)

    model_names = list(results.keys())
    cv_scores = [results[n]["cv_mean"] for n in model_names]
    cv_stds = [results[n]["cv_std"] for n in model_names]
    scoring = results[best_name].get("scoring", "r2" if task_type == "regression" else "f1")
    p = plot_model_comparison(model_names, cv_scores, cv_stds, scoring)
    if p:
        paths.append(p)

    if task_type == "regression" and y_test is not None:
        p = plot_error_distribution(y_test, y_pred)
        if p:
            paths.append(p)

    return paths
