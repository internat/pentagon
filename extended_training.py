"""
Расширенная тренировочная сессия (Extended Training Session)
============================================================

Непрерывное обучение алгоритма в течение НЕ МЕНЕЕ 40 минут.

Что делает сессия:
  1. Проход по реальным датасетам: все доли разбиения (70/30 … 95/5),
     все значения k для k-fold кросс-валидации (3/5/7/10).
  2. Синтетические датасеты разного масштаба (2 000 – 20 000 строк,
     3 – 10 признаков), регрессия и классификация, линейные и
     нелинейные связи, дисбаланс классов.
  3. Расширенный перебор гиперпараметров Random Forest
     (n_estimators × max_depth).
  4. Кривые обучения: как объём данных влияет на качество.
  5. Проверка устойчивости к random_state (стабильность результатов).

Каждый прогон записывается в results/training_history.json,
статус сессии — в results/extended_session.json (для дашборда).

Запуск:  python extended_training.py
Останов: Ctrl+C  (сессия корректно закроется со статусом "interrupted")
"""

import os
import sys
import time
import json
import random
from datetime import datetime

# Windows-консоль (cp1251) не умеет символы вроде «×»/«±» — принудительно UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import (
    train_stream, load_dataframe, suggest_target, append_history,
    sanitize, RESULTS_DIR, DATA_DIR, SESSION_PATH,
)
from src.preprocessing import preprocess_data, detect_task_type
from src.models import get_models

MIN_MINUTES = 40
MIN_SECONDS = MIN_MINUTES * 60

CHART_EVERY_RUNS = 12  # перерисовывать сводные графики каждые N прогонов


# --------------------------------------------------------------------------
# Session status
# --------------------------------------------------------------------------

def write_session(session):
    session["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = SESSION_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(sanitize(session), fh, ensure_ascii=False, indent=1)
    os.replace(tmp, SESSION_PATH)


# --------------------------------------------------------------------------
# Синтетические датасеты (примеры данных для дашборда + питание сессии)
# --------------------------------------------------------------------------

def make_regression_df(n=2000, n_features=6, noise=1.0, seed=42,
                       nonlinear=True, categorical=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, n_features))
    y = 4.0 * X[:, 0] + 2.0 * X[:, 1] - 1.5 * X[:, 2]
    if nonlinear and n_features >= 3:
        y = y + 3.0 * X[:, 2] ** 2 - 2.5 * np.sin(2.0 * X[:, 3 % n_features])
    y = y + noise * rng.normal(0, 1, size=n)

    data = {f"x{i + 1}": X[:, i] for i in range(n_features)}
    if categorical:
        groups = np.array(["A", "B", "C"])[rng.integers(0, 3, size=n)]
        y = y + np.where(groups == "A", 3.0, np.where(groups == "B", 0.0, -3.0))
        data["group"] = groups
    data["target"] = np.round(y, 3)
    return pd.DataFrame(data)


def make_classification_df(n=1500, n_features=6, seed=42, nonlinear=True,
                           imbalance=0.5, classes=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, n_features))
    score = 2.2 * X[:, 0] + 1.4 * X[:, 1] - 1.1 * X[:, 2]
    if nonlinear:
        score = score + 2.0 * X[:, 3 % n_features] ** 2 - 1.5 * np.sin(2 * X[:, 1])
    score = score + 0.7 * rng.normal(0, 1, size=n)

    if classes == 2:
        thr = np.quantile(score, 1.0 - imbalance)
        y = (score >= thr).astype(int)
        target_name = "label"
    else:
        q1, q2 = np.quantile(score, [1 / 3, 2 / 3])
        y = np.digitize(score, [q1, q2])
        target_name = "segment"

    data = {f"x{i + 1}": X[:, i] for i in range(n_features)}
    data["group"] = np.array(["A", "B", "C"])[rng.integers(0, 3, size=n)]
    data[target_name] = y
    return pd.DataFrame(data)


def ensure_example_datasets():
    """Создаёт 3 синтетических датасета-примера, если их ещё нет в data/."""
    specs = [
        ("synth_regression_2000.csv",
         lambda: make_regression_df(2000, 6, noise=1.0, seed=42,
                                    nonlinear=True, categorical=True)),
        ("synth_nonlinear_3000.csv",
         lambda: make_regression_df(3000, 8, noise=1.5, seed=7,
                                    nonlinear=True, categorical=False)),
        ("synth_fraud_1500.csv",
         lambda: make_classification_df(1500, 6, seed=11, nonlinear=True,
                                         imbalance=0.15, classes=2)),
    ]
    created = []
    for name, factory in specs:
        path = os.path.join(DATA_DIR, name)
        if not os.path.isfile(path):
            df = factory()
            df.to_csv(path, index=False)
            created.append(name)
    return created


# --------------------------------------------------------------------------
# Конфигурации прогонов (фазы сессии)
# --------------------------------------------------------------------------

def phase_real_data(cycle):
    """Все комбинации: датасет × доля теста × k-fold."""
    configs = []
    for dataset in ("study_scores.csv", "student_pass.csv"):
        for test_size in (0.10, 0.15, 0.20, 0.25, 0.30):
            for cv in (3, 5, 7, 10):
                configs.append({
                    "kind": "train",
                    "phase": "A. Реальные данные: разбиения × CV",
                    "dataset": dataset,
                    "cv_folds": cv,
                    "test_size": test_size,
                    "random_state": 42 + cycle,
                })
    random.Random(cycle).shuffle(configs)
    return configs


def phase_synthetic(cycle):
    """Синтетические датасеты разного масштаба и задачи."""
    configs = []
    rng = random.Random(1000 + cycle)
    sizes = (2000, 5000, 10000, 20000)
    feature_counts = (3, 6, 10)
    for i, n in enumerate(sizes):
        for j, k in enumerate(feature_counts):
            seed = 100 + cycle * 37 + i * 11 + j
            # регрессия
            df_r = make_regression_df(n, k, noise=0.5 + 0.5 * ((i + j) % 3),
                                      seed=seed, nonlinear=(i + j) % 2 == 0,
                                      categorical=(k % 2 == 0))
            configs.append({
                "kind": "train", "phase": "B. Синтетика: масштаб данных",
                "dataset": f"synthetic_regression_{n}x{k}_seed{seed}",
                "df": df_r, "target": "target", "task": "regression",
                "cv_folds": 5, "test_size": 0.2, "random_state": 42,
            })
            # классификация
            df_c = make_classification_df(n, k, seed=seed + 5,
                                          nonlinear=(i + j) % 2 == 1,
                                          imbalance=0.2 + 0.2 * (i % 3),
                                          classes=2 if k != 10 else 3)
            configs.append({
                "kind": "train", "phase": "B. Синтетика: масштаб данных",
                "dataset": f"synthetic_clf_{n}x{k}_seed{seed}",
                "df": df_c,
                "target": "segment" if (k == 10) else "label",
                "task": "classification",
                "cv_folds": 5, "test_size": 0.2, "random_state": 42,
            })
    rng.shuffle(configs)
    return configs


def phase_hyperparams(cycle):
    """Перебор гиперпараметров RandomForest на усечённых конвейерах."""
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import (RandomForestRegressor,
                                  RandomForestClassifier)

    configs = []
    for base in ("synth_regression_2000.csv", "synth_fraud_1500.csv",
                 "synth_nonlinear_3000.csv"):
        try:
            df = load_dataframe(base)
        except Exception:
            continue
        target = suggest_target(df)
        task = detect_task_type(df, target)
        for n_est in (100, 300, 600):
            for depth in (4, 8, 16, None):
                cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
                model = Pipeline(steps=[("model", cls(
                    n_estimators=n_est, max_depth=depth,
                    random_state=42 + cycle, n_jobs=-1))])
                configs.append({
                    "kind": "train",
                    "phase": "C. Перебор гиперпараметров (Random Forest)",
                    "dataset": base,
                    "models_override": {"Random Forest "
                                        f"(n={n_est}, depth={depth})": model},
                    "cv_folds": 5, "test_size": 0.2, "random_state": 42,
                })
    random.Random(500 + cycle).shuffle(configs)
    return configs


def phase_seed_stability(cycle):
    """Устойчивость результатов к разбиению (random_state)."""
    configs = []
    for seed in range(1 + cycle * 5, 6 + cycle * 5):
        for dataset in ("study_scores.csv", "student_pass.csv",
                        "synth_regression_2000.csv", "synth_fraud_1500.csv"):
            configs.append({
                "kind": "train",
                "phase": "D. Устойчивость к random_state",
                "dataset": dataset,
                "cv_folds": 5, "test_size": 0.2, "random_state": seed,
            })
    random.Random(900 + cycle).shuffle(configs)
    return configs


def phase_learning_curves(cycle):
    """Кривые обучения: качество в зависимости от объёма данных."""
    return [{
        "kind": "learning_curve",
        "phase": "E. Кривые обучения",
        "dataset": ds,
        "fractions": tuple(np.round(np.arange(0.1, 1.01, 0.1), 2)),
        "cv_folds": 5,
    } for ds in ("study_scores.csv", "synth_regression_2000.csv",
                 "synth_fraud_1500.csv", "student_pass.csv")]


def phase_batch(cycle):
    """Один «проход» по всем фазам."""
    batch = []
    batch += phase_real_data(cycle)
    batch += phase_synthetic(cycle)
    batch += phase_hyperparams(cycle)
    batch += phase_seed_stability(cycle)
    batch += phase_learning_curves(cycle)
    return batch


# --------------------------------------------------------------------------
# Кривая обучения
# --------------------------------------------------------------------------

def learning_curve_run(dataset, fractions, cv_folds=5):
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    df = load_dataframe(dataset)
    target = suggest_target(df)
    task = detect_task_type(df, target)

    with _silence():
        data = preprocess_data(df, target, task, test_size=0.2, random_state=42)

    scoring = "r2" if task == "regression" else (
        "f1" if len(np.unique(data["y_train"])) == 2 else "accuracy")

    curve = []
    for name, model in get_models(task).items():
        sizes, scores = [], []
        for f in fractions:
            n = max(4, int(len(data["X_train"]) * f))
            Xs, ys = data["X_train"].iloc[:n], data["y_train"].iloc[:n]
            k = max(2, min(cv_folds, n // 4))
            pipe = Pipeline(steps=[("preprocessor", data["preprocessor"]),
                                   ("model", model)])
            try:
                with _silence():
                    s = cross_val_score(pipe, Xs, ys, cv=k, scoring=scoring)
                sizes.append(int(n))
                scores.append(round(float(np.nanmean(s)), 4))
            except Exception:
                continue
        if scores:
            curve.append({"model": name, "sizes": sizes, "scores": scores})
    return {"dataset": dataset, "target": target, "task": task,
            "scoring": scoring, "curve": curve}


class _silence:
    def __enter__(self):
        import io
        from contextlib import redirect_stdout
        self._buf = io.StringIO()
        self._ctx = redirect_stdout(self._buf)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False


# --------------------------------------------------------------------------
# Главный цикл
# --------------------------------------------------------------------------

def run_one(cfg, session):
    """Выполняет один прогон и обновляет статус сессии."""
    t0 = time.time()

    if cfg["kind"] == "learning_curve":
        lc = learning_curve_run(cfg["dataset"], cfg["fractions"],
                                cfg.get("cv_folds", 5))
        session.setdefault("learning_curves", []).append(lc)
        elapsed = time.time() - t0
        session["runs"].append({
            "phase": cfg["phase"], "dataset": cfg["dataset"],
            "kind": "learning_curve", "duration_sec": round(elapsed, 2),
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        return lc

    final = None
    for event in train_stream(
        dataset=cfg["dataset"],
        target=cfg.get("target"),
        task_type=cfg.get("task"),
        cv_folds=cfg.get("cv_folds", 5),
        test_size=cfg.get("test_size", 0.2),
        source="extended_session",
        make_charts=False,
        save_model_to_disk=False,
        models_override=cfg.get("models_override"),
        random_state=cfg.get("random_state", 42),
        df=cfg.get("df"),
    ):
        if event["step"] == "error":
            raise RuntimeError(event.get("msg", "run failed"))
        if event["step"] == "done":
            final = event["result"]

    if final:
        session["runs"].append({
            "phase": cfg["phase"], "dataset": final["dataset"],
            "kind": "train", "task": final["task"],
            "best_model": final["best_model"],
            "best_score": final["best_score"],
            "cv_folds": final["cv_folds"],
            "duration_sec": final["duration_sec"],
            "id": None,  # заполнится ниже
            "ts": final["ts"],
        })
        # id из истории
        try:
            from src.pipeline import load_history
            hist = load_history()
            last_id = hist["runs"][-1]["id"] if hist["runs"] else None
            session["runs"][-1]["id"] = last_id
        except Exception:
            pass
    return final


def refresh_summary_charts(session):
    """Перерисовывает сводные графики 05/06/07 по актуальной истории."""
    from src.pipeline import load_history, load_session
    from src import dashboard_viz

    hist = load_history()
    trend = [{"score": r["best_score"], "ts": r.get("ts"),
              "dataset": r.get("dataset"), "task": r.get("task"),
              "model": r.get("best_model"), "id": r.get("id")}
             for r in hist["runs"] if r.get("best_score") is not None]
    winners = {}
    for r in hist["runs"]:
        bm = r.get("best_model")
        if bm:
            winners[bm] = winners.get(bm, 0) + 1

    made = []
    try:
        p = dashboard_viz.plot_history(trend)
        if p:
            made.append(os.path.basename(p))
        p = dashboard_viz.plot_model_wins(winners)
        if p:
            made.append(os.path.basename(p))
        lcs = session.get("learning_curves") or []
        if lcs:
            last = lcs[-1]
            p = dashboard_viz.plot_learning_curve(last["curve"], last["scoring"])
            if p:
                made.append(os.path.basename(p))
    except Exception as exc:
        print(f"  [chart error] {exc}")
    return made


def main():
    print("=" * 64)
    print("  РАСШИРЕННАЯ ТРЕНИРОВОЧНАЯ СЕССИЯ — минимум 40 минут")
    print("=" * 64)

    created = ensure_example_datasets()
    if created:
        print(f"  Созданы датасеты-примеры: {', '.join(created)}")

    t0 = time.time()
    session = {
        "status": "running",
        "title": "Расширенная тренировочная сессия",
        "min_minutes": MIN_MINUTES,
        "target_sec": MIN_SECONDS,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "elapsed_sec": 0,
        "runs_completed": 0,
        "runs_total_target": None,
        "runs": [],
        "learning_curves": [],
        "cycles_completed": 0,
        "phases": [
            "A. Реальные данные: разбиения × CV",
            "B. Синтетика: масштаб данных",
            "C. Перебор гиперпараметров (Random Forest)",
            "D. Устойчивость к random_state",
            "E. Кривые обучения",
        ],
        "best_score": None,
        "best_model": None,
        "charts": [],
        "notes": [],
    }
    write_session(session)

    cycle = 0
    interrupted = False
    fatal = None
    try:
        while True:
            elapsed = time.time() - t0
            if elapsed >= MIN_SECONDS:
                print(f"\n[OK] Достигнут целевой объём: {elapsed / 60:.1f} мин")
                break

            batch = phase_batch(cycle)
            print(f"\n--- Цикл {cycle + 1}: {len(batch)} конфигураций ---")

            for cfg in batch:
                elapsed = time.time() - t0
                if elapsed >= MIN_SECONDS:
                    break
                try:
                    res = run_one(cfg, session)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    session["notes"].append(f"{cfg.get('phase')}: {exc}")
                    res = None

                session["runs_completed"] += 1
                session["elapsed_sec"] = round(time.time() - t0, 1)

                if res and res.get("best_score") is not None:
                    if (session["best_score"] is None
                            or res["best_score"] > session["best_score"]):
                        session["best_score"] = res["best_score"]
                        session["best_model"] = res["best_model"]

                if session["runs_completed"] % 5 == 0:
                    write_session(session)
                    done = session["runs_completed"]
                    phase = str(cfg.get("phase", ""))[:40]
                    print(f"  [{done}] {elapsed / 60:.1f} мин | {phase} | лучший={session['best_score']}")

                if session["runs_completed"] % CHART_EVERY_RUNS == 0:
                    session["charts"] = refresh_summary_charts(session)
                    write_session(session)

            cycle += 1
            session["cycles_completed"] = cycle
            write_session(session)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[!] Сессия прервана пользователем (Ctrl+C)")
    except Exception as exc:          # сессия не должна «повисать» в статусе running
        fatal = exc
        session["notes"].append(f"FATAL: {exc}")
        print(f"\n[!] Сессия остановлена из-за ошибки: {exc}")

    # Финализация --------------------------------------------------------
    session["elapsed_sec"] = round(time.time() - t0, 1)
    session["status"] = ("interrupted" if interrupted else
                         ("error" if fatal else "finished"))
    session["finished_at"] = datetime.now().isoformat(timespec="seconds")
    session["charts"] = refresh_summary_charts(session)
    session["notes"].append(
        f"Сессия выполнена: {session['runs_completed']} прогонов, "
        f"{session['elapsed_sec'] / 60:.1f} минут, "
        f"{session['cycles_completed']} полных циклов.")
    write_session(session)

    print("\n" + "=" * 64)
    print(f"  СТАТУС: {session['status']}")
    print(f"  Время:  {session['elapsed_sec'] / 60:.1f} мин "
          f"(цель — не менее {MIN_MINUTES})")
    print(f"  Прогонов: {session['runs_completed']}")
    print(f"  Лучший показатель: {session['best_score']} "
          f"({session['best_model']})")
    print("=" * 64)


if __name__ == "__main__":
    main()
