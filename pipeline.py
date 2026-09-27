"""
AutoML Pipeline (non-interactive core)
=======================================

Модуль предоставляет потоковое (streaming) обучение и аналитику для
веб-дашборда и расширенной тренировочной сессии.

Никаких input() и интерактивных prompt — только данные на входе,
структурированные события на выходе.

Поток обучения (train_stream) генерирует события:
    {"step": "load" | "analyze" | "preprocess" | "model" | "best" |
              "test" | "charts" | "save" | "done" | "error", ...}
Финальное событие "done" содержит сериализованный результат прогона,
который целиком уходит в историю тренировок (results/training_history.json).
"""

import io
import json
import math
import os
import time
from contextlib import contextmanager, redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd

from .data_loader import get_dataset_info
from .preprocessing import preprocess_data, detect_task_type
from .models import get_models
from .training import evaluate_on_test
from .evaluation import save_model

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")

HISTORY_PATH = os.path.join(RESULTS_DIR, "training_history.json")
SESSION_PATH = os.path.join(RESULTS_DIR, "extended_session.json")
MODEL_META_PATH = os.path.join(MODELS_DIR, "best_model_meta.json")

LOCK_PATH = os.path.join(RESULTS_DIR, ".history.lock")

MAX_UPLOAD_MB = 25


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def sanitize(obj):
    """Конвертирует numpy/pandas объекты в JSON-совместимые."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


@contextmanager
def silenced():
    """Глушит print() из legacy-модулей (data_loader / training / evaluation)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        yield buf


def _lock():
    """Простой межпроцессный lock (одна запись в историю за раз)."""
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.05)
    return False


def _unlock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


@contextmanager
def file_lock():
    ok = _lock()
    try:
        yield ok
    finally:
        if ok:
            _unlock()


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def _safe_dataset_path(name):
    """Защищает от path traversal: файл должен лежать внутри data/."""
    name = os.path.basename(str(name))
    path = os.path.join(DATA_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Датасет не найден: {name}")
    return path


def load_dataframe(name):
    """Загружает CSV из папки data/ в DataFrame."""
    path = _safe_dataset_path(name)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Датасет пуст: {name}")
    return df


def list_datasets():
    """Список доступных датасетов с кратким описанием."""
    out = []
    if not os.path.isdir(DATA_DIR):
        return out
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.lower().endswith(".csv"):
            continue
        try:
            df = load_dataframe(fname)
        except Exception:
            continue
        target = suggest_target(df)
        out.append({
            "name": fname,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "size_kb": round(os.path.getsize(os.path.join(DATA_DIR, fname)) / 1024, 1),
            "target_hint": target,
            "task_hint": detect_task_type(df, target) if target else None,
            "modified": datetime.fromtimestamp(
                os.path.getmtime(os.path.join(DATA_DIR, fname))
            ).isoformat(timespec="seconds"),
        })
    return out


def suggest_target(df):
    """Эвристика: целевая переменная — последний «информативный» числовой столбец."""
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric:
        obj = list(df.select_dtypes(exclude=[np.number]).columns)
        return obj[-1] if obj else None

    # 1) точные совпадения по приоритету (имя столбца цели)
    preferred_exact = ("target", "label", "y", "outcome", "class", "pass",
                       "churn", "survived", "result", "price", "score")
    lower = {c: c.lower() for c in df.columns}
    for p in preferred_exact:
        for col, low in lower.items():
            if low == p:
                return col
    # 2) столбцы вида <что-то>_target / <что-то>_label / <что-то>_pass
    for suffix in ("target", "label", "class", "pass", "outcome"):
        for col, low in lower.items():
            if low.endswith("_" + suffix):
                return col
    # последний числовой столбец, где больше одного уникального значения
    for col in reversed(numeric):
        if df[col].nunique() > 1:
            return col
    return numeric[-1]


def describe_dataset(name):
    """Полное описание датасета для панели анализа."""
    df = load_dataframe(name)
    info = get_dataset_info(df)

    columns = []
    for col in df.columns:
        s = df[col]
        col_info = {
            "name": col,
            "dtype": str(s.dtype),
            "role": "categorical" if s.dtype == object else "numeric",
            "missing": int(s.isnull().sum()),
            "missing_pct": round(float(s.isnull().mean()) * 100, 2),
            "unique": int(s.nunique()),
        }
        if col_info["role"] == "numeric":
            desc = s.describe()
            col_info.update({
                "min": sanitize(s.min()),
                "max": sanitize(s.max()),
                "mean": sanitize(s.mean()),
                "std": sanitize(s.std()),
                "median": sanitize(s.median()),
            })
            _ = desc
        else:
            vc = s.value_counts().head(5)
            col_info["top"] = [{k: str(v) for k, v in vc.items()}]
        columns.append(col_info)

    target = suggest_target(df)
    task = detect_task_type(df, target) if target else None

    return {
        "name": name,
        "rows": info["rows"],
        "columns_count": info["columns"],
        "missing_total": int(sum(info["missing"].values())),
        "numeric_cols": info["numeric_cols"],
        "categorical_cols": info["categorical_cols"],
        "columns": columns,
        "suggested_target": target,
        "detected_task": task,
        "head": sanitize(df.head(8).to_dict(orient="list")),
    }


def save_uploaded_dataset(filename, content_bytes):
    """Сохраняет загруженный CSV в data/ и возвращает имя файла."""
    name = os.path.basename(str(filename)).strip()
    if not name.lower().endswith(".csv"):
        raise ValueError("Поддерживаются только .csv файлы")
    if len(content_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Файл больше {MAX_UPLOAD_MB} МБ")
    if not name:
        raise ValueError("Пустое имя файла")

    text = content_bytes.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise ValueError("Нужно минимум 2 столбца (признаки + цель)")
    if len(df) < 10:
        raise ValueError("Нужно минимум 10 строк")

    # безопасное имя: только буквы/цифры/точка/тире/подчёркивание
    safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)
    path = os.path.join(DATA_DIR, safe)
    df.to_csv(path, index=False)
    return safe, int(len(df)), int(df.shape[1])


# --------------------------------------------------------------------------
# Training (streaming)
# --------------------------------------------------------------------------

def _scoring_for(task_type, y):
    if task_type == "regression":
        return "r2"
    return "f1" if len(np.unique(y)) == 2 else "accuracy"


def train_stream(dataset, target=None, task_type=None, cv_folds=5,
                 test_size=0.2, source="dashboard", make_charts=True,
                 save_model_to_disk=True, models_override=None, random_state=42,
                 df=None, progress_base=0, progress_span=100):
    """
    Генератор событий обучения. Каждое событие — dict с полями step/msg/progress.
    Финальное событие {"step": "done", "result": {...}} содержит полный отчёт.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    from . import dashboard_viz

    t_start = time.time()
    progress = progress_base

    def ev(step, msg, **kw):
        return {"step": step, "msg": msg, "progress": min(100, int(progress)), **kw}

    # 1. Загрузка --------------------------------------------------------
    if df is None:
        yield ev("load", f"Загрузка датасета «{dataset}»…")
        try:
            df = load_dataframe(dataset)
        except Exception as exc:
            yield {"step": "error", "msg": f"Ошибка загрузки: {exc}", "progress": 0}
            return
    else:
        yield ev("load", f"Датасет «{dataset}» загружен в память")

    rows, cols = df.shape
    yield ev("analyze", f"Принято: {rows} строк × {cols} столбцов", rows=rows, columns=cols)
    progress = progress_base + 5 * progress_span // 100

    # 2. Определение цели и типа задачи ----------------------------------
    target = target or suggest_target(df)
    if target not in df.columns:
        yield {"step": "error", "msg": f"Столбец «{target}» не найден", "progress": progress}
        return
    task_type = task_type or detect_task_type(df, target)
    yield ev("preprocess",
             f"Цель: «{target}» · тип задачи: {task_type} · "
             f"кросс-валидация {cv_folds}-fold · разбиение "
             f"{int((1 - test_size) * 100)}/{int(test_size * 100)}",
             target=target, task=task_type)
    progress = progress_base + 12 * progress_span // 100

    # 3. Предобработка ----------------------------------------------------
    with silenced():
        data = preprocess_data(df, target, task_type, test_size=test_size,
                               random_state=random_state)
    if data is None:
        yield {"step": "error", "msg": "Предобработка не удалась (слишком мало данных)",
               "progress": progress}
        return

    yield ev("preprocess",
             f"Пайплайн готов: импутация медианой → one-hot кодирование → "
             f"StandardScaler · train {len(data['X_train'])} / test {len(data['X_test'])}",
             train_size=int(len(data["X_train"])), test_size=int(len(data["X_test"])))
    progress = progress_base + 18 * progress_span // 100

    # 4. Обучение моделей с кросс-валидацией -------------------------------
    models = models_override if models_override is not None else get_models(task_type)
    scoring = _scoring_for(task_type, data["y_train"])

    results = {}
    model_events = []
    n_models = max(1, len(models))

    for idx, (name, model_pipe) in enumerate(models.items()):
        m_t0 = time.time()
        full = Pipeline(steps=[("preprocessor", data["preprocessor"]),
                               ("model", model_pipe)])
        try:
            cv_scores = cross_val_score(full, data["X_train"], data["y_train"],
                                        cv=cv_folds, scoring=scoring)
        except Exception as exc:
            yield {"step": "error", "msg": f"Модель «{name}» не обучилась: {exc}",
                   "progress": progress}
            return
        full.fit(data["X_train"], data["y_train"])
        m_dur = time.time() - m_t0

        results[name] = {
            "pipeline": full,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_scores": [float(s) for s in cv_scores],
            "scoring": scoring,
            "fit_seconds": round(m_dur, 3),
        }
        model_events.append(sanitize({
            "name": name, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
            "cv_scores": cv_scores, "fit_seconds": round(m_dur, 3),
        }))

        progress = progress_base + int((18 + 55 * (idx + 1) / n_models)) * progress_span // 100
        yield ev("model",
                 f"[{idx + 1}/{n_models}] {name}: {scoring.upper()} = "
                 f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f} ({m_dur:.2f} с)",
                 model=sanitize({"name": name, "cv_mean": cv_scores.mean(),
                                 "cv_std": cv_scores.std(), "cv_scores": cv_scores,
                                 "fit_seconds": round(m_dur, 3)}))

    # 5. Выбор лучшей модели ---------------------------------------------
    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best = results[best_name]
    yield ev("best",
             f"Лучшая модель: {best_name} ({scoring.upper()} = {best['cv_mean']:.4f} ± {best['cv_std']:.4f})",
             best=best_name, best_score=best["cv_mean"])
    progress = progress_base + 78 * progress_span // 100

    # 6. Финальная оценка на тесте ----------------------------------------
    with silenced():
        metrics = evaluate_on_test(best["pipeline"], data["X_test"],
                                   data["y_test"], task_type)
    y_pred = np.asarray(metrics.pop("y_pred"))
    y_test = np.asarray(metrics.pop("y_test"))
    metrics = sanitize(metrics)

    yield ev("test", f"Тестовая оценка ({dataset}): " +
             ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                       if isinstance(v, (int, float))),
             metrics=metrics)
    progress = progress_base + 86 * progress_span // 100

    # 7. Графики ------------------------------------------------------------
    charts = []
    if make_charts:
        try:
            feature_cols = list(data["X_train"].columns)
            charts = dashboard_viz.charts_for_run(
                data, results, best_name, metrics, y_test, y_pred,
                feature_cols[0] if feature_cols else None, target, task_type)
            yield ev("charts", "Графики сохранены: " + ", ".join(os.path.basename(c) for c in charts),
                     charts=[os.path.basename(c) for c in charts])
        except Exception as exc:
            yield ev("charts", f"Графики не построены: {exc}")
    progress = progress_base + 92 * progress_span // 100

    # 8. Сохранение модели ---------------------------------------------------
    duration = round(time.time() - t_start, 3)
    if save_model_to_disk:
        try:
            save_model(best["pipeline"])
            meta = {
                "model": best_name,
                "dataset": dataset,
                "target": target,
                "task": task_type,
                "features": list(data["X_train"].columns),
                "feature_dtypes": {c: str(t) for c, t in data["X_train"].dtypes.items()},
                "scoring": scoring,
                "cv_folds": cv_folds,
                "test_size": test_size,
                "metrics": metrics,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            }
            with open(MODEL_META_PATH, "w", encoding="utf-8") as fh:
                json.dump(sanitize(meta), fh, ensure_ascii=False, indent=2)
            yield ev("save", f"Модель сохранена: models/best_model.pkl ({best_name})")
        except Exception as exc:
            yield ev("save", f"Сохранение модели не удалось: {exc}")

    # 9. Итоговый отчёт + история -------------------------------------------
    ranked = sorted(
        [{"name": n, "cv_mean": r["cv_mean"], "cv_std": r["cv_std"],
          "cv_scores": r["cv_scores"], "scoring": r["scoring"],
          "fit_seconds": r["fit_seconds"]} for n, r in results.items()],
        key=lambda x: x["cv_mean"], reverse=True)

    result = sanitize({
        "dataset": dataset,
        "rows": rows,
        "columns": cols,
        "target": target,
        "task": task_type,
        "cv_folds": cv_folds,
        "test_size": test_size,
        "scoring": scoring,
        "duration_sec": duration,
        "models": ranked,
        "best_model": best_name,
        "best_score": best["cv_mean"],
        "test_metrics": metrics,
        "charts": [os.path.basename(c) for c in charts],
        "source": source,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(data["X_train"])),
        "n_test": int(len(data["X_test"])),
    })

    entry_id = append_history(result)
    result["id"] = entry_id

    progress = progress_base + progress_span
    yield {"step": "done", "msg": f"Прогон #{entry_id} завершён за {duration} с",
           "progress": min(100, int(progress)), "result": result}


def run_training(**kwargs):
    """Синхронный запуск обучения. Возвращает финальный result (или None)."""
    final = None
    for event in train_stream(**kwargs):
        if event.get("step") == "done":
            final = event["result"]
        elif event.get("step") == "error":
            raise RuntimeError(event.get("msg", "Training failed"))
    return final


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def run_experiments_stream(dataset, target=None):
    """Классические эксперименты (доли разбиения, сравнение моделей, дообучение)."""
    from .evaluation import run_experiments as _legacy_experiments

    yield {"step": "load", "msg": f"Эксперименты на «{dataset}»…"}
    df = load_dataframe(dataset)
    target = target or suggest_target(df)
    task = detect_task_type(df, target)

    yield {"step": "run", "msg": "4 серии экспериментов: доли разбиения / сравнение моделей / дообучение / валидация прогноза"}
    with silenced():
        exp = _legacy_experiments(df, target, task)

    yield {"step": "done", "msg": "Эксперименты завершены", "result": sanitize(exp)}


# --------------------------------------------------------------------------
# Custom linear regression demo (собственная реализация)
# --------------------------------------------------------------------------

def custom_linear_demo(dataset, target=None):
    """Сравнение собственной LR (Normal Equation + Gradient Descent) со sklearn."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    from .custom_linear import CustomLinearRegression

    df = load_dataframe(dataset)
    target = target or suggest_target(df)
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    feature_cols = [c for c in numeric if c != target]
    if not feature_cols:
        raise ValueError("Нет числовых признаков")

    # берём первый признак (как в учебном примере с study_hours)
    feature = feature_cols[0]
    X = df[[feature]].values.astype(float)
    y = df[target].values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    variants = {}

    ne = CustomLinearRegression(method="normal_equation")
    ne.fit(X_train, y_train)
    variants["Собственная: Normal Equation"] = (ne.predict(X_test), ne.weights, ne.bias)

    gd = CustomLinearRegression(method="gradient_descent",
                                 learning_rate=0.1, n_iterations=1000)
    gd.fit(X_train, y_train)
    variants["Собственная: Gradient Descent"] = (gd.predict(X_test), gd.weights, gd.bias)

    sk = LinearRegression()
    sk.fit(X_train, y_train)
    variants["Sklearn LinearRegression"] = (sk.predict(X_test), sk.coef_, sk.intercept_)

    rows = []
    for name, (pred, w, b) in variants.items():
        rows.append({
            "method": name,
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
            "weights": [float(x) for x in np.atleast_1d(w)],
            "bias": float(b),
        })

    diff_ne = float(np.max(np.abs(variants["Собственная: Normal Equation"][0] -
                                  variants["Sklearn LinearRegression"][0])))
    diff_gd = float(np.max(np.abs(variants["Собственная: Gradient Descent"][0] -
                                  variants["Sklearn LinearRegression"][0])))

    return sanitize({
        "dataset": dataset,
        "target": target,
        "feature": feature,
        "rows": rows,
        "max_diff_normal_equation": diff_ne,
        "max_diff_gradient_descent": diff_gd,
        "formula": f"{target} = {float(np.atleast_1d(sk.coef_)[0]):.4f}·{feature} + {float(sk.intercept_):.4f}",
    })


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def predict(values):
    """
    Прогноз сохранённой моделью.
    values — dict {признак: число}.
    """
    import joblib

    if not os.path.isfile(MODEL_META_PATH):
        raise RuntimeError("Модель ещё не обучена — запустите тренировку")
    with open(MODEL_META_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)

    features = meta["features"]
    missing = [f for f in features if f not in values]
    if missing:
        raise ValueError("Не указаны признаки: " + ", ".join(missing))

    dtypes = meta.get("feature_dtypes") or {}
    row = []
    for f in features:
        v = values[f]
        dtype = str(dtypes.get(f, "float64"))
        if dtype.startswith(("float", "int", "uint")):
            row.append(float(v))          # числовой признак
        else:
            row.append(str(v))            # категориальный признак (one-hot в пайплайне)
    X = pd.DataFrame([row], columns=features)
    pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    raw = pipeline.predict(X)[0]
    pred = sanitize(raw)

    out = {
        "prediction": pred,
        "model": meta["model"],
        "target": meta["target"],
        "task": meta["task"],
        "dataset": meta["dataset"],
        "inputs": {f: (float(values[f])
                         if str(dtypes.get(f, "float64")).startswith(
                             ("float", "int", "uint"))
                         else str(values[f])) for f in features},
    }
    if meta["task"] == "classification":
        classes = getattr(pipeline.named_steps["model"], "classes_", None)
        if classes is not None:
            idx = int(np.argmax(raw)) if np.ndim(raw) else int(list(classes).index(raw))
            out["predicted_class"] = sanitize(classes[idx])
            try:
                proba = pipeline.predict_proba(X)[0]
                out["proba"] = sanitize({str(c): float(p) for c, p in zip(classes, proba)})
            except Exception:
                pass
    return out


# --------------------------------------------------------------------------
# History & statistics
# --------------------------------------------------------------------------

def load_history():
    """Читает историю тренировок. Формат: {"runs": [...], "next_id": N}"""
    if not os.path.isfile(HISTORY_PATH):
        return {"runs": [], "next_id": 1}
    try:
        with open(HISTORY_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "runs" not in data:
            raise ValueError
        return data
    except Exception:
        return {"runs": [], "next_id": 1}


def append_history(entry):
    """Атомарно дописывает прогон в историю. Возвращает id записи."""
    with file_lock():
        hist = load_history()
        run_id = hist.get("next_id", len(hist["runs"]) + 1)
        entry = dict(entry)
        entry["id"] = run_id
        hist["runs"].append(sanitize(entry))
        hist["next_id"] = run_id + 1
        tmp = HISTORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(hist, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, HISTORY_PATH)
    return run_id


def load_session():
    """Статус расширенной (40-минутной) тренировочной сессии."""
    if not os.path.isfile(SESSION_PATH):
        return None
    try:
        with open(SESSION_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def aggregate_stats():
    """Сводная статистика по всем тренировкам для раздела «Статистика»."""
    hist = load_history()
    runs = hist["runs"]

    total_time = sum(r.get("duration_sec", 0) or 0 for r in runs)
    by_task = {}
    winners = {}
    score_by_model = {}
    fit_by_model = {}

    for r in runs:
        task = r.get("task", "unknown")
        by_task.setdefault(task, {"runs": 0, "best": None})
        by_task[task]["runs"] += 1
        s = r.get("best_score")
        if s is not None and (by_task[task]["best"] is None or s > by_task[task]["best"]):
            by_task[task]["best"] = s

        bm = r.get("best_model")
        if bm:
            winners[bm] = winners.get(bm, 0) + 1

        for m in r.get("models", []):
            name = m.get("name")
            score_by_model.setdefault(name, []).append(m.get("cv_mean"))
            if m.get("fit_seconds") is not None:
                fit_by_model.setdefault(name, []).append(m.get("fit_seconds"))

    avg_by_model = {k: round(float(np.mean(v)), 4) for k, v in score_by_model.items()}
    avg_fit = {k: round(float(np.mean(v)), 3) for k, v in fit_by_model.items()}

    trend = [{"id": r.get("id"), "ts": r.get("ts"), "score": r.get("best_score"),
              "dataset": r.get("dataset"), "task": r.get("task"),
              "model": r.get("best_model")} for r in runs if r.get("best_score") is not None]

    # последние прогоны — компактная версия
    recent = []
    for r in reversed(runs[-40:]):
        recent.append({
            "id": r.get("id"), "ts": r.get("ts"), "dataset": r.get("dataset"),
            "task": r.get("task"), "target": r.get("target"),
            "best_model": r.get("best_model"), "best_score": r.get("best_score"),
            "scoring": r.get("scoring"), "cv_folds": r.get("cv_folds"),
            "duration_sec": r.get("duration_sec"), "rows": r.get("rows"),
            "source": r.get("source"),
        })

    datasets = sorted({r.get("dataset") for r in runs if r.get("dataset")})
    try:
        data_files = len([f for f in os.listdir(DATA_DIR)
                          if f.lower().endswith(".csv")])
    except OSError:
        data_files = None

    best_overall = None
    if runs:
        br = max((r for r in runs if r.get("best_score") is not None),
                 key=lambda r: r["best_score"], default=None)
        if br:
            best_overall = {
                "score": br["best_score"], "model": br["best_model"],
                "dataset": br["dataset"], "task": br["task"],
                "ts": br.get("ts"), "id": br.get("id"),
            }

    session = load_session()

    return sanitize({
        "total_runs": len(runs),
        "total_train_time_sec": round(total_time, 2),
        "models_trained": int(sum(len(r.get("models", [])) for r in runs)),
        "cv_folds_total": int(sum(r.get("cv_folds", 0) or 0 for r in runs)),
        "datasets": datasets,
        "data_files": data_files,
        "by_task": by_task,
        "winners": winners,
        "avg_score_by_model": avg_by_model,
        "avg_fit_seconds_by_model": avg_fit,
        "trend": trend,
        "recent": recent,
        "best_overall": best_overall,
        "session": session,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    })
