"""
Flask Server — Intelligent Data Analysis System (AutoML Dashboard)
==================================================================

Единственный сайт проекта: http://localhost:5000
Запуск:  start.bat  →  python start.py  →  python server.py

API:
  GET  /                       — дашборд (index.html)
  GET  /results/<file>         — графики
  GET  /data/<file>            — датасеты (CSV на скачивание)
  GET  /docs/<file>            — документы проекта (MD)
  GET  /api/datasets           — список датасетов
  GET  /api/describe?dataset=  — описание датасета
  POST /api/upload             — загрузка своего CSV
  POST /api/train              — обучение (SSE-поток событий)
  POST /api/experiments        — серия экспериментов
  GET  /api/stats              — сводная статистика всех тренировок
  GET  /api/history?limit=     — история тренировок
  GET  /api/session            — статус расширенной сессии (40+ мин)
  GET  /api/predict-meta       — метаданные сохранённой модели
  POST /api/predict            — прогноз сохранённой моделью
  GET  /api/custom-linear      — сравнение собственной LR со sklearn
  GET  /api/system             — версии, окружение, источники кода
"""

import io
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime

# Windows-консоль cp1251 не умеет кириллицу/«±» — принудительно UTF-8 в логах.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, request, jsonify, send_from_directory, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src import pipeline

# --------------------------------------------------------------------------
# Проектная мета-информация
# --------------------------------------------------------------------------

# Ссылка на репозиторий проекта (задаётся переменной окружения REPO_URL).
# Пока не задана — на сайте показываются репозитории-источники (см. REFERENCES).
REPO_URL = os.environ.get("REPO_URL", "").strip()

REFERENCES = [
    {"name": "jakevdp/PythonDataScienceHandbook",
     "url": "https://github.com/jakevdp/PythonDataScienceHandbook",
     "note": "учебник Data Science: структура анализа, pandas/numpy/sklearn"},
    {"name": "TrainingByPackt/Data-Science-Projects-with-Python",
     "url": "https://github.com/TrainingByPackt/Data-Science-Projects-with-Python",
     "note": "end-to-end ML-проекты: пайплайн загрузка → обучение → оценка"},
    {"name": "scikit-learn/scikit-learn",
     "url": "https://github.com/scikit-learn/scikit-learn",
     "note": "Pipeline, ColumnTransformer, кросс-валидация, метрики"},
    {"name": "UrbsLab/STREAMLINE",
     "url": "https://github.com/UrbsLab/STREAMLINE",
     "note": "AutoML-pipeline: автоматическое сравнение и выбор моделей"},
    {"name": "pycaret/pycaret",
     "url": "https://github.com/pycaret/pycaret",
     "note": "AutoML с единым простым API для разных типов задач"},
]

PROJECT = {
    "title": "Интеллектуальная система анализа данных",
    "subtitle": "AutoML-пайплайн на основе математических моделей",
    "report_id": "AIDA-2026-001",
    "version": "2.0",
    "started_at": datetime.now().isoformat(timespec="seconds"),
    "repo_url": REPO_URL or None,
    "references": REFERENCES,
}

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

# Не сортируем ключи JSON — порядок столбцов датасета должен сохраняться
app.json.sort_keys = False

SERVER_STARTED = time.time()
TRAINING_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Страницы
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory("results", filename)


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory("data", filename, mimetype="text/csv")


@app.route("/docs/<path:filename>")
def serve_docs(filename):
    if not filename.lower().endswith(".md"):
        return jsonify({"error": "Только .md документы"}), 404
    path = os.path.join(".", filename)
    if not os.path.isfile(path):
        return jsonify({"error": "Документ не найден"}), 404
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return Response(text, mimetype="text/plain; charset=utf-8")


# --------------------------------------------------------------------------
# Датасеты
# --------------------------------------------------------------------------

@app.route("/api/datasets")
def api_datasets():
    try:
        return jsonify({"datasets": pipeline.list_datasets()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/describe")
def api_describe():
    name = request.args.get("dataset", "")
    try:
        return jsonify(pipeline.describe_dataset(name))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/upload", methods=["POST"])
def api_upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "Файл не передан"}), 400
    try:
        name, rows, cols = pipeline.save_uploaded_dataset(
            file.filename, file.read())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"dataset": name, "rows": rows, "columns": cols,
                    "message": f"Датасет {name} сохранён ({rows}×{cols})"})


# --------------------------------------------------------------------------
# Обучение (SSE)
# --------------------------------------------------------------------------

@app.route("/api/train", methods=["POST"])
def api_train():
    payload = request.get_json(silent=True) or {}

    if not TRAINING_LOCK.acquire(blocking=False):
        return jsonify({"error": "Обучение уже идёт — дождитесь завершения"}), 409

    def generate():
        try:
            stream = pipeline.train_stream(
                dataset=payload.get("dataset", ""),
                target=payload.get("target"),
                task_type=payload.get("task"),
                cv_folds=int(payload.get("cv_folds", 5)),
                test_size=float(payload.get("test_size", 0.2)),
                source=payload.get("source", "dashboard"),
                make_charts=bool(payload.get("make_charts", True)),
            )
            for event in stream:
                yield f"data: {json.dumps(pipeline.sanitize(event), ensure_ascii=False)}\n\n"
        except Exception as exc:
            traceback.print_exc()
            err = {"step": "error", "msg": f"Ошибка: {exc}", "progress": 0}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        finally:
            TRAINING_LOCK.release()

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


# --------------------------------------------------------------------------
# Эксперименты
# --------------------------------------------------------------------------

@app.route("/api/experiments", methods=["POST"])
def api_experiments():
    payload = request.get_json(silent=True) or {}
    if not TRAINING_LOCK.acquire(blocking=False):
        return jsonify({"error": "Идёт обучение — повторите позже"}), 409
    try:
        final = None
        for event in pipeline.run_experiments_stream(
                payload.get("dataset", ""), payload.get("target")):
            if event["step"] == "error":
                return jsonify({"error": event.get("msg")}), 500
            if event["step"] == "done":
                final = event["result"]
        return jsonify({"result": final})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        TRAINING_LOCK.release()


# --------------------------------------------------------------------------
# Статистика / история / сессия
# --------------------------------------------------------------------------

@app.route("/api/stats")
def api_stats():
    try:
        return jsonify(pipeline.aggregate_stats())
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/history")
def api_history():
    limit = request.args.get("limit", default=100, type=int)
    hist = pipeline.load_history()
    runs = hist["runs"][-limit:]
    return jsonify({"total": len(hist["runs"]), "runs": runs[::-1]})


@app.route("/api/session")
def api_session():
    return jsonify({"session": pipeline.load_session()})


# --------------------------------------------------------------------------
# Прогноз
# --------------------------------------------------------------------------

@app.route("/api/predict-meta")
def api_predict_meta():
    if not os.path.isfile(pipeline.MODEL_META_PATH):
        return jsonify({"model": None})
    with open(pipeline.MODEL_META_PATH, encoding="utf-8") as fh:
        return jsonify(json.load(fh))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    values = payload.get("values") or {}
    try:
        return jsonify(pipeline.predict(values))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# --------------------------------------------------------------------------
# Собственная реализация линейной регрессии
# --------------------------------------------------------------------------

@app.route("/api/custom-linear")
def api_custom_linear():
    dataset = request.args.get("dataset", "study_scores.csv")
    target = request.args.get("target")
    try:
        return jsonify(pipeline.custom_linear_demo(dataset, target))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# --------------------------------------------------------------------------
# Системная информация
# --------------------------------------------------------------------------

@app.route("/api/system")
def api_system():
    import platform
    import numpy
    import pandas
    import sklearn
    import flask

    uptime = time.time() - SERVER_STARTED
    return jsonify({
        **PROJECT,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "flask": getattr(flask, "__version__", "3.x"),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "models_regression": list(__import__("src.models",
                                             fromlist=["x"]).get_regression_models().keys()),
        "models_classification": list(__import__("src.models",
                                                 fromlist=["x"]).get_classification_models().keys()),
        "server_time": datetime.now().isoformat(timespec="seconds"),
        "uptime_sec": round(uptime, 1),
        "docs": [f for f in sorted(os.listdir("."))
                 if f.lower().endswith(".md")],
        "endpoints": sorted({rule.rule for rule in app.url_map.iter_rules()
                             if rule.rule.startswith("/api")}),
    })


# --------------------------------------------------------------------------
# Fallback
# --------------------------------------------------------------------------

@app.route("/api/<path:path>", methods=["GET", "POST"])
def api_fallback(path):
    return jsonify({
        "error": f"Эндпоинт /api/{path} не существует",
        "available": sorted({r.rule for r in app.url_map.iter_rules()
                             if r.rule.startswith("/api")}),
    }), 404


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    port = int(os.environ.get("PORT", 5000))
    print("=" * 64)
    print("  INTELLIGENT DATA ANALYSIS SYSTEM — AutoML Dashboard")
    print("=" * 64)
    print(f"  Дашборд:  http://localhost:{port}")
    print("  Останов:  Ctrl+C")
    print("=" * 64)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
