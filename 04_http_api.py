"""
Пример 4 · Работа через HTTP API дашборда
==========================================
Клиент к запущенному серверу (start.bat → http://localhost:5000).
Используется только стандартная библиотека (urllib), внешних зависимостей нет.

Запуск:  python examples/04_http_api.py
         (сервер должен быть запущен через start.bat)
"""

import json
import os
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.environ.get("DASHBOARD_URL", "http://localhost:5000")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    print(f"Сервер: {BASE}\n")

    # 1. Информация о системе
    sysinfo = get("/api/system")
    print(f"[1] {sysinfo['title']} · v{sysinfo['version']} · "
          f"Python {sysinfo['python']} · sklearn {sysinfo['sklearn']}")

    # 2. Список датасетов
    ds = get("/api/datasets")["datasets"]
    print(f"[2] Датасеты: " + ", ".join(f"{d['name']} ({d['rows']})" for d in ds))

    # 3. Описание датасета
    info = get("/api/describe?dataset=study_scores.csv")
    print(f"[3] study_scores.csv: цель={info['suggested_target']}, "
          f"задача={info['detected_task']}, пропусков={info['missing_total']}")

    # 4. Обучение (POST /api/train возвращает SSE-поток событий)
    print("[4] Тренировка…")
    final = None
    with urllib.request.urlopen(
            urllib.request.Request(
                BASE + "/api/train",
                data=json.dumps({"dataset": "student_pass.csv",
                                 "cv_folds": 5, "test_size": 0.2,
                                 "source": "http_example"}).encode("utf-8"),
                headers={"Content-Type": "application/json"}), timeout=600) as resp:
        buf = b""
        while True:
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                for line in block.split(b"\n"):
                    if line.startswith(b"data: "):
                        ev = json.loads(line[6:].decode("utf-8"))
                        if ev.get("msg"):
                            print("    ", ev["msg"])
                        if ev.get("step") == "done":
                            final = ev.get("result")
    if final:
        print(f"     Лучшая модель: {final['best_model']} "
              f"({final['scoring'].upper()} = {final['best_score']:.4f})")

    # 5. Статистика всех тренировок
    stats = get("/api/stats")
    print(f"[5] Всего прогонов: {stats['total_runs']}, "
          f"моделей обучено: {stats['models_trained']}")

    # 6. Статус расширенной сессии (≥ 40 минут)
    sess = (get("/api/session") or {}).get("session")
    if sess:
        print(f"[6] Расширенная сессия: {sess['status']}, "
              f"{sess['runs_completed']} прогонов, "
              f"{(sess.get('elapsed_sec') or 0) / 60:.1f} мин")

    # 7. Прогноз
    meta = get("/api/predict-meta")
    if meta.get("features"):
        values = {f: 6.0 for f in meta["features"]}
        pred = post("/api/predict", {"values": values})
        print(f"[7] Прогноз {values} → {pred.get('prediction', pred)}")

    print("\n[OK] Все обращения к API выполнены успешно.")


if __name__ == "__main__":
    main()
