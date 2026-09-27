"""
Пример 1 · Быстрый старт
========================
Обучение всех моделей на встроенном датасете, выбор лучшей,
финальная оценка на тесте и прогноз.

Запуск:  python examples/01_quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows-консоль cp1251: не даём print упасть на символах вроде «→»/«±»
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import run_training, predict


def main():
    # 1. Полный цикл: предобработка → 4 модели × 5-fold CV → выбор лучшей
    result = run_training(dataset="study_scores.csv",
                          cv_folds=5, test_size=0.2,
                          source="example")

    print("=" * 60)
    print(f"Датасет:  {result['dataset']} ({result['rows']} строк)")
    print(f"Цель:     {result['target']} · задача: {result['task']}")
    print(f"Лучшая:   {result['best_model']}")
    print(f"CV {result['scoring'].upper()} = {result['best_score']:.4f}")
    print("-" * 60)
    print("Все модели:")
    for m in result["models"]:
        print(f"  {m['name']:<40} {m['cv_mean']:.4f} ± {m['cv_std']:.4f}")
    print("-" * 60)
    print("Тестовая оценка лучшей модели:")
    for k, v in result["test_metrics"].items():
        print(f"  {k:<10} = {v:.4f}")
    print("-" * 60)

    # 2. Прогноз сохранённой моделью (models/best_model.pkl)
    forecast = predict({"study_hours": 7.0})
    print(f"Прогноз при study_hours=7.0 → "
          f"{result['target']} ≈ {forecast['prediction']:.2f}")
    print(f"(модель: {forecast['model']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
