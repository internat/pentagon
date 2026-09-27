"""
Пример 2 · Свой датасет и нестандартные настройки
==================================================
Три сценария:
  а) создание/подключение собственного CSV,
  б) обучение с другими параметрами (k-fold, доля теста),
  в) задача классификации.

Запуск:  python examples/02_custom_dataset.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from src.pipeline import run_training, load_dataframe, list_datasets

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "example_housing.csv")


def make_dataset():
    """Синтетический датасет «жильё»: площадь/комнаты/возраст → цена."""
    rng = np.random.default_rng(0)
    n = 400
    area = rng.normal(70, 20, n).clip(25, 180)
    rooms = rng.integers(1, 6, n)
    age = rng.uniform(0, 50, n)
    price = (area * 1.9 + rooms * 4.5 - age * 0.8
             + rng.normal(0, 12, n)).round(2)
    df = pd.DataFrame({
        "area_m2": area.round(1),
        "rooms": rooms,
        "age_years": age.round(1),
        "price": price,
    })
    df.to_csv(OUT, index=False)
    return df


def main():
    # --- а) свой CSV -----------------------------------------------------
    if not os.path.isfile(OUT):
        make_dataset()
        print(f"[OK] Создан датасет: {OUT}")
    df = load_dataframe(os.path.basename(OUT))
    print(f"[OK] Загружен: {df.shape[0]} строк × {df.shape[1]} столбцов")
    print("Доступные датасеты:", [d["name"] for d in list_datasets()])

    # --- б) другие параметры разбиения -----------------------------------
    for cv in (3, 10):
        for test_size in (0.1, 0.3):
            r = run_training(dataset=os.path.basename(OUT),
                             cv_folds=cv, test_size=test_size,
                             source="example", make_charts=False)
            print(f"  CV={cv:<2} тест={int(test_size * 100):>2}%  →  "
                  f"{r['best_model']:<40} {r['best_score']:.4f}")

    # --- в) классификация -------------------------------------------------
    clf = run_training(dataset="student_pass.csv",
                       cv_folds=7, test_size=0.25,
                       source="example", make_charts=False)
    print("\nКлассификация (student_pass.csv):")
    print(f"  лучшая модель: {clf['best_model']}")
    print(f"  CV F1 = {clf['best_score']:.4f}")
    for k, v in clf["test_metrics"].items():
        print(f"  тест {k:<10} = {v:.4f}")


if __name__ == "__main__":
    main()
