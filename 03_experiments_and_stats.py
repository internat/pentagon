"""
Пример 3 · Эксперименты, статистика и собственная математика
=============================================================
  а) 4 серии исследовательских экспериментов,
  б) чтение истории всех тренировок и сводной статистики,
  в) проверка собственной реализации линейной регрессии.

Запуск:  python examples/03_experiments_and_stats.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import (load_dataframe, suggest_target, run_experiments_stream,
                          aggregate_stats, custom_linear_demo)


def collect(generator):
    """Собирает события стрима, возвращает последнее 'result'."""
    final = None
    for ev in generator:
        print("  ·", ev.get("msg"))
        if ev.get("step") == "done":
            final = ev.get("result")
    return final


def main():
    # --- а) эксперименты ---------------------------------------------------
    print("=" * 60)
    print("СЕРИЯ ЭКСПЕРИМЕНТОВ (study_scores.csv)")
    print("=" * 60)
    df = load_dataframe("study_scores.csv")
    exp = collect(run_experiments_stream("study_scores.csv",
                                         suggest_target(df)))

    print("\n1) Влияние доли train/test:")
    for k, v in exp.items():
        if k.startswith("split_"):
            print(f"   {k.replace('split_', ''):>8} → лучший CV {v:.4f}")

    print("\n2) Сравнение моделей (80/20):")
    for name, score in sorted(exp.get("model_comparison", {}).items(),
                              key=lambda x: -x[1]):
        print(f"   {name:<40} {score:.4f}")

    print("\n3) Дообучение на новых данных (Δ):")
    rt = exp.get("retraining", {})
    for name, before in rt.get("old_results", {}).items():
        after = rt.get("new_results", {}).get(name)
        if after is not None:
            print(f"   {name:<40} {before:.4f} → {after:.4f} "
                  f"({after - before:+.4f})")

    # --- б) статистика -----------------------------------------------------
    print("\n" + "=" * 60)
    print("СВОДНАЯ СТАТИСТИКА ПО ВСЕМ ТРЕНИРОВКАМ")
    print("=" * 60)
    stats = aggregate_stats()
    print(f"  Прогонов:            {stats['total_runs']}")
    print(f"  Моделей обучено:     {stats['models_trained']}")
    print(f"  Fold-проходов:       {stats['cv_folds_total']}")
    print(f"  Суммарное время:     {stats['total_train_time_sec']:.1f} с")
    print(f"  Датасетов:           {len(stats['datasets'])}")
    if stats.get("best_overall"):
        b = stats["best_overall"]
        print(f"  Лучший результат:    {b['score']:.4f} "
              f"({b['model']} · {b['dataset']})")
    print("  Победы моделей:")
    for name, wins in sorted(stats.get("winners", {}).items(),
                             key=lambda x: -x[1]):
        print(f"    {name:<40} {wins}")
    session = stats.get("session")
    if session:
        print(f"  Расширенная сессия:  {session.get('status')}, "
              f"{session.get('runs_completed')} прогонов, "
              f"{(session.get('elapsed_sec') or 0) / 60:.1f} мин")

    # --- в) собственная реализация ----------------------------------------
    print("\n" + "=" * 60)
    print("СОБСТВЕННАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ vs SKLEARN")
    print("=" * 60)
    demo = custom_linear_demo("study_scores.csv")
    for row in demo["rows"]:
        print(f"  {row['method']:<32} RMSE={row['rmse']:.4f}  "
              f"R2={row['r2']:.4f}")
    print(f"  Δ Normal Equation vs sklearn: "
          f"{demo['max_diff_normal_equation']:.2e}")
    print(f"  Δ Gradient Descent vs sklearn: "
          f"{demo['max_diff_gradient_descent']:.2e}")
    print(f"  Формула: {demo['formula']}")


if __name__ == "__main__":
    main()
