# Воспроизведение

Все команды выполняются **из корня нового репозитория**. Исходная рабочая папка и архивы прошлых study_01/02 не требуются. Output новых fit должен быть новым, существующие исследования не перезаписываются.

## Окружение и данные

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/download_data.py --verify-only
.venv/bin/python -m pytest -q
```

Python3.12, pinned lock. Реально проверено отдельное окружение Python3.12.12 на macOS arm64, созданное `uv venv --python 3.12` / `uv pip sync --python .venv/bin/python requirements.lock.txt`. Данные HF закреплены revision/SHA в [манифесте](../configs/data_sources.json); инструкция по тяжёлым архивам — [ARTIFACTS](ARTIFACTS.md). Для команд fit/evaluation `--device cpu` доступен вместо `mps`.

## Быстрый сквозной тест

```bash
.venv/bin/python scripts/reproduce.py smoke --out runs/smoke
```

Study03 smoke использует маленькую development-выборку реальных данных, обучает tiny GPT/RM, продолжение CE, RL H1/H5 и modified reward, выполняет selection/seal/final. Study04 smoke строит синтетические frozen inputs, реально выполняет4 RL updates, time-matched SFT, обе фиксированные сетки, selection/seal, ranking и greedy slates. Это инженерные проверки, их значения не включены в исследовательские таблицы. Повторный запуск — с другим `--out`.

## Полный исследовательский запуск

```bash
.venv/bin/python scripts/reproduce.py full --out runs/full --reports validation/full --device mps
```

Включает прежний ограниченный supervised pilot, RM, выбор только по validation, три основных GPT seeds, hybrid controls/RL, frozen selection, final, pure RL/SFT дополнения, λ-сетку, повторную final оценку, bootstrap и графики. На другом устройстве число time-matched SFT updates и численные траектории могут отличаться. Такая повторная тренировка не превращает уже изученный final в новое независимое подтверждение. Исторический old-GPT checkpoint отдельного более раннего рецепта не переобучается этой командой; основные обязательные модели самостоятельны.

Раздельные стадии (один и тот же новый путь `runs/manual03`):

```bash
# Prepare/reconstruct exact rows + full eligible train pool; audits.
.venv/bin/python experiments/study03/prepare.py --out runs/manual03
.venv/bin/python experiments/study03/distributions.py --out runs/manual03
# Historical bounded supervised pilot, then frozen choice / 3 main seeds.
.venv/bin/python experiments/study03/train.py pilot --out runs/manual03 --device mps
# Fit continuous-response RM + evaluator and choose in-catalog RM by validation.
.venv/bin/python experiments/study03/run.py reward --out runs/manual03 --device mps
.venv/bin/python experiments/study03/select_reward.py --out runs/manual03
.venv/bin/python experiments/study03/previous_control.py marginal --out runs/manual03 --device mps
.venv/bin/python experiments/study03/train.py main --out runs/manual03 --device mps
# Baselines, time-matched SFT, rerank, H1/H5 RL, raw/duration correction; seal.
.venv/bin/python experiments/study03/run.py policies --out runs/manual03 --device mps
# Final evaluated only after selection/seal.
.venv/bin/python experiments/study03/run.py final --out runs/manual03 --device mps
.venv/bin/python experiments/study03/previous_control.py assess --out runs/manual03 --device mps
# Supplement inputs from THIS run; pure RL/SFT, alternatives and evaluation.
.venv/bin/python experiments/study04/prepare_inputs.py --study03 runs/manual03 --out runs/manual_inputs
.venv/bin/python experiments/study04/run.py all --inputs runs/manual_inputs --out runs/manual04 --device mps
# Derived tables/figures; no fits or checkpoint changes.
.venv/bin/python experiments/study03/summarize.py --out runs/manual03 --reports validation/manual03
.venv/bin/python experiments/study03/reward_domain_analysis.py --out runs/manual03 --reports validation/manual03
.venv/bin/python experiments/study03/plot_data.py --out runs/manual03 --reports validation/manual03
.venv/bin/python experiments/study03/augment_horizon.py --out runs/manual03 --reports validation/manual03
.venv/bin/python experiments/study04/summarize.py --inputs runs/manual_inputs --out runs/manual04 --reports validation/manual04
.venv/bin/python experiments/study04/plots.py --out runs/manual04 --reports validation/manual04
```

Configs [study03](../configs/study03/study.json), [selected supervised recipe](../configs/study03/selected_supervised.json), [study04](../configs/study04/supplement.json), [methods](METHODS.md). Seeds11/29/47; data seed20261020. Основной GPT checkpoint выбирается по validation NDCG20. RL initial/half/end и SFT initial/best-NDCG/last сравниваются по raw R20 при NDCG≥95% reference; если недостаточно допустимых изменений, остаётся initial. β0.1/H1 budget640 фиксированы, H5 budget128 при равных действиях. Supplement коэффициенты выбираются только на validation по правилам1/3/5%-guard и reward matching; все неуспешные точки также публикуются. Final не используется для дополнительной настройки.

## Replay завершённого эксперимента

После SHA-checked импорта двух архивов:

```bash
.venv/bin/python scripts/reproduce.py tables
.venv/bin/python experiments/study03/replay.py --device cpu --limit 64
.venv/bin/python experiments/study04/replay.py --device cpu --limit 64
.venv/bin/python experiments/study03/reference.py rebuild --out validation/reference_new
.venv/bin/python scripts/check_submission.py --recomputed validation/recomputed
```

`tables` не обучает модели: пересчитывает агрегаты/CI из сохранённых per-event результатов, диагностические оценки RM и проверки rankings; пишет только в `validation/recomputed`. Replay заново читает веса и получает рекомендации на первых64 фиксированных контекстах, отдельно от пересчёта формул по сохранённым ranking IDs. Float32CPU/MPS могут менять почти равные scores; bitwise совпадение не обещается. Reconstruction сравнивает30 массивов по точным dtype/shape/bytes hashes с frozen manifest.

Точное воспроизведение опубликованных исходных чисел и повторение полного обучения — разные уровни. [VALIDATION](VALIDATION.md) перечисляет реально выполненные проверки нового пакета. **Полное research-переобучение в чистом окружении не повторялось.**
