# Проверки самостоятельной сдачи

Проверки выполнены 20 октября 2026 при упаковке завершённых экспериментов. Нового исследовательского обучения/поиска не было; ниже явно отделены инженерные mini-fits, чтение весов и пересчёт сохранённых результатов. Все команды запускались из корня нового проекта в отдельном окружении Python3.12.12 с pinned45 зависимостями. Дополнительно только отслеживаемые Git-файлы экспортированы в чистую папку:31 тест и единая команда обоих smoke там тоже прошли, с отдельно скопированными SHA-checked public parquet; старый checkout и модельные архивы для этой проверки не использовались.

| Уровень | Что проверено | Что не утверждается |
|---|---|---|
| Unit/regression tests | 31 passed: shifts, ties, masks, frozen gradients, RL objective/baseline, ranking formulas, macro slates; безопасные manifests/downloads | Это не доказательство качества рекомендаций |
| Маленький end-to-end | Study03 на public development данных: tiny GPT/RM/SFT/H1/H5/duration RL, seal и final; study04 на synthetic inputs: реальные4 RL updates/SFT/grid/selection/final | Smoke-числа не исследовательские результаты |
| Raw и selection | Полные SHA256 двух pinned parquet; exact reconstruction30 frozen массивов dtype/shape/bytes | Полное повторное скачивание 423MB при упаковке не выполнялось |
| Saved ranking replay | Независимо пересчитаны метрики54 study03 и138 study04 outputs; strict past, train-only pool, равные actions, служебные токены/повторы | Пересчёт формул не является переобучением |
| Derived tables | Пересчёт обоих studies и bootstrap из сохранённых outputs; сопоставление43 таблиц по значениям ≤1e−12; README проверяется против CSV | Старый final не становится untouched holdout |
| Actual checkpoint inference | CPU, первые64 общих контекста:21 study03 и42 study04 конфигурации | Не полный replay каждого контекста и не bitwise CPU/MPS equivalence |
| Packaging | Внутренние ссылки, размеры, отсутствие machine paths/типичных токенов в tracked text; exclusions больших payloads; SHA-проверка2215 исходных научных файлов без изменений | Pattern scan не является универсальным security audit |

Результаты автоматических проверок: [сводка пакета](../results/validation/package_checks.json), [study03 ranking/seal](../results/validation/study03_verification.json), [study04 ranking/seal](../results/validation/study04_verification.json), [30 reference arrays](../results/validation/reference_reconstruction.json), [study03 CPU inference](../results/validation/study03_replay.json), [study04 CPU inference](../results/validation/study04_replay.json). [Проверка8 внешних URL](../results/validation/external_links.json), [сохранность исходников исследований](../results/validation/original_preservation.json). Данные/веса и подробные per-user logs не коммитятся.

CPU/MPS float32 могут менять порядок близких scores. В study03 минимум совпадения Top20 rank cells99.84375%; максимальное расхождение mean proxy на отдельном контексте **0.0013491** (H5 seed29), а не ноль. В study04 минимум99.84375%, максимальная разница0.00008744. Точные результаты всех конфигураций хранятся в JSON, без подмены near-tie перестановок полным совпадением.

Архивные seals: study03 проверено51 payload и отдельно11 старых source records; study04 проверено343 payload, отдельно15 старых source records. Старые source hashes не перепривязаны к перенесённому коду и не читают старые абсолютные пути. Новые relative-path seals проверены в smoke. Source нового проекта фиксируется Git commit; неизменность исходных архивов подтверждена полными SHA. См. [PORTING](PORTING.md).

**Полное исследовательское обучение в чистом окружении не повторялось.** В новом окружении повторены unit tests, оба маленьких pipeline, exact reconstruction, derived tables, ranking metrics и limited checkpoint inference. Затраты полного исследования в README/отчёте относятся к выполненным историческим fit. Установка lock проверена на macOS arm64; перенос wheels/скорости/MPS на другую платформу не обещается. Повторение полного обучения доступно отдельной командой из публичных данных.
