# Источники и права

Данные не включены в Git. [Yambda dataset card](https://huggingface.co/datasets/yandex/yambda) указывает Apache-2.0; это атрибуция условий источника, а не новая лицензия от автора этой сдачи. При скачивании/использовании следует соблюдать актуальные условия самого источника. Версия и SHA закреплены в `configs/data_sources.json`.

Научные источники:

- [Petrov & Macdonald: Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning](https://arxiv.org/abs/2403.04875).
- [Matveev et al.: Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction](https://arxiv.org/abs/2607.02818).
- [Yambda: A Large-Scale Dataset for Recommender Systems](https://arxiv.org/abs/2505.22238).
- [KION official dataset repository](https://github.com/irsafilo/KION_DATASET) рассмотрен как резерв; данные и код KION не использованы.

Работа реализует собственный эксперимент по мотивам этих постановок, не заявляет воспроизведение их benchmark-чисел и не включает их исходный код. Используемые библиотеки перечислены с версиями в `requirements.lock.txt`; их собственные лицензии сохраняются. Настоящая сдача не назначает сторонним данным, статьям или библиотекам произвольную лицензию. Для авторского исследовательского кода отдельная открытая лицензия пока не выбрана; наличие публичного GitHub-репозитория само по себе не означает выдачу такой лицензии.
