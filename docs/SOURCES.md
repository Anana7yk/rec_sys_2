# Источники: проверено19.10.2026

[Yambda dataset card](https://huggingface.co/datasets/yandex/yambda) описывает observed listens/feedback, organic flag, duration и replay>100%. В study03 используются все listen-строки; это отличается от Listen+ (>50%) benchmark. Реальная схема/диапазоны проверяются локально, а не принимаются из краткой таблицы документации.

[Ploshkin et al., Yambda, §3.3](https://arxiv.org/html/2505.22238v2) и официальный temporal benchmark задают основу интерпретации временных единиц/глобального порядка. Воспроизводимый public snapshot закреплён отдельным HF revision и SHA256; наши сокращённые когорты/каталог и next-row метрики отличаются от опубликованного benchmark.

[Petrov & Macdonald, Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning](https://arxiv.org/abs/2403.04875) рассматривает Next-K генерацию, teacher/student initialization и RL для diversity/popularity goals. Здесь инициализация — next-item CE с нуля, а learned completion reward дополнительно проверяется на action-sensitive signal. Наши результаты не являются воспроизведением опубликованных чисел.

[Matveev et al., Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction, v2](https://arxiv.org/abs/2607.02818) использует two-tower retriever, off-policy REINFORCE, feedback simulator и model-based session evaluation. В study03 нет эквивалентной модели пользовательского перехода: append item — искусственное построение списка. Отсутствие exposure/propensity данных и иные допущения не позволяют переносить их session/OPE интерпретацию на наш experiment.

[KION official repository](https://github.com/irsafilo/KION_DATASET) сохранён как рассмотренный резерв. Новый датасет не добавлялся: бюджет направлен на контроль supervised support, RM ablations и сильные альтернативы RL на одной Yambda-популяции.

Дополнительное ограничение retention: §3.3 статьи Yambda описывает upstream отбор пользователей по активности в начальном и последующем периодах. Поэтому оценка в release условна на эту исходную selection; новый внутренний user holdout не становится случайной выборкой потенциально ушедших пользователей. Это особенно ограничивает интерпретацию возвратов/churn. Календарные интервалы отбора из анонимизированных timestamps не восстанавливаются.
