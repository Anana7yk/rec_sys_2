# Examples: actual saved recommendations

Rule fixed for this descriptive view: seed11, first three increasing user IDs with a changed pure Top20. These are illustrative, not a representative sample or evidence of user utility. Numbers are anonymized raw item IDs. Curly-bracketed history groups have no within-timestamp order. The logged next item is a ranking label only; no response is assigned to generated actions.

## User 1000, t=24125630

Last available groups: {487660} → {1968345}. Logged next item: 4456727.

| Policy | Top10 item IDs | Top20 proxy | Mean duration, sec |
|---|---|---:|---:|
| pure | [6804216, 6750599, 616275, 2590425, 8013961, 98184, 5380896, 2420782, 4752856, 4305990] | 0.88682 | 154.5 |
| pure_rl | [6804216, 616275, 6750599, 4752856, 2590425, 98184, 2420782, 5380896, 8013961, 1351140] | 0.89336 | 152.0 |
| hybrid | [6804216, 6750599, 8781100, 616275, 4635305, 2656132, 3149530, 1968345, 3932794, 2590425] | 0.87342 | 155.8 |
| hybrid_rl | [6804216, 6750599, 616275, 8781100, 4635305, 2656132, 3932794, 1968345, 3149530, 4752856] | 0.87382 | 153.8 |

## User 4100, t=24343430

Last available groups: {2752429} → {1160424, 3796580, 6776077}. Logged next item: 3154295.

| Policy | Top10 item IDs | Top20 proxy | Mean duration, sec |
|---|---|---:|---:|
| pure | [5862961, 2386490, 6275528, 781438, 6901374, 4242661, 6058126, 6901050, 5866532, 6245596] | 0.69156 | 200.8 |
| pure_rl | [5862961, 6901374, 6678692, 5866532, 1795075, 4242661, 6275528, 2577142, 6058126, 9166040] | 0.72214 | 179.2 |
| hybrid | [9185288, 1071243, 3405787, 7093933, 8023039, 1160424, 7011829, 1304094, 2644025, 2495752] | 0.69944 | 192.8 |
| hybrid_rl | [9185288, 1071243, 8023039, 3405787, 1160424, 2644025, 1304094, 7093933, 7011829, 2495752] | 0.70173 | 194.0 |

## User 4900, t=24218455

Last available groups: {6042277} → {6229666}. Logged next item: 156357.

| Policy | Top10 item IDs | Top20 proxy | Mean duration, sec |
|---|---|---:|---:|
| pure | [2993967, 5916055, 2338988, 8373797, 7557612, 321375, 8005816, 1161112, 4859530, 6335794] | 0.46908 | 217.0 |
| pure_rl | [321375, 8005816, 5916055, 8373797, 2338988, 8245351, 4859530, 1161112, 3492944, 3884920] | 0.47523 | 217.5 |
| hybrid | [8817922, 163697, 6852761, 696835, 8442938, 2267298, 7839672, 6338555, 2951061, 113055] | 0.47300 | 237.8 |
| hybrid_rl | [8817922, 163697, 6852761, 696835, 8442938, 2267298, 7839672, 6338555, 2951061, 113055] | 0.47300 | 237.8 |

Across all500 users, pure RL shifts mean recommended duration by about−10sec and raises popular-decile concentration; hybrid RL shifts it by about−3sec and reduces history repeats. Individual examples can depart from these averages. Aggregate estimates and uncertainty, not these three lists, determine the conclusion.
