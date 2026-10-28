Baseline answers on attacked passages, tau=0.16746135687313898 (prob_margin, k=5). fooled = answer lies inside the inserted sentence; gated = would be sent to the reranker.

| set | attacked | fooled | pct_fooled_gated | correct | pct_correct_gated |
|---|---|---|---|---|---|
| addonesent | 787 | 228 | 18.9 | 483 | 12.0 |
| addsent | 2560 | 750 | 17.1 | 1535 | 13.5 |
| overlap-end | 10569 | 150 | 37.3 | 8959 | 14.2 |
| overlap-random | 10570 | 170 | 33.5 | 8922 | 14.0 |
| overlap-start | 10569 | 168 | 37.5 | 8901 | 14.5 |
| random-end | 10569 | 7 | 14.3 | 9057 | 13.9 |
| random-random | 10569 | 15 | 20.0 | 9037 | 14.2 |
| random-start | 10569 | 11 | 9.1 | 9058 | 14.4 |
