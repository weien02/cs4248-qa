# Project status (last updated 2026-10-27)

Read this first after pulling. Results and their interpretation: `report/robustness_error_analysis.md`.

## Branches

- **`zilin`**: everything below. It includes `edbert-dev` (version pins, reranker crash fix), so it is a superset of both branches.
- `main` doesn't have any of this yet. Next step: open a PR from `zilin` into `main`.

## What has been run (SQuAD v1.1, roberta-base, NUS SoC cluster, one seed)

| Step | Job / script | Output |
|---|---|---|
| Baseline on `train_main` | `pipeline.sh` → `baseline` | `out/baseline-roberta-base/` (cluster only). Dev EM 85.79 / F1 92.08 |
| Cross-fitted baselines | `cf0`, `cf1` | `data/candidates/fold{0,1}.jsonl` (cluster only, ~330 MB) |
| Heldout + dev candidates, gating sweep | `dump` | `data/candidates/{heldout,dev}.jsonl`, `results/gating/` |
| Rerankers: hard / easy / random negatives | `rr-{hard,easy,random}` | `out/reranker-*/`, `results/dev/`, `results/ablations.csv` |
| Reranker with soft labels | `rr-hardsoft` (`--soft_labels`) | `results/ablations.csv` (tag `hardsoft`) |
| Robustness: 6 distractor sets + AddSent / AddOneSent | `cluster/robustness.sh` | `results/robustness/` |
| Error analysis (hard, easy, random) | `scripts/error_analysis.py` (CPU) | `results/error_analysis/` |

Model checkpoints, candidate files and per-question prediction files are not in git (too large); they live in `~/cs4248-qa` on the cluster account `gaozilin`. Everything in `results/` except `predictions_*` / `reranker_choice_*` is committed.

## Headline results (dev)

| System | EM | F1 |
|---|---|---|
| Baseline | 85.79 | 92.08 |
| **Gated, hard negatives (final system)** | **86.75** | **92.29** |
| Gated, easy / random negatives | 86.61 / 82.90 | 92.37 / 90.52 |
| Gated, hard + soft labels | 86.31 | 92.16 |
| Rerank everything (no gating), hard | 85.44 | 91.16 |
| Perfect reranker on the gated 18.6% | 90.69 | 94.19 |

The short version:
- Gating is necessary.
- The reranker works mainly by correcting span boundaries.
- Hard ≈ easy negatives because of label noise: training data has one reference answer per question.
- Under AddSent the system drops ~27 EM, like the baseline, and the confidence gate does not detect the attack.

Full analysis: `report/robustness_error_analysis.md`.

## Open to-dos

Suggested owners follow the proposal's work breakdown.

1. **Merge into `main`** (Wei En): open a PR `zilin` → `main`.
2. **Significance / seeds** (Wei En): all gains are from one seed. Either rerun the baseline and hard reranker with 2 more `--seed` values, or run a paired bootstrap on `results/dev/predictions_*` (no GPU needed). This is required before claiming +0.96 EM, and especially hard > easy.
3. **Missing ablations from the proposal:**
   - (Edbert) `--no_score` reranker: tests whether easy/random negatives exploit the baseline-score shortcut. One `train_reranker.py` job, ~30 min.
   - (Hovan) threshold sweep with the **real** reranker: rerun `run_system.py` on dev with `--gate_frac` 0.05/0.1/0.3/0.5. Reranker inference only, a few minutes. `gating_sweep.csv` only covers the perfect-reranker case.
   - (Hovan) k other than 5, optional.
4. **DeBERTa** (Bryan): the proposal mentions it but it was never run. Either drop it from the report, or fix it: the model name is `microsoft/deberta-v3-base`, and it must be added to the download list in `cluster/setup_env.sh` because jobs run offline.
5. **Report and slides** (everyone): Zilin's sections are drafted in `report/robustness_error_analysis.md`. Still to write: introduction and literature review, method, baseline and gating sections, conclusion.

## Cluster notes (things that bit us)

- Use the pinned versions in `requirements.txt` (transformers < 5, datasets < 4). If your `qa` env was built before the pins, re-run `pip install -r requirements.txt` inside it.
- Jobs run with `HF_HUB_OFFLINE=1`: every model must be downloaded first (`cluster/setup_env.sh`).
- The per-user GPU limit is 2 jobs at a time (`QOSMaxGRESPerUser` in `squeue` = waiting for your own job to finish).
- xlogin has very little memory. Downloads there can fail (`Cannot allocate memory`); re-run them, or copy the files in from another machine.
- `results/` is committed. **If you have local result files on the cluster, `git pull` refuses to overwrite them.** Move `results/` aside, pull, then restore with `cp -rn backup/. results/`.
- Your own Python on xlogin needs `source cluster/conda.sh` first; GPU jobs activate the env themselves.
