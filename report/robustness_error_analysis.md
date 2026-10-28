# Robustness and Error Analysis (draft report sections)

Owner: Zilin. Every number here comes from files in `results/`; the commands that regenerate them are in the README ("Robustness" section) and `report/STATUS.md`.

**System under test:** roberta-base span baseline (trained on `train_main`), top-k = 5 candidates, confidence = probability margin between the top two candidates. τ is set on heldout so that the 20% least-confident questions are reranked (τ = 0.1675, which sends 18.6% of dev). Rerankers: roberta-base cross-encoders trained on cross-fitted fold0/fold1 candidates. All numbers are SQuAD v1.1 dev with the official EM/F1 (best match over all reference answers), single training seed.

## 1. Main results

| System | EM | F1 | Fixed | Broken |
|---|---|---|---|---|
| Baseline | 85.79 | 92.08 | | |
| **Gated, hard negatives (proposed)** | **86.75** | **92.29** | 274 | 172 |
| Gated, easy negatives | 86.61 | 92.37 | 268 | 181 |
| Gated, random negatives | 82.90 | 90.52 | 179 | 484 |
| Gated, hard negatives + soft labels | 86.31 | 92.16 | 249 | 194 |
| Rerank all, hard | 85.44 | 91.16 | 462 | 499 |
| Rerank all, easy | 84.81 | 91.43 | 453 | 557 |
| Rerank all, random | 57.43 | 76.16 | 345 | 3,343 |
| Upper bound: perfect reranker on the gated 18.6% | 90.69 | 94.19 | 518 | 0 |

Source: `results/ablations.csv`. "Fixed"/"broken" count questions whose exact match changes compared with the baseline.

- Selective reranking gives **+0.96 EM / +0.21 F1**. It reaches about 20% of the +4.9 EM that a perfect reranker could gain on the same gated questions.
- **Gating is necessary.** The same reranker applied to every question loses 0.35 EM, because on confident questions it breaks more answers than it fixes.
- **Hard and easy negatives are statistically indistinguishable** (+0.14 EM, −0.08 F1). Random negatives hurt. Section 2.5 explains the tie.

Gold-answer recall of the baseline's candidate list on dev: 85.8% at rank 1, **96.1% within the top 5**, 97.7% within the top 10, 98.6% within the top 20. So k = 5 already contains the answer for almost every rerankable question.

## 2. Error analysis

Source: `results/error_analysis/<tag>/` (`summary.md`, CSVs, `calibration.png`), from `scripts/error_analysis.py`.

### 2.1 Baseline errors are mostly span-boundary errors

| Baseline outcome | % of dev | Share of errors |
|---|---|---|
| Correct | 85.79 | |
| **Boundary**: overlaps a reference answer but not exact | **10.23** | **72%** |
| Ranking: no overlap, gold in the top 5 | 1.79 | 13% |
| Gold at ranks 6–20 | 1.28 | 9% |
| Gold not in the 20-best list | 0.92 | 6% |

The baseline almost always finds the right region of the passage. Its errors are mainly in where the answer starts and ends.

### 2.2 The reranker acts as a boundary corrector

Of the 274 questions the gated system fixes, **88%** were near-misses (the baseline's answer overlapped a reference). Of the 172 it breaks, **80%** become near-misses. Typical cases:

- Fixed: *"What caused the delay during the development of the AC motor?"* The baseline answered *"financial strain of buying up patents and hiring the engineers needed to build it"*; the reranker chose *"financial strain"* (gold).
- Fixed: *"Which famous Dutch artist who painted The Night Watch is represented …?"* The baseline answered *"Dürer"*; the reranker chose *"Rembrandt"* (gold, ranked 3rd by the baseline). This is one of the rarer ranking fixes.
- Broken: *"How many years have imperialistic practices existed?"* The baseline answered *"thousands"* (gold); the reranker chose *"thousands of years"*.

### 2.3 The confidence gate is well calibrated

| Confidence decile (1 = least confident) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8–10 |
|---|---|---|---|---|---|---|---|---|
| Baseline EM | 61.1 | 69.0 | 77.7 | 82.0 | 85.2 | 90.5 | 95.7 | 97.6–99.9 |
| Reranker applied to every question | **68.8** | **71.4** | 77.4 | 80.0 | 81.4 | 87.6 | 93.3 | 96.0–99.6 |

- Baseline accuracy rises steadily with confidence, so the probability margin is a good uncertainty signal.
- The reranker gains +7.7 EM in decile 1 and +2.5 in decile 2. It breaks even at decile 3 and **hurts in every decile above**, by up to 3.9 EM.
- The 20% threshold chosen on heldout lands at the break-even point. Figure: `results/error_analysis/hard/calibration.png`.

The gate catches 50% of boundary errors and 35% of ranking errors. However, 65% of the 1,964 gated questions were already correct, and those are where all 172 broken answers come from. Gated questions whose gold answer is outside the top 5 (83) cannot be fixed by any reranker.

### 2.4 Who benefits: why-questions and long answers

| Group | n | Baseline EM | Gated EM | Δ |
|---|---|---|---|---|
| why | 151 | 58.9 | 69.5 | **+10.6** |
| gold answer of 6+ words | 592 | 72.5 | 77.2 | **+4.7** |
| how much | 154 | 85.7 | 87.0 | +1.3 |
| what | 5,895 | 85.3 | 86.3 | +1.0 |
| which | 708 | 88.3 | 89.3 | +1.0 |
| when | 695 | 93.4 | 94.5 | +1.2 |
| who | 1,175 | 92.0 | 92.2 | +0.2 |
| where | 460 | 79.6 | 79.4 | −0.2 |
| 1-word answers | 4,923 | 88.6 | 89.4 | +0.8 |

The gains concentrate where the correct boundary is most ambiguous: explanatory why-questions and long answers. Short factual answers gain little. The why-group is small, so +10.6 is indicative, not exact.

### 2.5 Why hard negatives don't beat easy negatives: label noise

Training questions have **one** reference answer, whereas dev has about three, and 44% of dev questions have at least two distinct correct answers. A candidate is labelled negative whenever it isn't an exact match of that single reference. We measured how often this mislabels correct answers by applying the training rule to dev: the first reference counts as gold, and the other references are used only for checking.

| Negatives | n | **Exactly another annotator's answer** | Longer span containing gold | Shorter span inside gold | Partial overlap | Unrelated |
|---|---|---|---|---|---|---|
| Hard (top 5) | 32,470 | **13.1%** | 56.3% | 22.7% | 3.4% | 4.5% |
| Easy (ranks 6–20) | 139,409 | 1.7% | 60.7% | 19.2% | 8.6% | 9.8% |
| *Training folds, hard* | 253,747 | n/a (single reference) | 59.9% | 31.2% | 3.3% | 5.5% |

- **About 1 in 8 hard negatives is a correct answer labelled wrong**, nearly 8× the rate for easy negatives.
- Both pools consist mostly of boundary variants of the gold span: about 90% overlap it. That's because the 20-best list is dominated by small variations around the same span. So "hard" and "easy" negatives are far more alike than their names suggest.
- The hard pool's better training signal is offset by its much higher label noise.
- This also explains the gap between heldout and dev scores (baseline EM 69.85 vs 85.79). Rescoring dev with only its first reference gives 71.43 EM / 84.73 F1, matching heldout.

### 2.6 Ablation: soft labels make things worse

We retrained the hard reranker with soft targets: each candidate's weight is F1(candidate, gold)², with the gold candidate at 1, normalised (`--soft_labels`). Result: gated **86.31 EM** (−0.44 vs hard) and rerank-all 84.40 EM (−1.04). Rerank-all F1 is essentially unchanged (91.23 vs 91.16), and the heldout check agrees (68.78 vs 70.14 EM).

Interpretation: the reranker's value is boundary precision (Section 2.2). Soft labels gave partial credit to *every* overlapping span: not only the 13% that are other annotators' answers, but also the roughly 80% that are simply too long or too short. The model learned that being approximately right is enough, which costs EM. Fixing the label noise properly needs a way to tell "another valid answer" from "a sloppy boundary", and single-reference training data can't provide one.

## 3. Robustness

Source: `results/robustness/` (`robustness_summary.md`, `gate_vs_capture.md`, `robustness_drop.png`), from `cluster/robustness.sh` and `scripts/robustness_report.py`.

**Test sets:**
- **Six distractor versions of dev** (`scripts/make_distractors.py`). Each passage gets one sentence from a *different* article, never one containing a gold answer. The sentence is either **random** or the one with the **most word overlap with the question**, inserted at the passage **start**, a **random** sentence boundary, or the **end**.
- **Adversarial SQuAD** (Jia & Liang, 2017). **AddSent**: human-checked sentences written to look like they answer the question. **AddOneSent**: one such sentence per question. Both use the official worst-case score over each of the 1,000 original questions' versions.
- τ stays at the value chosen on clean heldout.

**Capture rate** = the share of answers taken from inside the inserted sentence.

| Test set | Baseline ΔEM | Gated (hard) ΔEM | Baseline capture | % sent to reranker |
|---|---|---|---|---|
| random-start | −0.09 | −0.04 | 0.1% | 18.7 |
| random-random | −0.28 | −0.23 | 0.1% | 18.7 |
| random-end | −0.09 | +0.01 | 0.1% | 18.6 |
| overlap-start | −1.57 | −1.33 | 1.6% | 19.5 |
| overlap-random | −1.38 | −1.30 | 1.6% | 18.8 |
| overlap-end | −1.02 | −0.95 | 1.4% | 19.0 |
| **AddOneSent** | **−19.1** (84.9 → 65.8) | −20.3 (86.7 → 66.4) | 29.0% | 17.2 |
| **AddSent** | **−26.9** (84.9 → 58.0) | −28.0 (86.7 → 58.7) | 29.3% | 18.1 |

ΔEM is measured against the same questions on clean text. For AddSent and AddOneSent the clean score comes from their 1,000 original questions.

### Findings

1. **Irrelevant text is harmless; question-like text is not.**
   - Random off-topic sentences change EM by less than 0.3 at any position.
   - Sentences that share words with the question cost about 1–1.6 EM, slightly more at the start of the passage.
   - AddSent costs about 27 EM: 29% of the baseline's answers come from the adversarial sentence. Example: *"What day was the Super Bowl played on?"* With *"The Champ Bowl was played on the day of Easter."* appended, the baseline answers *"Easter"*.
2. **The gate does not detect adversarial attacks** (`gate_vs_capture.md`). On AddSent, only **17.1%** of the 750 fooled baseline answers would be sent to the reranker, hardly more than the 13.5% of correct answers. The baseline is confidently wrong. The share of questions gated is also unchanged (18.1% vs 18.6% on clean dev). Even a perfect reranker on the gated questions only recovers 4.7 of the ~27 EM lost. For the milder overlap distractors, fooled answers *are* gated more often than correct ones (33–38% vs 14%), so the gate partly notices them.
3. **The reranker is drawn to adversarial sentences.** Applied to every question on AddSent, the hard reranker drops further than the baseline (−30.3 vs −26.9 EM), and its capture rate rises from 29.3% to 31.4% (easy: 29.2%). It judges a candidate by how well its sentence matches the question, which is exactly what AddSent sentences are designed to exploit. Gating limits the damage, but the system's clean-dev advantage over the baseline largely disappears under attack (+0.7 EM, −0.16 F1 on AddSent).
4. On every distractor set, the gated system keeps its roughly +1 EM advantage over the baseline. Its drop is about the same as the baseline's or a little smaller, but the difference is within noise.

## 4. Limitations

- **One training seed for every model.** Gains under about 0.5 EM (hard vs easy, the robustness differences between systems) are within run-to-run noise. A multi-seed run or a paired bootstrap on dev is needed before claiming them.
- AddSent and AddOneSent cover 1,000 questions, so differences under about 1–2 EM are noise.
- Heldout has one reference answer per question, so its scores are only comparable with other heldout scores.
- The reranker uses the same encoder as the baseline (roberta-base). We didn't test whether the gains grow with a weaker baseline or a stronger reranker.
