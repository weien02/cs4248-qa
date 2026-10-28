# Error analysis: `easy` on `dev.jsonl`
10570 questions, k=5, prob_margin tau=0.1675, 18.6% sent to the reranker.

## Where errors come from (% of all questions)
| error_type | baseline | gated | rerank_all |
|---|---|---|---|
| correct | 85.79 | 86.61 | 84.81 |
| boundary | 10.23 | 9.14 | 10.43 |
| ranking | 1.79 | 1.93 | 2.45 |
| recall_k | 1.28 | 1.36 | 1.33 |
| recall_all | 0.92 | 0.96 | 0.98 |
`ranking` is what a reranker can fix; `recall_*` needs a better baseline or larger k.

## Does the confidence gate catch the right questions?
| baseline_outcome | n | gated | pct_gated | gated_and_correct_after |
|---|---|---|---|---|
| correct | 9068 | 1270 | 14.0 | 1089 |
| boundary | 1081 | 544 | 50.3 | 235 |
| ranking | 189 | 67 | 35.4 | 33 |
| recall_k | 135 | 43 | 31.9 | 0 |
| recall_all | 97 | 40 | 41.2 | 0 |
Ideally `ranking` errors have a high `pct_gated` and `correct` a low one.

## By question type
| question_type | n | EM_baseline | EM_gated | EM_rerank_all | EM_oracle_gated | wrong_to_right | right_to_wrong | pct_gated |
|---|---|---|---|---|---|---|---|---|
| how | 398 | 70.60 | 72.11 | 68.84 | 80.65 | 22 | 16 | 32.4 |
| how many | 642 | 90.19 | 90.50 | 87.07 | 93.61 | 12 | 10 | 10.1 |
| how much | 154 | 85.71 | 86.36 | 85.71 | 92.86 | 3 | 2 | 16.9 |
| other | 292 | 80.48 | 82.53 | 83.90 | 86.64 | 9 | 3 | 18.2 |
| what | 5895 | 85.34 | 86.04 | 84.07 | 90.31 | 141 | 100 | 20.1 |
| when | 695 | 93.38 | 94.82 | 93.96 | 96.55 | 13 | 3 | 7.1 |
| where | 460 | 79.57 | 79.13 | 79.78 | 86.09 | 17 | 19 | 27.0 |
| which | 708 | 88.28 | 90.25 | 89.69 | 92.09 | 18 | 4 | 16.4 |
| who | 1175 | 92.00 | 92.00 | 89.53 | 94.21 | 16 | 16 | 12.8 |
| why | 151 | 58.94 | 64.90 | 60.26 | 78.15 | 17 | 8 | 43.7 |

## By gold answer length (words)
| answer_words | n | EM_baseline | EM_gated | EM_rerank_all | EM_oracle_gated | wrong_to_right | right_to_wrong | pct_gated |
|---|---|---|---|---|---|---|---|---|
| 1 | 4923 | 88.60 | 89.36 | 87.16 | 92.50 | 109 | 72 | 14.8 |
| 2 | 2709 | 85.83 | 86.42 | 85.05 | 90.11 | 57 | 41 | 18.4 |
| 3 | 1382 | 84.15 | 84.66 | 83.57 | 90.23 | 44 | 37 | 21.9 |
| 4-5 | 964 | 81.85 | 81.95 | 79.98 | 87.97 | 23 | 22 | 25.4 |
| 6+ | 592 | 72.47 | 76.86 | 74.83 | 83.78 | 35 | 9 | 32.3 |

## By baseline confidence decile (1 = least confident)
| confidence_decile | n | EM_baseline | EM_gated | EM_rerank_all | EM_oracle_gated | wrong_to_right | right_to_wrong | pct_gated |
|---|---|---|---|---|---|---|---|---|
| 1 | 1057 | 61.12 | 66.89 | 66.89 | 89.69 | 159 | 98 | 100.0 |
| 2 | 1057 | 68.97 | 71.43 | 71.71 | 89.40 | 109 | 83 | 85.8 |
| 3 | 1057 | 77.67 | 77.67 | 76.54 | 77.67 | 0 | 0 | 0.0 |
| 4 | 1057 | 82.02 | 82.02 | 79.66 | 82.02 | 0 | 0 | 0.0 |
| 5 | 1057 | 85.24 | 85.24 | 79.85 | 85.24 | 0 | 0 | 0.0 |
| 6 | 1057 | 90.54 | 90.54 | 86.57 | 90.54 | 0 | 0 | 0.0 |
| 7 | 1057 | 95.65 | 95.65 | 92.15 | 95.65 | 0 | 0 | 0.0 |
| 8 | 1057 | 97.63 | 97.63 | 96.40 | 97.63 | 0 | 0 | 0.0 |
| 9 | 1057 | 99.15 | 99.15 | 98.49 | 99.15 | 0 | 0 | 0.0 |
| 10 | 1057 | 99.91 | 99.91 | 99.81 | 99.91 | 0 | 0 | 0.0 |

## What do the reranker's flips swap? (gated system)
| flip | n | wrong_side | near_miss | unrelated | pct_near_miss |
|---|---|---|---|---|---|
| wrong_to_right | 268 | baseline | 235 | 33 | 87.7 |
| right_to_wrong | 181 | gated | 152 | 29 | 84.0 |
`near_miss`: the wrong answer overlaps a reference (span boundary off). `unrelated`: no overlap.

## Are the reranker's training negatives really wrong? (% of negatives)
| source | negatives | n | matches_other_ref | contains_gold | inside_gold | partial_overlap | no_overlap |
|---|---|---|---|---|---|---|---|
| dev.jsonl (first ref only) | hard (top-k) | 32470 | 13.1 | 56.3 | 22.7 | 3.4 | 4.5 |
| dev.jsonl (first ref only) | easy (rank k+) | 139409 | 1.7 | 60.7 | 19.2 | 8.6 | 9.8 |
| fold0.jsonl+fold1.jsonl | hard (top-k) | 253747 | 0.0 | 59.9 | 31.2 | 3.3 | 5.5 |
| fold0.jsonl+fold1.jsonl | easy (rank k+) | 1087054 | 0.0 | 59.4 | 21.0 | 8.4 | 11.2 |
Negatives are labelled as in training: not an exact match of ONE reference answer. `matches_other_ref` = another annotator gave exactly this answer (a false negative; only measurable on dev). `contains_gold` / `inside_gold` = a longer / shorter span around the answer, often also acceptable.

## Examples
### Fixed by the reranker (wrong -> right) (268 total)
- **Q:** What do C4 plants need?  
  **gold:** 'more ATP than NADPH' | **baseline:** 'more ATP' | **gated:** 'more ATP than NADPH' | conf=0.148 | type=what  
  **top-5:** 1. 'more ATP' (8.0); 2. 'more ATP than NADPH' (7.6); 3. 'ATP' (6.9); 4. 'ATP than NADPH' (6.5); 5. 'ATP' (4.7)
- **Q:** Where were interviews held while the parliament was in its temporary building?  
  **gold:** 'courtyard' | **baseline:** 'in the courtyard adjoining the Assembly Hall, which is part of the School of Divinity of the University of Edinburgh' | **gated:** 'the courtyard adjoining the Assembly Hall' | conf=0.055 | type=where  
  **top-5:** 1. 'in the courtyard adjoining the Assembly Hall, which is part of the School of Divinity of the University of Edinburgh' (10.9); 2. 'in the courtyard adjoining the Assembly Hall' (10.7); 3. 'the courtyard adjoining the Assembly Hall, which is part of the School of Divinity of the University of Edinburgh' (10.6); 4. 'the courtyard adjoining the Assembly Hall' (10.4); 5. 'courtyard adjoining the Assembly Hall, which is part of the School of Divinity of the University of Edinburgh' (10.0)
- **Q:** When did the accident occur?  
  **gold:** 'the fall of 1937' | **baseline:** '1937' | **gated:** 'fall of 1937' | conf=0.137 | type=when  
  **top-5:** 1. '1937' (7.1); 2. 'fall of 1937' (6.8); 3. 'the fall of 1937' (5.6); 4. 'In the fall of 1937' (5.3); 5. '1937, after midnight one night' (2.9)
- **Q:** What is the main reason consulting pharmacists are increasingly working directly with patients?  
  **gold:** 'because many elderly people are now taking numerous medications but continue to live outside of institutional settings' | **baseline:** 'many elderly people are now taking numerous medications' | **gated:** 'many elderly people are now taking numerous medications but continue to live outside of institutional settings' | conf=0.119 | type=what  
  **top-5:** 1. 'many elderly people are now taking numerous medications' (15.2); 2. 'many elderly people are now taking numerous medications but continue to live outside of institutional settings' (14.9); 3. 'elderly people are now taking numerous medications' (14.0); 4. 'elderly people are now taking numerous medications but continue to live outside of institutional settings' (13.7); 5. 'because many elderly people are now taking numerous medications' (12.0)
- **Q:** How did Vaudreuil react when Johnson was seen as larger threat?  
  **gold:** 'sent Dieskau to Fort St. Frédéric to meet that threat' | **baseline:** 'Vaudreuil sent Dieskau to Fort St. Frédéric' | **gated:** 'sent Dieskau to Fort St. Frédéric' | conf=0.143 | type=when  
  **top-5:** 1. 'Vaudreuil sent Dieskau to Fort St. Frédéric' (8.8); 2. 'Vaudreuil sent Dieskau to Fort St. Frédéric to meet that threat' (8.3); 3. 'sent Dieskau to Fort St. Frédéric' (8.2); 4. 'sent Dieskau to Fort St. Frédéric to meet that threat' (7.7); 5. 'Vaudreuil sent Dieskau' (6.6)
- **Q:** The Kronenberg Palace had been an exceptional example of what type of architecture?  
  **gold:** 'bourgeois' | **baseline:** 'bourgeois architecture' | **gated:** 'bourgeois' | conf=0.073 | type=what  
  **top-5:** 1. 'bourgeois architecture' (15.3); 2. 'bourgeois' (15.1); 3. 'the bourgeois architecture' (12.5); 4. 'the bourgeois' (12.3); 5. 'bourgeois architecture of the later periods' (11.0)
- **Q:** What conjecture holds that there are always a minimum of 4 primes  between the squares of consecutive primes greater than 2?  
  **gold:** "Brocard's conjecture" | **baseline:** "H. Brocard's" | **gated:** "Brocard's conjecture" | conf=0.100 | type=what  
  **top-5:** 1. "H. Brocard's" (14.7); 2. "H. Brocard's conjecture" (14.3); 3. 'H. Brocard' (14.1); 4. "Brocard's" (14.0); 5. "Brocard's conjecture" (13.6)
- **Q:** Why did the university see a drop in applicants?   
  **gold:** 'a result of increasing crime and poverty' | **baseline:** 'increasing crime and poverty' | **gated:** 'increasing crime and poverty in the Hyde Park neighborhood' | conf=0.074 | type=why  
  **top-5:** 1. 'increasing crime and poverty' (16.5); 2. 'increasing crime and poverty in the Hyde Park neighborhood' (16.3); 3. 'as a result of increasing crime and poverty' (13.2); 4. 'crime and poverty' (13.2); 5. 'as a result of increasing crime and poverty in the Hyde Park neighborhood' (13.1)
- **Q:** How did the United States plan to subdue imperialistic tendencies?   
  **gold:** 'the Monroe Doctrine' | **baseline:** 'military force' | **gated:** 'Monroe Doctrine' | conf=0.022 | type=how  
  **top-5:** 1. 'military force' (-1.2); 2. 'the Monroe Doctrine' (-1.3); 3. 'Monroe Doctrine' (-1.4); 4. 'backed by military force' (-1.8); 5. 'military force, but were more often affected from behind the scenes' (-2.4)
- **Q:** What report had the correct date?  
  **gold:** '"Variations of Snow and Ice in the past and at present on a Global and Regional Scale"' | **baseline:** 'WWF' | **gated:** 'WWF report' | conf=0.105 | type=what  
  **top-5:** 1. 'WWF' (13.0); 2. 'WWF report' (12.8); 3. 'the WWF' (11.3); 4. 'the WWF report' (11.0); 5. '2035 has been correctly quoted by the IPCC from the WWF' (9.5)

### Broken by the reranker (right -> wrong) (181 total)
- **Q:** What is a mechanism that can help plants block virus replication?  
  **gold:** 'RNA silencing mechanisms' | **baseline:** 'RNA silencing mechanisms' | **gated:** 'RNA silencing' | conf=0.139 | type=what  
  **top-5:** 1. 'RNA silencing mechanisms' (16.6); 2. 'RNA silencing' (16.3); 3. 'RNA' (11.9); 4. 'silencing mechanisms' (10.2); 5. 'silencing' (10.0)
- **Q:** Where is the Asian influence strongest in Victoria?  
  **gold:** 'Bendigo' | **baseline:** 'Bendigo' | **gated:** 'Bendigo and its environs' | conf=0.031 | type=where  
  **top-5:** 1. 'Bendigo' (14.2); 2. 'Bendigo and its environs' (14.2); 3. 'in Bendigo' (9.8); 4. 'in Bendigo and its environs' (9.8); 5. 'Bendigo and its environs.' (9.5)
- **Q:** What kind of non-peer-reviewed sources does the IPCC use?  
  **gold:** 'model results, reports from government agencies and non-governmental organizations, and industry journals' | **baseline:** 'model results, reports from government agencies and non-governmental organizations, and industry journals' | **gated:** 'grey literature' | conf=0.110 | type=what  
  **top-5:** 1. 'model results, reports from government agencies and non-governmental organizations, and industry journals' (7.5); 2. 'grey literature' (7.2); 3. 'the "grey literature' (6.2); 4. '"grey literature' (5.7); 5. 'industry journals' (5.5)
- **Q:** Who can decide on legal matters in Islam?  
  **gold:** 'ulemas' | **baseline:** 'ulemas' | **gated:** 'teachers' | conf=0.056 | type=who  
  **top-5:** 1. 'ulemas' (8.1); 2. 'teachers in Islam, ranging from mullahs (the teachers at madrassas) to ulemas' (7.9); 3. 'teachers' (7.2); 4. 'mullahs (the teachers at madrassas) to ulemas' (6.7); 5. 'teachers in Islam, ranging from mullahs' (5.7)
- **Q:** How many biomolecules contain no oxygen?  
  **gold:** 'Only a few' | **baseline:** 'a few' | **gated:** 'Only a few common complex biomolecules' | conf=0.031 | type=how many  
  **top-5:** 1. 'a few' (5.0); 2. 'Only a few' (4.9); 3. 'a few common complex biomolecules' (4.2); 4. 'Only a few common complex biomolecules' (4.1); 5. 'few' (2.4)
- **Q:** What term corresponds to the maximum measurement of time across all functions of n?   
  **gold:** 'worst-case time complexity' | **baseline:** 'worst-case time complexity' | **gated:** 'worst-case time complexity T(n)' | conf=0.109 | type=what  
  **top-5:** 1. 'worst-case time complexity' (10.0); 2. 'worst-case time complexity T(n)' (9.7); 3. 'the worst-case time complexity' (9.3); 4. 'T(n)' (9.1); 5. 'the worst-case time complexity T(n)' (8.9)
- **Q:** Why was the Rhine regulated?   
  **gold:** 'constant flooding' | **baseline:** 'to counteract the constant flooding and strong sedimentation in the western Rhine Delta' | **gated:** 'in order to counteract the constant flooding and strong sedimentation in the western Rhine Delta' | conf=0.029 | type=why  
  **top-5:** 1. 'to counteract the constant flooding and strong sedimentation in the western Rhine Delta' (12.2); 2. 'in order to counteract the constant flooding and strong sedimentation in the western Rhine Delta' (12.2); 3. 'to counteract the constant flooding and strong sedimentation' (11.6); 4. 'in order to counteract the constant flooding and strong sedimentation' (11.5); 5. 'to counteract the constant flooding' (10.5)
- **Q:** What happens to the lead fusible plugs if the water level of the boiler drops?  
  **gold:** 'melts' | **baseline:** 'the lead melts' | **gated:** 'the lead melts and the steam escapes' | conf=0.022 | type=what  
  **top-5:** 1. 'the lead melts' (12.6); 2. 'the lead melts and the steam escapes' (12.5); 3. 'lead melts' (11.4); 4. 'the lead melts and the steam escapes, warning the operators, who may then manually suppress the fire' (11.3); 5. 'lead melts and the steam escapes' (11.3)
- **Q:** What do bundle sheath chloroplasts specialize in?  
  **gold:** 'to carry out the Calvin cycle and make sugar' | **baseline:** 'to carry out the Calvin cycle and make sugar' | **gated:** 'each stage of photosynthesis' | conf=0.037 | type=what  
  **top-5:** 1. 'to carry out the Calvin cycle and make sugar' (8.5); 2. 'carry out the Calvin cycle and make sugar' (8.4); 3. 'each stage of photosynthesis' (8.2); 4. 'make sugar' (7.9); 5. 'photosynthesis' (7.8)
- **Q:** How do you determine the acceleration of a rope when two people are pulling it?  
  **gold:** 'Associating forces with vectors' | **baseline:** 'it is impossible' | **gated:** 'impossible' | conf=0.065 | type=when  
  **top-5:** 1. 'it is impossible' (2.8); 2. 'you do not know which direction either person is pulling, it is impossible' (2.6); 3. 'impossible' (2.3); 4. 'you do not know which direction either person is pulling' (1.8); 5. 'you do not know which direction' (0.9)

### Confidently wrong (not gated, gold in top-k) (122 total)
- **Q:** Where does the Lek join?   
  **gold:** 'Noord River' | **baseline:** 'Wijk bij Duurstede' | **gated:** 'Wijk bij Duurstede' | conf=0.565 | type=where  
  **top-5:** 1. 'Wijk bij Duurstede' (8.2); 2. 'at Wijk bij Duurstede' (6.6); 3. 'Noord River' (5.8); 4. 'the Noord River' (5.4); 5. 'Noord River into the Nieuwe Maas' (5.0)
- **Q:** When was the last season that this original theme was used?  
  **gold:** '17' | **baseline:** '1979–80' | **gated:** '1979–80' | conf=0.522 | type=when  
  **top-5:** 1. '1979–80' (13.8); 2. '17 (1979–80' (12.0); 3. '1979–80).' (12.0); 4. 'season 17 (1979–80' (11.8); 5. '17' (11.3)
- **Q:** What is Cultural Imperialism often referred to as?  
  **gold:** 'soft power' | **baseline:** "when a country's influence is felt in social and cultural circles" | **gated:** "when a country's influence is felt in social and cultural circles" | conf=0.530 | type=what  
  **top-5:** 1. "when a country's influence is felt in social and cultural circles" (8.2); 2. "when a country's influence is felt in social and cultural circles, i.e. its soft power" (6.9); 3. "a country's influence is felt in social and cultural circles" (5.6); 4. 'its soft power' (5.0); 5. 'soft power' (4.9)
- **Q:** What is the expression used to denote a worst case complexity as expressed by time taken?  
  **gold:** 'O(n2)' | **baseline:** 'quicksort' | **gated:** 'quicksort' | conf=0.252 | type=what  
  **top-5:** 1. 'quicksort' (5.1); 2. 'O(n log n).' (4.4); 3. 'O(n2)' (3.7); 4. 'O(n2' (3.2); 5. 'O(n log n' (2.7)
- **Q:** Which direction did the disease first move in?  
  **gold:** 'northwest across Europe' | **baseline:** 'east' | **gated:** 'east' | conf=0.508 | type=which  
  **top-5:** 1. 'east' (8.8); 2. 'northwest across Europe, striking France, Spain, Portugal and England by June 1348, then turned and spread east' (7.5); 3. 'northwest' (6.1); 4. 'turned and spread east' (4.9); 5. 'spread east' (4.4)
- **Q:** What country was under the control of Norman barons?  
  **gold:** 'Wales' | **baseline:** 'the Marches' | **gated:** 'the Marches' | conf=0.617 | type=what  
  **top-5:** 1. 'the Marches' (5.4); 2. 'Marches' (3.9); 3. 'Wales' (1.6); 4. "the Marches came completely under the dominance of William's most trusted Norman barons" (1.2); 5. 'Subsequent to the Conquest, however, the Marches' (0.3)
- **Q:** What do red algal chloroplasts have that green chloroplasts don't?  
  **gold:** 'phycobilisomes' | **baseline:** 'chlorophyll b' | **gated:** 'chlorophyll b' | conf=0.491 | type=what  
  **top-5:** 1. 'chlorophyll b' (15.1); 2. 'chlorophyll b instead' (13.7); 3. 'phycobilisomes, and contain chlorophyll b' (13.6); 4. 'phycobilisomes' (12.6); 5. 'phycobilisomes, and contain chlorophyll b instead' (12.1)
- **Q:** What company had a contest to win a free Super Bowl commercial?  
  **gold:** 'QuickBooks.' | **baseline:** 'Death Wish Coffee' | **gated:** 'Death Wish Coffee' | conf=0.347 | type=what  
  **top-5:** 1. 'Death Wish Coffee' (13.5); 2. 'QuickBooks sponsored a "Small Business Big Game" contest, in which Death Wish Coffee' (12.7); 3. 'QuickBooks' (11.4); 4. 'Death Wish Coffee had a 30-second commercial aired free of charge courtesy of QuickBooks' (9.8); 5. 'Death Wish Coffee had a 30-second commercial aired free of charge courtesy of QuickBooks. Death Wish Coffee' (9.2)
- **Q:** What is the translation of the old north branch of rhe Rhine?  
  **gold:** 'Bent Rhine' | **baseline:** 'Kromme Rijn' | **gated:** 'Kromme Rijn' | conf=0.199 | type=what  
  **top-5:** 1. 'Kromme Rijn' (12.4); 2. 'Kromme Rijn ("Bent Rhine")' (12.0); 3. 'Kromme Rijn ("Bent Rhine' (10.6); 4. 'Bent Rhine")' (9.6); 5. 'Kromme Rijn ("Bent Rhine") past Utrecht' (8.9)
- **Q:** What type of protest falls under civil disobedience without aggression?  
  **gold:** 'nonviolent civil disobedience' | **baseline:** 'lawful' | **gated:** 'lawful' | conf=0.233 | type=what  
  **top-5:** 1. 'lawful' (12.8); 2. 'lawful protest demonstration, nonviolent' (12.0); 3. 'nonviolent' (11.8); 4. 'lawful protest demonstration' (11.0); 5. 'lawful protest' (11.0)

### Gold missing from the n-best list (97 total)
- **Q:** Why was old oil withdrawn from the market?  
  **gold:** 'Price controls' | **baseline:** 'creating greater scarcity' | **gated:** 'creating greater scarcity' | conf=0.517 | type=why  
  **top-5:** 1. 'creating greater scarcity' (13.2); 2. 'greater scarcity' (11.9); 3. 'creating greater scarcity. The rule also discouraged development of alternative energies' (10.5); 4. ', creating greater scarcity' (9.8); 5. 'greater scarcity. The rule also discouraged development of alternative energies' (9.2)
- **Q:** Who was the opposing party in the Runyon case?  
  **gold:** 'McCrary' | **baseline:** 'Wisconsin' | **gated:** 'Meyer v. Nebraska' | conf=0.044 | type=who  
  **top-5:** 1. 'Wisconsin' (-2.0); 2. 'Wisconsin v. Yoder' (-2.2); 3. 'Meyer v. Nebraska' (-2.6); 4. 'Wisconsin v. Yoder, 406 U.S. 205 (1972); Pierce v. Society of Sisters' (-2.9); 5. 'Pierce v. Society of Sisters, 268 U.S. 510 (1925); Meyer v. Nebraska' (-3.2)
- **Q:** What group of people cannot be part of civil disobedience?  
  **gold:** 'sovereign branches of government' | **baseline:** 'citizen' | **gated:** 'citizen' | conf=0.227 | type=what  
  **top-5:** 1. 'citizen' (5.0); 2. 'private citizen' (4.4); 3. 'a citizen' (3.8); 4. "citizen's" (2.2); 5. 'public official rather than private citizen' (1.6)
- **Q:** What is a commonly used measurement used to determine the complexity of a computational problem?  
  **gold:** 'how much time the best algorithm requires to solve the problem' | **baseline:** 'the size of the input in bits' | **gated:** 'the size of the input in bits' | conf=0.078 | type=what  
  **top-5:** 1. 'the size of the input in bits' (11.8); 2. 'size of the input in bits' (11.6); 3. 'size of the instance. This is usually taken to be the size of the input in bits' (10.8); 4. 'input in bits' (10.1); 5. 'the size of the instance. This is usually taken to be the size of the input in bits' (9.9)
- **Q:** Strang is an Anglo-Saxon word which means what in modern English?  
  **gold:** 'strong' | **baseline:** 'deed' | **gated:** 'deed' | conf=0.232 | type=what  
  **top-5:** 1. 'deed' (0.5); 2. 'dead", "cow", "house" and "strong" are pronounced "deed' (-0.3); 3. 'deed", "coo", "hoos' (-0.4); 4. 'dead' (-0.8); 5. '"dead", "cow", "house" and "strong" are pronounced "deed' (-1.0)
- **Q:** What protection comes after the innate response?  
  **gold:** 'the adaptive immune system' | **baseline:** 'an immunological memory' | **gated:** 'an immunological memory' | conf=0.115 | type=what  
  **top-5:** 1. 'an immunological memory' (8.4); 2. 'immunological memory' (8.1); 3. 'after the pathogen has been eliminated, in the form of an immunological memory' (6.8); 4. 'in the form of an immunological memory' (5.8); 5. 'pathogen has been eliminated, in the form of an immunological memory' (5.8)
- **Q:** What was Warner Sinback  
  **gold:** "a data network based on this voice-phone network was designed to connect GE's four computer sales and service centers" | **baseline:** 'high-level marketing manager' | **gated:** 'a high-level marketing manager' | conf=0.044 | type=what  
  **top-5:** 1. 'high-level marketing manager' (12.7); 2. 'a high-level marketing manager' (12.6); 3. 'marketing manager' (10.5); 4. 'Sinback, a high-level marketing manager' (7.8); 5. 'level marketing manager' (7.2)
- **Q:** What language is spoken in Kenya?  
  **gold:** "Kenya's various ethnic groups typically speak their mother tongues within their own communities" | **baseline:** 'Swahili' | **gated:** 'Swahili' | conf=0.245 | type=what  
  **top-5:** 1. 'Swahili' (3.5); 2. 'English and Swahili' (2.9); 3. 'Swahili, are used in varying degrees of fluency for communication with other populations. English' (1.7); 4. 'English and Swahili, are used in varying degrees of fluency for communication with other populations. English' (1.1); 5. 'English' (0.6)
- **Q:** How did the 2001 IPCC report compare to reality for 2001-2006?  
  **gold:** 'temperatures and sea levels have been rising at or above the maximum rates' | **baseline:** "near the top end of the range given by IPCC's 2001 projection" | **gated:** 'near the top end of the range' | conf=0.018 | type=how  
  **top-5:** 1. "near the top end of the range given by IPCC's 2001 projection" (5.7); 2. 'near the top end of the range' (5.7); 3. 'above the top of the range of the IPCC projection' (5.1); 4. "the actual temperature rise was near the top end of the range given by IPCC's 2001 projection" (4.9); 5. 'the actual sea level rise was above the top of the range of the IPCC projection' (4.9)
- **Q:** Which French kind issued this declaration?  
  **gold:** 'Louis XIV' | **baseline:** 'Huguenots' | **gated:** 'Huguenots' | conf=0.636 | type=which  
  **top-5:** 1. 'Huguenots' (4.2); 2. 'Protestantism' (2.1); 3. 'the Huguenots' (2.0); 4. 'Protestantism in France, and forced the Huguenots' (1.8); 5. 'Huguenots' (1.0)

