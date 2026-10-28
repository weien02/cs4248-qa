# Error analysis: `random` on `dev.jsonl`
10570 questions, k=5, prob_margin tau=0.1675, 18.6% sent to the reranker.

## Where errors come from (% of all questions)
| error_type | baseline | gated | rerank_all |
|---|---|---|---|
| correct | 85.79 | 82.90 | 57.43 |
| boundary | 10.23 | 12.94 | 37.48 |
| ranking | 1.79 | 2.12 | 3.44 |
| recall_k | 1.28 | 1.14 | 0.83 |
| recall_all | 0.92 | 0.89 | 0.81 |
`ranking` is what a reranker can fix; `recall_*` needs a better baseline or larger k.

## Does the confidence gate catch the right questions?
| baseline_outcome | n | gated | pct_gated | gated_and_correct_after |
|---|---|---|---|---|
| correct | 9068 | 1270 | 14.0 | 786 |
| boundary | 1081 | 544 | 50.3 | 158 |
| ranking | 189 | 67 | 35.4 | 21 |
| recall_k | 135 | 43 | 31.9 | 0 |
| recall_all | 97 | 40 | 41.2 | 0 |
Ideally `ranking` errors have a high `pct_gated` and `correct` a low one.

## By question type
| question_type | n | EM_baseline | EM_gated | EM_rerank_all | EM_oracle_gated | wrong_to_right | right_to_wrong | pct_gated |
|---|---|---|---|---|---|---|---|---|
| how | 398 | 70.60 | 65.08 | 42.71 | 80.65 | 13 | 35 | 32.4 |
| how many | 642 | 90.19 | 86.45 | 38.63 | 93.61 | 6 | 30 | 10.1 |
| how much | 154 | 85.71 | 82.47 | 51.30 | 92.86 | 6 | 11 | 16.9 |
| other | 292 | 80.48 | 78.08 | 60.96 | 86.64 | 5 | 12 | 18.2 |
| what | 5895 | 85.34 | 82.34 | 58.13 | 90.31 | 97 | 274 | 20.1 |
| when | 695 | 93.38 | 93.24 | 72.37 | 96.55 | 9 | 10 | 7.1 |
| where | 460 | 79.57 | 74.13 | 51.52 | 86.09 | 11 | 36 | 27.0 |
| which | 708 | 88.28 | 86.30 | 62.15 | 92.09 | 11 | 25 | 16.4 |
| who | 1175 | 92.00 | 89.11 | 60.17 | 94.21 | 8 | 42 | 12.8 |
| why | 151 | 58.94 | 61.59 | 53.64 | 78.15 | 13 | 9 | 43.7 |

## By gold answer length (words)
| answer_words | n | EM_baseline | EM_gated | EM_rerank_all | EM_oracle_gated | wrong_to_right | right_to_wrong | pct_gated |
|---|---|---|---|---|---|---|---|---|
| 1 | 4923 | 88.60 | 85.42 | 54.70 | 92.50 | 56 | 213 | 14.8 |
| 2 | 2709 | 85.83 | 82.76 | 59.43 | 90.11 | 39 | 122 | 18.4 |
| 3 | 1382 | 84.15 | 81.33 | 61.94 | 90.23 | 27 | 66 | 21.9 |
| 4-5 | 964 | 81.85 | 79.15 | 59.96 | 87.97 | 25 | 51 | 25.4 |
| 6+ | 592 | 72.47 | 72.47 | 56.25 | 83.78 | 32 | 32 | 32.3 |

## By baseline confidence decile (1 = least confident)
| confidence_decile | n | EM_baseline | EM_gated | EM_rerank_all | EM_oracle_gated | wrong_to_right | right_to_wrong | pct_gated |
|---|---|---|---|---|---|---|---|---|
| 1 | 1057 | 61.12 | 48.25 | 48.25 | 89.69 | 106 | 242 | 100.0 |
| 2 | 1057 | 68.97 | 52.98 | 51.09 | 89.40 | 73 | 242 | 85.8 |
| 3 | 1057 | 77.67 | 77.67 | 53.45 | 77.67 | 0 | 0 | 0.0 |
| 4 | 1057 | 82.02 | 82.02 | 57.33 | 82.02 | 0 | 0 | 0.0 |
| 5 | 1057 | 85.24 | 85.24 | 53.74 | 85.24 | 0 | 0 | 0.0 |
| 6 | 1057 | 90.54 | 90.54 | 54.30 | 90.54 | 0 | 0 | 0.0 |
| 7 | 1057 | 95.65 | 95.65 | 57.71 | 95.65 | 0 | 0 | 0.0 |
| 8 | 1057 | 97.63 | 97.63 | 59.13 | 97.63 | 0 | 0 | 0.0 |
| 9 | 1057 | 99.15 | 99.15 | 68.21 | 99.15 | 0 | 0 | 0.0 |
| 10 | 1057 | 99.91 | 99.91 | 71.05 | 99.91 | 0 | 0 | 0.0 |

## What do the reranker's flips swap? (gated system)
| flip | n | wrong_side | near_miss | unrelated | pct_near_miss |
|---|---|---|---|---|---|
| wrong_to_right | 179 | baseline | 158 | 21 | 88.3 |
| right_to_wrong | 484 | gated | 439 | 45 | 90.7 |
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
### Fixed by the reranker (wrong -> right) (179 total)
- **Q:** Which two Mongol leaders added Persia to the Mongol Empire?  
  **gold:** 'Subutai and Jebe' | **baseline:** "Genghis Khan's grandson Batu and the Golden Horde" | **gated:** 'Subutai and Jebe' | conf=0.076 | type=which  
  **top-5:** 1. "Genghis Khan's grandson Batu and the Golden Horde" (5.0); 2. 'Genghis Khan' (4.8); 3. "Genghis Khan's grandson Batu" (3.4); 4. 'Batu and the Golden Horde' (3.3); 5. 'Subutai and Jebe' (3.0)
- **Q:** What was the outcome of living in the California Redwood tree?  
  **gold:** 'successfully preventing it from being cut down' | **baseline:** 'cut down' | **gated:** 'successfully preventing it from being cut down' | conf=0.046 | type=what  
  **top-5:** 1. 'cut down' (10.6); 2. 'successfully preventing it from being cut down' (10.5); 3. 'preventing it from being cut down' (10.2); 4. 'being cut down' (7.2); 5. 'cut down.' (7.0)
- **Q:** What high maintenance part did Tesla's AC motor not require?  
  **gold:** 'mechanical brushes' | **baseline:** 'sparking and the high maintenance of constantly servicing and replacing mechanical brushes' | **gated:** 'a commutator' | conf=0.029 | type=what  
  **top-5:** 1. 'sparking and the high maintenance of constantly servicing and replacing mechanical brushes' (12.6); 2. 'sparking' (12.5); 3. 'mechanical brushes' (12.1); 4. 'a commutator' (11.4); 5. 'commutator' (11.4)
- **Q:** Which two courts apply European Union law?  
  **gold:** 'courts of member states and the Court of Justice of the European Union' | **baseline:** 'member states and the Court of Justice of the European Union' | **gated:** 'the courts of member states and the Court of Justice of the European Union' | conf=0.046 | type=which  
  **top-5:** 1. 'member states and the Court of Justice of the European Union' (15.1); 2. 'courts of member states and the Court of Justice of the European Union' (15.0); 3. 'the courts of member states and the Court of Justice of the European Union' (14.2); 4. 'member states and the Court of Justice of the European Union.' (13.0); 5. 'courts of member states and the Court of Justice of the European Union.' (12.9)
- **Q:** What do C4 plants need?  
  **gold:** 'more ATP than NADPH' | **baseline:** 'more ATP' | **gated:** 'more ATP than NADPH' | conf=0.148 | type=what  
  **top-5:** 1. 'more ATP' (8.0); 2. 'more ATP than NADPH' (7.6); 3. 'ATP' (6.9); 4. 'ATP than NADPH' (6.5); 5. 'ATP' (4.7)
- **Q:** What report had the correct date?  
  **gold:** '"Variations of Snow and Ice in the past and at present on a Global and Regional Scale"' | **baseline:** 'WWF' | **gated:** 'WWF report' | conf=0.105 | type=what  
  **top-5:** 1. 'WWF' (13.0); 2. 'WWF report' (12.8); 3. 'the WWF' (11.3); 4. 'the WWF report' (11.0); 5. '2035 has been correctly quoted by the IPCC from the WWF' (9.5)
- **Q:** Who led the attack of the French colony in 1565?  
  **gold:** 'Pedro Menéndez de Avilés' | **baseline:** 'Spanish force' | **gated:** 'Pedro Menéndez de Avilés' | conf=0.029 | type=who  
  **top-5:** 1. 'Spanish force' (6.1); 2. 'Pedro Menéndez de Avilés' (6.0); 3. 'a Spanish force' (5.9); 4. 'Philip II of Spain ordered Pedro Menéndez de Avilés' (5.3); 5. 'Spanish' (5.2)
- **Q:** How many were killed by plague in Naples in 1656?  
  **gold:** "about half of Naples' 300,000 inhabitants" | **baseline:** "about half of Naples' 300,000" | **gated:** "about half of Naples' 300,000 inhabitants" | conf=0.127 | type=how many  
  **top-5:** 1. "about half of Naples' 300,000" (11.4); 2. '300,000' (10.9); 3. "half of Naples' 300,000" (10.9); 4. "about half of Naples' 300,000 inhabitants" (10.8); 5. '300,000 inhabitants' (10.3)
- **Q:** Why has the corruption not be in the public view?  
  **gold:** 'Because the operations of the armed forces have been traditionally cloaked by the ubiquitous blanket of “state security”' | **baseline:** 'Because the operations of the armed forces have been traditionally cloaked by the ubiquitous blanket of “state security' | **gated:** 'Because the operations of the armed forces have been traditionally cloaked by the ubiquitous blanket of “state security”' | conf=0.000 | type=why  
  **top-5:** 1. 'Because the operations of the armed forces have been traditionally cloaked by the ubiquitous blanket of “state security' (12.6); 2. 'the operations of the armed forces have been traditionally cloaked by the ubiquitous blanket of “state security' (12.6); 3. 'Because the operations of the armed forces have been traditionally cloaked by the ubiquitous blanket of “state security”' (11.9); 4. 'the operations of the armed forces have been traditionally cloaked by the ubiquitous blanket of “state security”' (11.9); 5. 'operations of the armed forces have been traditionally cloaked by the ubiquitous blanket of “state security' (11.4)
- **Q:** Which channel did the 5 applications request for broadcast?  
  **gold:** 'VHF channel 7' | **baseline:** '7' | **gated:** 'VHF channel 7' | conf=0.121 | type=which  
  **top-5:** 1. '7' (13.8); 2. 'VHF channel 7' (13.6); 3. '7,' (9.7); 4. 'channel 7' (9.6); 5. 'VHF channel 7,' (9.5)

### Broken by the reranker (right -> wrong) (484 total)
- **Q:** How did some suspect that Polo learned about China instead of by actually visiting it?  
  **gold:** 'through contact with Persian traders' | **baseline:** 'through contact with Persian traders' | **gated:** 'contact with Persian traders since many of the places he named were in Persian' | conf=0.135 | type=how  
  **top-5:** 1. 'through contact with Persian traders' (12.1); 2. 'contact with Persian traders' (11.8); 3. 'Persian traders' (10.0); 4. 'through contact with Persian traders since many of the places he named were in Persian' (9.4); 5. 'contact with Persian traders since many of the places he named were in Persian' (9.1)
- **Q:** What is the basic unit of territorial division in Poland?  
  **gold:** 'a commune' | **baseline:** 'a commune' | **gated:** 'a commune (gmina' | conf=0.156 | type=what  
  **top-5:** 1. 'a commune' (13.6); 2. 'a commune (gmina).' (13.1); 3. 'commune' (12.3); 4. 'a commune (gmina' (12.3); 5. 'commune (gmina).' (11.9)
- **Q:** How were leaders back in Europe feeling about news from Celeron expedition?  
  **gold:** 'proposing that action be taken' | **baseline:** 'each side proposing that action be taken' | **gated:** 'reports on the situation in the Ohio Country were making their way to London and Paris, each side proposing that action be taken' | conf=0.054 | type=how  
  **top-5:** 1. 'each side proposing that action be taken' (8.2); 2. 'proposing that action be taken' (8.0); 3. 'each side proposing that action be taken.' (7.2); 4. 'proposing that action be taken.' (7.0); 5. 'reports on the situation in the Ohio Country were making their way to London and Paris, each side proposing that action be taken' (6.3)
- **Q:** How many lines run out of Downtown Los Angeles?  
  **gold:** 'Six' | **baseline:** 'Six' | **gated:** 'seven' | conf=0.066 | type=how many  
  **top-5:** 1. 'Six' (9.8); 2. 'Six of the seven' (9.6); 3. 'seven' (8.6); 4. 'Six of the seven lines' (8.6); 5. 'seven lines' (7.7)
- **Q:** The presence of who is highly likely even in small projects?  
  **gold:** 'the mortgage banker' | **baseline:** 'mortgage banker' | **gated:** 'Mortgage bankers, accountants' | conf=0.117 | type=who  
  **top-5:** 1. 'mortgage banker' (13.0); 2. 'the mortgage banker' (12.8); 3. 'Mortgage bankers' (10.5); 4. 'The presence of the mortgage banker' (9.5); 5. 'Mortgage bankers, accountants' (7.0)
- **Q:** What type of authority do Sudbury schools prefer?  
  **gold:** 'popularly based authority' | **baseline:** 'popularly based' | **gated:** 'dictatorial authority' | conf=0.085 | type=what  
  **top-5:** 1. 'popularly based' (12.6); 2. 'popularly based authority' (12.4); 3. 'popularly based authority can maintain order more effectively than dictatorial authority' (11.4); 4. 'popularly based authority can maintain order more effectively than dictatorial' (10.9); 5. 'dictatorial authority' (8.8)
- **Q:** _____ Helps the biospher from UV.  
  **gold:** 'high-altitude ozone layer' | **baseline:** 'ozone layer' | **gated:** 'ozone (O\n3), strongly absorbs UVB radiation and consequently the high-altitude ozone layer' | conf=0.015 | type=other  
  **top-5:** 1. 'ozone layer' (4.9); 2. 'ozone' (4.9); 3. 'ozone (O\n3), strongly absorbs UVB radiation and consequently the high-altitude ozone layer' (4.8); 4. 'ozone (O\n3), strongly absorbs UVB radiation and consequently the high-altitude ozone' (4.8); 5. 'ozone' (4.0)
- **Q:** What is the delta in the Rhine delimited in the west by?  
  **gold:** 'Alter Rhein' | **baseline:** 'Alter Rhein' | **gated:** 'Alter Rhein ("Old Rhine' | conf=0.000 | type=what  
  **top-5:** 1. 'Alter Rhein' (18.3); 2. 'the Alter Rhein' (18.3); 3. 'Alter Rhein ("Old Rhine")' (17.7); 4. 'the Alter Rhein ("Old Rhine")' (17.7); 5. 'Alter Rhein ("Old Rhine' (15.5)
- **Q:** What did he do with Wardenclyffe?  
  **gold:** 'sold' | **baseline:** 'sold' | **gated:** "sold Wardenclyffe for $20,000 ($472,500 in today's dollars)." | conf=0.005 | type=what  
  **top-5:** 1. 'sold' (9.1); 2. 'sold Wardenclyffe for $20,000' (9.1); 3. 'sold Wardenclyffe' (8.5); 4. "sold Wardenclyffe for $20,000 ($472,500 in today's dollars)." (8.1); 5. 'he sold' (8.0)
- **Q:** What are the biggest game animals of Kenya called?  
  **gold:** 'The "Big Five"' | **baseline:** 'Big Five' | **gated:** 'lion, leopard, buffalo, rhinoceros, and elephant' | conf=0.067 | type=what  
  **top-5:** 1. 'Big Five' (9.6); 2. 'Big Five" game animals of Africa, that is the lion, leopard, buffalo, rhinoceros, and elephant' (9.4); 3. 'lion, leopard, buffalo, rhinoceros, and elephant' (9.1); 4. '"Big Five' (9.0); 5. '"Big Five" game animals of Africa, that is the lion, leopard, buffalo, rhinoceros, and elephant' (8.7)

### Confidently wrong (not gated, gold in top-k) (122 total)
- **Q:** Where does the cold and warm water meet the warm water?  
  **gold:** 'Rheinbrech' | **baseline:** 'near the surface' | **gated:** 'near the surface' | conf=0.479 | type=where  
  **top-5:** 1. 'near the surface' (3.9); 2. "near the surface and at first doesn't mix with the warmer, green waters of Upper Lake" (2.6); 3. 'The cold water flows near the surface' (1.5); 4. 'cold water flows near the surface' (1.3); 5. 'Upper Lake' (0.7)
- **Q:** Which army attacked and destroyed this colony?  
  **gold:** 'Spanish' | **baseline:** 'Pedro Menendez' | **gated:** 'Pedro Menendez' | conf=0.170 | type=which  
  **top-5:** 1. 'Pedro Menendez' (7.3); 2. 'French' (6.8); 3. 'Spanish forces led by Pedro Menendez' (6.4); 4. 'Pedro Menendez. Menendez' (5.7); 5. 'Spanish' (5.1)
- **Q:** Which region invented the machine gun?  
  **gold:** 'European' | **baseline:** 'Southern Africa' | **gated:** 'Southern Africa' | conf=0.550 | type=which  
  **top-5:** 1. 'Southern Africa' (5.4); 2. 'Europe' (3.9); 3. 'Europe also continued to advance in military technology. European' (2.9); 4. 'Zulus in Southern Africa' (2.5); 5. 'European' (2.4)
- **Q:** Rather than taxation, what are private schools largely funded by?  
  **gold:** 'tuition' | **baseline:** 'public (government) funding' | **gated:** 'public (government) funding' | conf=0.419 | type=what  
  **top-5:** 1. 'public (government) funding' (15.2); 2. 'charging their students tuition, rather than relying on mandatory taxation through public (government) funding' (14.0); 3. 'charging their students tuition' (13.2); 4. 'public (government)' (12.8); 5. 'by charging their students tuition, rather than relying on mandatory taxation through public (government) funding' (12.1)
- **Q:** When did Kenya become an independent country?  
  **gold:** '1963' | **baseline:** '12 December 1964' | **gated:** '12 December 1964' | conf=0.246 | type=when  
  **top-5:** 1. '12 December 1964' (6.0); 2. '1963 of the United Kingdom. Exactly 12 months later on 12 December 1964' (5.2); 3. '1963' (5.0); 4. 'Kenya Independence Act 1963 of the United Kingdom. Exactly 12 months later on 12 December 1964' (4.4); 5. '1964' (4.3)
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

### Gold missing from the n-best list (97 total)
- **Q:** What is a commonly used measurement used to determine the complexity of a computational problem?  
  **gold:** 'how much time the best algorithm requires to solve the problem' | **baseline:** 'the size of the input in bits' | **gated:** 'the size of the input in bits' | conf=0.078 | type=what  
  **top-5:** 1. 'the size of the input in bits' (11.8); 2. 'size of the input in bits' (11.6); 3. 'size of the instance. This is usually taken to be the size of the input in bits' (10.8); 4. 'input in bits' (10.1); 5. 'the size of the instance. This is usually taken to be the size of the input in bits' (9.9)
- **Q:** What was dificult to reconcile the photoelectric effect and the missing ultraviolet catastrophe?  
  **gold:** 'electromagnetic theory' | **baseline:** 'troublesome' | **gated:** 'troublesome' | conf=0.284 | type=what  
  **top-5:** 1. 'troublesome' (6.0); 2. 'attempting to reconcile electromagnetic theory with two observations, the photoelectric effect, and the nonexistence of the ultraviolet catastrophe, proved troublesome' (5.2); 3. 'proved troublesome' (5.1); 4. 'electromagnetic theory with two observations, the photoelectric effect, and the nonexistence of the ultraviolet catastrophe, proved troublesome' (3.5); 5. 'troublesome.' (3.0)
- **Q:** What did Astor expect the money be used for?  
  **gold:** 'for Tesla to further develop and produce a new lighting system' | **baseline:** 'fund his Colorado Springs experiments' | **gated:** 'Colorado Springs experiments' | conf=0.159 | type=what  
  **top-5:** 1. 'fund his Colorado Springs experiments' (10.4); 2. 'to fund his Colorado Springs experiments' (10.0); 3. 'Colorado Springs experiments' (9.2); 4. 'his Colorado Springs experiments' (9.0); 5. 'fund his Colorado Springs experiments.' (7.4)
- **Q:** Where is the Rhine?   
  **gold:** 'Europe' | **baseline:** 'Swiss canton of Graubünden in the southeastern Swiss Alps' | **gated:** 'Swiss canton of Graubünden in the southeastern Swiss Alps' | conf=0.000 | type=where  
  **top-5:** 1. 'Swiss canton of Graubünden in the southeastern Swiss Alps' (7.7); 2. 'southeastern Swiss Alps' (7.7); 3. 'the Swiss canton of Graubünden in the southeastern Swiss Alps' (7.3); 4. 'begins in the Swiss canton of Graubünden in the southeastern Swiss Alps' (6.9); 5. 'the southeastern Swiss Alps' (6.8)
- **Q:** Why was old oil withdrawn from the market?  
  **gold:** 'Price controls' | **baseline:** 'creating greater scarcity' | **gated:** 'creating greater scarcity' | conf=0.517 | type=why  
  **top-5:** 1. 'creating greater scarcity' (13.2); 2. 'greater scarcity' (11.9); 3. 'creating greater scarcity. The rule also discouraged development of alternative energies' (10.5); 4. ', creating greater scarcity' (9.8); 5. 'greater scarcity. The rule also discouraged development of alternative energies' (9.2)
- **Q:** Who was the opposing party in the Runyon case?  
  **gold:** 'McCrary' | **baseline:** 'Wisconsin' | **gated:** 'Meyer v. Nebraska' | conf=0.044 | type=who  
  **top-5:** 1. 'Wisconsin' (-2.0); 2. 'Wisconsin v. Yoder' (-2.2); 3. 'Meyer v. Nebraska' (-2.6); 4. 'Wisconsin v. Yoder, 406 U.S. 205 (1972); Pierce v. Society of Sisters' (-2.9); 5. 'Pierce v. Society of Sisters, 268 U.S. 510 (1925); Meyer v. Nebraska' (-3.2)
- **Q:** What group of people cannot be part of civil disobedience?  
  **gold:** 'sovereign branches of government' | **baseline:** 'citizen' | **gated:** 'citizen' | conf=0.227 | type=what  
  **top-5:** 1. 'citizen' (5.0); 2. 'private citizen' (4.4); 3. 'a citizen' (3.8); 4. "citizen's" (2.2); 5. 'public official rather than private citizen' (1.6)
- **Q:** Strang is an Anglo-Saxon word which means what in modern English?  
  **gold:** 'strong' | **baseline:** 'deed' | **gated:** 'deed' | conf=0.232 | type=what  
  **top-5:** 1. 'deed' (0.5); 2. 'dead", "cow", "house" and "strong" are pronounced "deed' (-0.3); 3. 'deed", "coo", "hoos' (-0.4); 4. 'dead' (-0.8); 5. '"dead", "cow", "house" and "strong" are pronounced "deed' (-1.0)
- **Q:** What protection comes after the innate response?  
  **gold:** 'the adaptive immune system' | **baseline:** 'an immunological memory' | **gated:** 'immunological memory' | conf=0.115 | type=what  
  **top-5:** 1. 'an immunological memory' (8.4); 2. 'immunological memory' (8.1); 3. 'after the pathogen has been eliminated, in the form of an immunological memory' (6.8); 4. 'in the form of an immunological memory' (5.8); 5. 'pathogen has been eliminated, in the form of an immunological memory' (5.8)
- **Q:** What was Warner Sinback  
  **gold:** "a data network based on this voice-phone network was designed to connect GE's four computer sales and service centers" | **baseline:** 'high-level marketing manager' | **gated:** 'high-level marketing manager' | conf=0.044 | type=what  
  **top-5:** 1. 'high-level marketing manager' (12.7); 2. 'a high-level marketing manager' (12.6); 3. 'marketing manager' (10.5); 4. 'Sinback, a high-level marketing manager' (7.8); 5. 'level marketing manager' (7.2)

