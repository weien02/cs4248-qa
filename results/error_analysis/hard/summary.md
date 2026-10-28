# Error analysis: `hard` on `dev.jsonl`
10570 questions, k=5, prob_margin tau=0.1675, 18.6% sent to the reranker.

## Where errors come from (% of all questions)
| error_type | baseline | gated | rerank_all |
|---|---|---|---|
| correct | 85.79 | 86.75 | 85.44 |
| boundary | 10.23 | 8.93 | 9.22 |
| ranking | 1.79 | 2.04 | 3.03 |
| recall_k | 1.28 | 1.32 | 1.32 |
| recall_all | 0.92 | 0.96 | 0.99 |
`ranking` is what a reranker can fix; `recall_*` needs a better baseline or larger k.

## Does the confidence gate catch the right questions?
| baseline_outcome | n | gated | pct_gated | gated_and_correct_after |
|---|---|---|---|---|
| correct | 9068 | 1270 | 14.0 | 1098 |
| boundary | 1081 | 544 | 50.3 | 241 |
| ranking | 189 | 67 | 35.4 | 33 |
| recall_k | 135 | 43 | 31.9 | 0 |
| recall_all | 97 | 40 | 41.2 | 0 |
Ideally `ranking` errors have a high `pct_gated` and `correct` a low one.

## By question type
| question_type | n | EM_baseline | EM_gated | EM_rerank_all | EM_oracle_gated | wrong_to_right | right_to_wrong | pct_gated |
|---|---|---|---|---|---|---|---|---|
| how | 398 | 70.60 | 71.36 | 69.35 | 80.65 | 22 | 19 | 32.4 |
| how many | 642 | 90.19 | 90.65 | 88.63 | 93.61 | 9 | 6 | 10.1 |
| how much | 154 | 85.71 | 87.01 | 87.01 | 92.86 | 4 | 2 | 16.9 |
| other | 292 | 80.48 | 82.19 | 83.90 | 86.64 | 9 | 4 | 18.2 |
| what | 5895 | 85.34 | 86.31 | 84.99 | 90.31 | 148 | 91 | 20.1 |
| when | 695 | 93.38 | 94.53 | 94.10 | 96.55 | 13 | 5 | 7.1 |
| where | 460 | 79.57 | 79.35 | 77.39 | 86.09 | 15 | 16 | 27.0 |
| which | 708 | 88.28 | 89.27 | 87.85 | 92.09 | 19 | 12 | 16.4 |
| who | 1175 | 92.00 | 92.17 | 90.55 | 94.21 | 17 | 15 | 12.8 |
| why | 151 | 58.94 | 69.54 | 66.89 | 78.15 | 18 | 2 | 43.7 |

## By gold answer length (words)
| answer_words | n | EM_baseline | EM_gated | EM_rerank_all | EM_oracle_gated | wrong_to_right | right_to_wrong | pct_gated |
|---|---|---|---|---|---|---|---|---|
| 1 | 4923 | 88.60 | 89.36 | 87.61 | 92.50 | 106 | 69 | 14.8 |
| 2 | 2709 | 85.83 | 86.30 | 85.46 | 90.11 | 61 | 48 | 18.4 |
| 3 | 1382 | 84.15 | 84.95 | 84.95 | 90.23 | 37 | 26 | 21.9 |
| 4-5 | 964 | 81.85 | 83.20 | 80.81 | 87.97 | 32 | 19 | 25.4 |
| 6+ | 592 | 72.47 | 77.20 | 76.01 | 83.78 | 38 | 10 | 32.3 |

## By baseline confidence decile (1 = least confident)
| confidence_decile | n | EM_baseline | EM_gated | EM_rerank_all | EM_oracle_gated | wrong_to_right | right_to_wrong | pct_gated |
|---|---|---|---|---|---|---|---|---|
| 1 | 1057 | 61.12 | 68.78 | 68.78 | 89.69 | 170 | 89 | 100.0 |
| 2 | 1057 | 68.97 | 70.96 | 71.43 | 89.40 | 104 | 83 | 85.8 |
| 3 | 1057 | 77.67 | 77.67 | 77.39 | 77.67 | 0 | 0 | 0.0 |
| 4 | 1057 | 82.02 | 82.02 | 80.04 | 82.02 | 0 | 0 | 0.0 |
| 5 | 1057 | 85.24 | 85.24 | 81.36 | 85.24 | 0 | 0 | 0.0 |
| 6 | 1057 | 90.54 | 90.54 | 87.61 | 90.54 | 0 | 0 | 0.0 |
| 7 | 1057 | 95.65 | 95.65 | 93.28 | 95.65 | 0 | 0 | 0.0 |
| 8 | 1057 | 97.63 | 97.63 | 96.03 | 97.63 | 0 | 0 | 0.0 |
| 9 | 1057 | 99.15 | 99.15 | 98.86 | 99.15 | 0 | 0 | 0.0 |
| 10 | 1057 | 99.91 | 99.91 | 99.62 | 99.91 | 0 | 0 | 0.0 |

## What do the reranker's flips swap? (gated system)
| flip | n | wrong_side | near_miss | unrelated | pct_near_miss |
|---|---|---|---|---|---|
| wrong_to_right | 274 | baseline | 241 | 33 | 88.0 |
| right_to_wrong | 172 | gated | 138 | 34 | 80.2 |
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
### Fixed by the reranker (wrong -> right) (274 total)
- **Q:** What type of number do modern mathematicians consider 1 to be?  
  **gold:** 'its own special category as a "unit"' | **baseline:** 'prime' | **gated:** 'unit' | conf=0.045 | type=what  
  **top-5:** 1. 'prime' (5.6); 2. 'a "unit' (5.4); 3. 'unit' (5.4); 4. 'prime. By the Middle Ages and Renaissance many mathematicians included 1 as the first prime' (5.1); 5. 'prime number' (5.0)
- **Q:** A dollar spent by a poor person is likely to provide what for them?  
  **gold:** 'a great deal of utility' | **baseline:** 'utility' | **gated:** 'a great deal of utility' | conf=0.145 | type=what  
  **top-5:** 1. 'utility' (10.4); 2. 'great deal of utility' (9.9); 3. 'a great deal of utility' (9.8); 4. 'utility to that person' (8.8); 5. 'utility to that person, such as basic necessities like food, water, and healthcare' (8.7)
- **Q:** What caused the delay during the development of the AC motor.  
  **gold:** 'financial strain' | **baseline:** 'financial strain of buying up patents and hiring the engineers needed to build it' | **gated:** 'financial strain' | conf=0.044 | type=what  
  **top-5:** 1. 'financial strain of buying up patents and hiring the engineers needed to build it' (11.3); 2. 'financial strain' (11.2); 3. 'the financial strain of buying up patents and hiring the engineers needed to build it' (11.1); 4. 'financial strain of buying up patents and hiring the engineers' (10.9); 5. 'the financial strain' (10.9)
- **Q:** Which famous Dutch artist who painted The Night Watch is represented in the drawings collection of the V&A?  
  **gold:** 'Rembrandt' | **baseline:** 'Dürer' | **gated:** 'Rembrandt' | conf=0.018 | type=which  
  **top-5:** 1. 'Dürer' (-1.7); 2. 'Dürer, Giovanni Benedetto Castiglione, Bernardo Buontalenti, Rembrandt' (-1.7); 3. 'Rembrandt' (-1.8); 4. 'Dürer, Giovanni Benedetto Castiglione, Bernardo Buontalenti, Rembrandt, Antonio Verrio' (-3.1); 5. 'Rembrandt, Antonio Verrio' (-3.2)
- **Q:** Where did Iroquois Confederation control?  
  **gold:** 'present-day Upstate New York and the Ohio Country' | **baseline:** 'much of present-day Upstate New York and the Ohio Country' | **gated:** 'Upstate New York and the Ohio Country' | conf=0.064 | type=where  
  **top-5:** 1. 'much of present-day Upstate New York and the Ohio Country' (9.3); 2. 'present-day Upstate New York and the Ohio Country' (9.2); 3. 'Upstate New York and the Ohio Country' (9.0); 4. 'The Iroquois Confederation dominated much of present-day Upstate New York and the Ohio Country' (7.2); 5. 'much of present-day Upstate New York and the Ohio Country,' (6.3)
- **Q:** Name one way in which a person can become a Professing Member?  
  **gold:** 'transfer from another Christian denomination' | **baseline:** 'through transfer from another Christian denomination' | **gated:** 'transfer from another Christian denomination' | conf=0.079 | type=which  
  **top-5:** 1. 'through transfer from another Christian denomination' (9.6); 2. 'transfer from another Christian denomination' (9.3); 3. 'confirmation and sometimes the profession of faith' (8.7); 4. 'through confirmation and sometimes the profession of faith' (8.7); 5. 'profession of faith' (7.8)
- **Q:** What conjecture holds that there are always a minimum of 4 primes  between the squares of consecutive primes greater than 2?  
  **gold:** "Brocard's conjecture" | **baseline:** "H. Brocard's" | **gated:** "Brocard's conjecture" | conf=0.100 | type=what  
  **top-5:** 1. "H. Brocard's" (14.7); 2. "H. Brocard's conjecture" (14.3); 3. 'H. Brocard' (14.1); 4. "Brocard's" (14.0); 5. "Brocard's conjecture" (13.6)
- **Q:** What did the Jewish people do so pagan items wouldn't be in the temple of Jerusalem?  
  **gold:** 'gathered in the streets' | **baseline:** 'Unarmed Jews gathered in the streets' | **gated:** 'gathered in the streets' | conf=0.000 | type=what  
  **top-5:** 1. 'Unarmed Jews gathered in the streets' (9.0); 2. 'gathered in the streets' (9.0); 3. 'Jews gathered in the streets' (7.5); 4. 'Unarmed Jews gathered' (7.1); 5. 'gathered' (7.1)
- **Q:** When did Germany found their first settlement?  
  **gold:** '1884' | **baseline:** '1883–84' | **gated:** '1884' | conf=0.163 | type=when  
  **top-5:** 1. '1883–84' (8.4); 2. '1883' (8.0); 3. '1884' (6.3); 4. 'in 1883–84' (5.6); 5. 'in 1883' (5.2)
- **Q:** What time do they normally have tea?  
  **gold:** "10 o'clock tea (chai ya saa nne) and 4 pm tea" | **baseline:** '4 pm' | **gated:** "10 o'clock" | conf=0.045 | type=what  
  **top-5:** 1. '4 pm' (9.0); 2. "10 o'clock tea (chai ya saa nne) and 4 pm" (8.9); 3. "10 o'clock" (8.2); 4. '4 pm tea' (7.2); 5. "10 o'clock tea (chai ya saa nne) and 4 pm tea" (7.1)

### Broken by the reranker (right -> wrong) (172 total)
- **Q:** How many years have imperialistic practices existed?  
  **gold:** 'thousands' | **baseline:** 'thousands' | **gated:** 'thousands of years' | conf=0.015 | type=how many  
  **top-5:** 1. 'thousands' (12.7); 2. 'thousands of years' (12.7); 3. 'thousands of' (9.0); 4. 'thousands of years,' (7.9); 5. 'for thousands' (7.8)
- **Q:** What type of steam engines produced most power up to the early 20th century?  
  **gold:** 'Reciprocating piston' | **baseline:** 'Reciprocating piston type' | **gated:** 'Reciprocating piston type steam engines' | conf=0.112 | type=what  
  **top-5:** 1. 'Reciprocating piston type' (16.6); 2. 'Reciprocating piston type steam engines' (16.2); 3. 'Reciprocating piston' (16.1); 4. 'piston type' (14.8); 5. 'piston type steam engines' (14.4)
- **Q:** What does many workers willing to work for a lot of time competing for a job that only requires a few workers result in?  
  **gold:** 'low wage' | **baseline:** 'low wage' | **gated:** 'a low wage for that job' | conf=0.008 | type=what  
  **top-5:** 1. 'low wage' (13.7); 2. 'low wage for that job' (13.7); 3. 'a low wage' (13.7); 4. 'a low wage for that job' (13.6); 5. 'will result in a low wage' (10.2)
- **Q:** What were the protesters doing with Christ's counsel?  
  **gold:** "ignoring Christ's counsel" | **baseline:** 'ignoring' | **gated:** "Render unto Caesar the things that are Caesar's" | conf=0.141 | type=what  
  **top-5:** 1. 'ignoring' (11.8); 2. 'ignoring Christ\'s counsel to "Render unto Caesar the things that are Caesar\'s";' (11.3); 3. 'ignoring Christ\'s counsel to "Render unto Caesar the things that are Caesar\'s' (11.0); 4. 'Render unto Caesar the things that are Caesar\'s";' (10.4); 5. "Render unto Caesar the things that are Caesar's" (10.0)
- **Q:** What was AUSTPAC  
  **gold:** 'AUSTPAC was an Australian public X.25 network operated by Telstra' | **baseline:** 'an Australian public X.25 network operated by Telstra' | **gated:** "Australia's first public packet-switched data network" | conf=0.083 | type=what  
  **top-5:** 1. 'an Australian public X.25 network operated by Telstra' (15.5); 2. "Australia's first public packet-switched data network" (15.2); 3. 'an Australian public X.25 network' (14.2); 4. 'Australian public X.25 network operated by Telstra' (13.7); 5. "AUSTPAC was Australia's first public packet-switched data network" (13.1)
- **Q:** What theorems are responsible for determining questions of time and space requirements?  
  **gold:** 'time and space hierarchy theorems' | **baseline:** 'time and space hierarchy theorems' | **gated:** 'time and space hierarchy' | conf=0.123 | type=what  
  **top-5:** 1. 'time and space hierarchy theorems' (15.4); 2. 'time and space hierarchy theorems respectively. They are called hierarchy theorems' (15.1); 3. 'hierarchy theorems' (14.5); 4. 'time and space hierarchy' (13.8); 5. 'time and space hierarchy theorems respectively. They are called hierarchy' (12.7)
- **Q:** What is the main purpose of the jurisdictions and central conferences?  
  **gold:** 'to elect and appoint bishops' | **baseline:** 'to elect and appoint bishops' | **gated:** 'to elect and appoint bishops, the chief administrators of the church' | conf=0.157 | type=what  
  **top-5:** 1. 'to elect and appoint bishops' (16.4); 2. 'elect and appoint bishops' (15.9); 3. 'to elect and appoint bishops, the chief administrators of the church' (15.8); 4. 'elect and appoint bishops, the chief administrators of the church' (15.3); 5. 'to elect and appoint bishops, the chief administrators of the church.' (12.5)
- **Q:** What happened to the Apollo program in for the rest of 1970 after the incident regarding Apollo 13?  
  **gold:** 'grounded' | **baseline:** 'grounded' | **gated:** 'Apollo was grounded again' | conf=0.074 | type=what  
  **top-5:** 1. 'grounded' (14.1); 2. 'grounded again' (13.8); 3. 'Apollo was grounded' (13.7); 4. 'Apollo was grounded again' (13.4); 5. 'grounded again, for the remainder of 1970 while the oxygen tank was redesigned and an extra one was added' (10.7)
- **Q:** Where are bills typically gestated in Stage 1?  
  **gold:** 'in the relevant committee or committees' | **baseline:** 'in the relevant committee or committees' | **gated:** 'the relevant committee or committees' | conf=0.143 | type=where  
  **top-5:** 1. 'in the relevant committee or committees' (15.5); 2. 'the relevant committee or committees' (15.1); 3. 'relevant committee or committees' (15.0); 4. 'committee or committees' (13.1); 5. 'Stage 1 usually takes place, initially, in the relevant committee or committees' (11.0)
- **Q:** Who is the president of TUMAS?  
  **gold:** 'Rev. Paul T. Stallsworth' | **baseline:** 'Rev. Paul T. Stallsworth' | **gated:** 'Paul T. Stallsworth' | conf=0.093 | type=who  
  **top-5:** 1. 'Rev. Paul T. Stallsworth' (17.2); 2. 'Paul T. Stallsworth' (17.1); 3. 'Rev. Paul T. Stallsworth,' (12.2); 4. 'Paul T. Stallsworth,' (12.0); 5. 'Stallsworth' (10.4)

### Confidently wrong (not gated, gold in top-k) (122 total)
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
- **Q:** What was the black death originally blamed on?  
  **gold:** 'the heavens' | **baseline:** 'bad air' | **gated:** 'bad air' | conf=0.757 | type=what  
  **top-5:** 1. 'bad air' (9.3); 2. 'the heavens' (7.2); 3. 'heavens' (6.1); 4. 'the plague was caused by bad air' (4.8); 5. 'plague was caused by bad air' (4.8)
- **Q:** What is another country that permits physicians to give out drugs from within their practice?  
  **gold:** 'Austria' | **baseline:** 'United Kingdom' | **gated:** 'United Kingdom' | conf=0.426 | type=what  
  **top-5:** 1. 'United Kingdom' (10.8); 2. 'the United Kingdom' (9.7); 3. 'Austria' (9.1); 4. 'United Kingdom,' (6.3); 5. 'the United Kingdom,' (5.2)

### Gold missing from the n-best list (97 total)
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
- **Q:** In Antigone, who was the target of civil disobedience?  
  **gold:** 'Creon, the current King of Thebes' | **baseline:** 'Antigone' | **gated:** 'Polynices' | conf=0.030 | type=who  
  **top-5:** 1. 'Antigone' (6.2); 2. 'her brother Polynices' (6.0); 3. 'Antigone, one of the daughters of former King of Thebes, Oedipus' (5.7); 4. 'Polynices' (5.5); 5. 'Antigone, one of the daughters of former King of Thebes' (5.3)
- **Q:** What Columbia President went to Harvard?  
  **gold:** 'Juan Manuel Santos' | **baseline:** 'Barack Obama' | **gated:** 'Barack Obama' | conf=0.481 | type=what  
  **top-5:** 1. 'Barack Obama' (-3.1); 2. 'Obama' (-4.8); 3. 'Barack Obama; Chilean President Sebastián Piñera; Colombian President Juan Manuel Santos; Costa Rican President José María Figueres' (-4.9); 4. 'Barack Obama; Chilean President Sebastián Piñera' (-4.9); 5. 'Alejandro Toledo' (-5.0)

