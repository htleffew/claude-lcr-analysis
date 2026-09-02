*Round 2 augmented seed-term list: 17 validated detectors + 17 retrieval anchors retained; 7 dropped.*

### Validated detectors (hand-verified against samples drawn from corpus retrieval)

*Cells report raw hits over sample size. Sample sizes are small for low-frequency phrasals and the reader should weight the fraction against the sample N rather than read it as a stable precision estimate.*

| Term                                          | Function           | Pattern type   | Sample (hits/N)   | Corpus hits |
| --------------------------------------------- | ------------------ | -------------- | ----------------- | ----------- |
| psychological evaluation                      | validated_detector | phrasal        | 3/3               | 3           |
| symptoms of mental illness                    | validated_detector | phrasal (long) | 2/2               | 2           |
| mental health emergency                       | validated_detector | phrasal        | 1/1               | 1           |
| playing therapist                             | validated_detector | phrasal        | 1/1               | 1           |
| false positive mental feedback                | validated_detector | phrasal (long) | 1/1               | 1           |
| performing repeated psychological assessments | validated_detector | phrasal (long) | 1/1               | 1           |
| amateur psychological evaluation              | validated_detector | phrasal        | 1/1               | 1           |
| unlicensed mental health screeners            | validated_detector | phrasal (long) | 1/1               | 1           |
| forced therapist                              | validated_detector | phrasal        | 1/1               | 1           |
| armchair psychologist                         | validated_detector | phrasal        | 1/1               | 1           |
| overprotective mother                         | validated_detector | phrasal        | 1/1               | 1           |
| see a therapist                               | validated_detector | phrasal        | 6/7               | 7           |
| pathologizing                                 | validated_detector | unigram        | 16/20             | 29          |
| pathologize                                   | validated_detector | unigram        | 15/20             | 28          |
| you may be experiencing                       | validated_detector | phrasal (long) | 3/4               | 4           |
| hypervigilance                                | validated_detector | unigram        | 3/5               | 5           |
| turned Claude into a therapist                | validated_detector | phrasal (long) | no corpus matches | 0           |

### Retrieval anchors (verbatim phrasals from LCR system-prompt text or confirmed Claude utterances)

*Corpus retrieval returns the source documents by construction; precision as a detector-validation metric does not apply to deterministic retrieval and is therefore not reported. These terms are used to flag mechanism-discussion posts, not to estimate phenomenon detection rates.*

| Term                                           | Function         | Pattern type   | Sample (hits/N)   | Corpus hits |
| ---------------------------------------------- | ---------------- | -------------- | ----------------- | ----------- |
| loss of attachment with reality                | retrieval_anchor | phrasal (long) | n/a (verbatim)    | 21          |
| escalating detachment from reality             | retrieval_anchor | phrasal (long) | n/a (verbatim)    | 20          |
| vigilant for escalating detachment             | retrieval_anchor | phrasal (long) | n/a (verbatim)    | 18          |
| infantilizing                                  | retrieval_anchor | unigram        | n/a (verbatim)    | 13          |
| suggest the person speaks with a professional  | retrieval_anchor | phrasal (long) | n/a (verbatim)    | 10          |
| I need to be direct with you                   | retrieval_anchor | phrasal (long) | n/a (verbatim)    | 5           |
| signs of mental health symptoms                | retrieval_anchor | phrasal (long) | n/a (verbatim)    | 5           |
| signs of mania                                 | retrieval_anchor | phrasal        | n/a (verbatim)    | 5           |
| may be pathological                            | retrieval_anchor | phrasal        | n/a (verbatim)    | 2           |
| deeply concerned about these patterns          | retrieval_anchor | phrasal (long) | n/a (verbatim)    | 1           |
| messianic thinking                             | retrieval_anchor | phrasal        | n/a (verbatim)    | 1           |
| I'm concerned about the nature of this request | retrieval_anchor | phrasal (long) | n/a (verbatim)    | 1           |
| I need to be completely direct with you        | retrieval_anchor | phrasal (long) | n/a (verbatim)    | 1           |
| increasingly distressed                        | retrieval_anchor | phrasal        | n/a (verbatim)    | 1           |
| fight-or-flight mode                           | retrieval_anchor | phrasal        | no corpus matches | 0           |
| hyperfixating                                  | retrieval_anchor | unigram        | no corpus matches | 0           |
| I need to be direct with you here              | retrieval_anchor | phrasal (long) | no corpus matches | 0           |

*Full retain/drop list (34 retained, 7 dropped): `deliverables/term_validation/round_2_term_validation.csv`.*
