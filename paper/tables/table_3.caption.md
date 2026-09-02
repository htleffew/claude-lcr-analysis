**Table 3.** Round 2 augmented seed-term list, stratified by retrieval function.
*Validated detectors* (top section) were hand-verified against samples drawn
from corpus retrieval; the cell reports raw hits over sample size, with small
sample sizes for low-frequency phrasals noted so that the reader can weight the
fraction against the sample N rather than read it as a stable precision
estimate. *Retrieval anchors* (bottom section) were lifted verbatim from LCR
system-prompt text or from confirmed Claude utterances in ground-truth positive
cases; corpus retrieval returns the source documents by construction, so
precision as a detector-validation metric does not apply to deterministic
retrieval and is not reported for that section. Retrieval anchors are used to
flag posts that quote or reproduce LCR-mechanism content, not to estimate the
prevalence of the LCR phenomenon itself. Dropped terms had sample-based
precision below 0.50 or were confirmed as generic discourse openers with no
LCR-specific signal. The full retain/drop list is in
`deliverables/term_validation/round_2_term_validation.csv`.
