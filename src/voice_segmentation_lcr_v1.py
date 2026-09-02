"""
voice_segmentation_lcr_v1.py
Phase 4 — Voice Segmentation  [method §C.4]
LCR (Long Conversation Reminders) Pass 1b canonical corpus

VERSION LINEAGE -- READ FIRST
-----------------------------
This is the FIRST iteration of the LCR voice-segmentation pipeline. It was
superseded by voice_segmentation_lcr_v2.py (row-level confidence-weighted
labels) and then by voice_segmentation_lcr_v3_span.py (the current canonical
pipeline, which emits span-level labels per the methods-library §9.0 schema).

v1 and v2 are retained in source-tree form for reproducibility and provenance:
they were run on the same Pass 1b canonical corpus and their outputs informed
the Phase 4 voice-segmentation failure-modes memo. The PAPER uses v3-span
outputs exclusively. If you are reproducing the published analyses, use
voice_segmentation_lcr_v3_span.py, not this file.

See notebooks/audit_trail/phase_4_voice_segmentation_failure_modes.md for the
documented reason each predecessor was superseded.

Purpose
-------
Separate model-attributed speech from user speech in 26,158 rows of unstructured
Reddit discourse.  Outputs a five-way voice label per row:

    Direct_Quote_Of_Model         — user is quoting the model's words verbatim
    Paraphrase_Of_Model           — user is summarising or paraphrasing model output
    LCR_System_Prompt_Reproduction — verbatim reproduction of the LCR system-prompt text
                                    itself (not model output, but the injected prompt)
    User_Original_Content         — user is speaking in their own voice only
    auto_moderator_content        — AutoModerator boilerplate; excluded from downstream
                                    phenomenological analysis

plus a confidence tag (high / medium / low) that feeds the §C.4 decision rule.

Architecture
------------
Layer 0  — AutoModerator stop-phrase filter: applied first.  Rows matching known
            AutoModerator boilerplate are tagged `auto_moderator_content` and
            bypassed.  AutoModerator dominated trigrams ranks 1-8 in Phase 2 freq.

Layer 0b — LCR system-prompt verbatim recognition: rows containing verbatim
            phrasals from the LCR system-prompt text are tagged
            `LCR_System_Prompt_Reproduction`.  These are model-mechanism evidence
            but system-prompt-attributed, not model-output-attributed.

Layer 1a — Strong-signal regex (§9.1): explicit markers (blockquotes, attribution
            phrases with quotation marks, speaker-label conventions).
            Any match assigns high confidence.

Layer 1b — Paraphrase-signal regex (§9.1 extended): corpus-specific paraphrase
            vocabulary derived from KWIC Phase 2 inspection and the 35 LCR Round 1
            positive cases.  Matches assign medium confidence.
            LCR-specific additions:
              - Third-person model self-attribution in pathologizing contexts
                (Claude said I had, the model claimed I was, it diagnosed me as)
              - User-quotation patterns of model utterances
                (it told me I was manic, Claude said I'm spiraling)
              - Indirect-reported-speech patterns
                (Claude basically said, it kept telling me, it's been suggesting)

Layer 1c — Weak corpus-specific signals (low confidence).

Layer 2  — LLM-assisted fallback (§9.2): targets a STRATIFIED RANDOM SAMPLE of
            the regex-residual rows.  Budget: <= 100 calls total.  The sample is
            drawn before the loop so the budget is allocated across the full
            corpus.  Model: claude-haiku-4-5 via the Anthropic SDK.
            Rows not selected for LLM review default to User_Original_Content/low.

Stratification (Phase 3 decision)
----------------------------------
The Phase 3 unit decision requires segmentation to respect the post / comment
stratum.  `type` is carried through untouched.  Confidence distribution tables
are reported per stratum.  LLM budget is allocated proportionally between
posts (1,173) and comments (24,985).  r/claudexplorers is slightly oversampled
in the validation sample per the Phase 3 observation of elevated mechanism vocabulary.

LCR-specific corpus cleaning
-----------------------------
AutoModerator boilerplate identified from Phase 2 Pass 1b trigram frequency analysis:
ranks 1-8 were all AutoModerator-generated phrases.  These are tagged before any
voice-segmentation logic runs.

LCR system-prompt verbatim phrasals (high-confidence anchors from Round 2
augmented terms):
  - "loss of attachment with reality"
  - "escalating detachment from reality"
  - "vigilant for escalating detachment from reality"
  - "suggest the person speaks with a professional or trusted person for support"
  - "without either sugar coating them or being infantilizing"

LLM call budget tracking
-------------------------
Every LLM call is logged to:
  deliverables/phase_4_voice_segmentation/llm_call_log.jsonl
with fields: row_id, model, prompt_system, prompt_user, raw_response,
             assigned_label, assigned_confidence, timestamp, call_number.
The log is the audit-trail record of the LLM's role (tool, not validator).

LLM model used
--------------
    claude-haiku-4-5  (cheapest Anthropic hosted Haiku; appropriate for
    classification on short texts under single-operator budget)
Design date: 2026-05-17

Usage
-----
    python src/voice_segmentation_lcr_v1.py

Outputs (all relative to repo root)
-------------------------------------
    data/lcr_pass1b_canonical_voice_segmented.csv
    deliverables/phase_4_voice_segmentation/llm_call_log.jsonl
    deliverables/phase_4_voice_segmentation/confidence_distribution.csv

Validation sample (written separately):
    notebooks/audit_trail/phase_4_voice_segmentation_validation_sample.csv
"""

import csv
import json
import re
import random
import datetime
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV  = REPO_ROOT / "data" / "lcr_pass1b_canonical.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "lcr_pass1b_canonical_voice_segmented.csv"
DELIVERABLES_DIR = REPO_ROOT / "deliverables" / "phase_4_voice_segmentation"
LLM_LOG   = DELIVERABLES_DIR / "llm_call_log.jsonl"
CONF_DIST = DELIVERABLES_DIR / "confidence_distribution.csv"

DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

# Clear any previous LLM log before a fresh run
if LLM_LOG.exists():
    LLM_LOG.unlink()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_DIRECT    = "Direct_Quote_Of_Model"
LABEL_PARAPHRASE = "Paraphrase_Of_Model"
LABEL_LCR_SP    = "LCR_System_Prompt_Reproduction"
LABEL_USER      = "User_Original_Content"
LABEL_AUTOMOD   = "auto_moderator_content"

CONF_HIGH = "high"
CONF_MED  = "medium"
CONF_LOW  = "low"

LLM_BUDGET = 100     # hard cap on total fallback calls for the run
LLM_MODEL  = "claude-haiku-4-5"

# ---------------------------------------------------------------------------
# Layer 0 — AutoModerator stop-phrase filter
# ---------------------------------------------------------------------------
# Source: Phase 2 Pass 1b trigram frequency analysis — ranks 1-8 were all
# AutoModerator boilerplate.  These patterns are applied BEFORE any voice
# segmentation logic and tag the row as auto_moderator_content.
#
# AutoModerator phrases use the reddit "I am a bot" convention and explicit
# moderator-action language.  A single match on any of these is sufficient.

_AUTOMOD_PATTERNS = [
    re.compile(r'\bI am a bot\b', re.IGNORECASE),
    re.compile(r'\bbot action performed automatically\b', re.IGNORECASE),
    re.compile(r'\bPlease contact the moderators\b', re.IGNORECASE),
    re.compile(r'\bthis action was performed automatically\b', re.IGNORECASE),
    re.compile(r'\bif you have any questions\b.*\bmoderators\b', re.IGNORECASE | re.DOTALL),
    re.compile(r'\bmod team\b.*\bautomatically\b', re.IGNORECASE | re.DOTALL),
    re.compile(r'\bsubreddit rules\b.*\bbe removed\b', re.IGNORECASE | re.DOTALL),
    re.compile(r'\bAutoModerator\b', re.IGNORECASE),
    # Specific boilerplate phrases observed in LCR corpus (trigram ranks 1-8)
    re.compile(r'\bposts? (?:will )?be (?:automatically )?removed\b', re.IGNORECASE),
    re.compile(r'\byour (?:post|submission) (?:has been|was) (?:removed|flagged)\b',
               re.IGNORECASE),
    re.compile(r'\bPlease read the (?:rules|subreddit rules)\b', re.IGNORECASE),
    re.compile(r'\bThis is a reminder\b.*\brules\b', re.IGNORECASE | re.DOTALL),
]


def _is_automod(text: str) -> bool:
    """Return True if text matches AutoModerator boilerplate patterns."""
    if not text:
        return False
    for pat in _AUTOMOD_PATTERNS:
        if pat.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 0b — LCR system-prompt verbatim recognition
# ---------------------------------------------------------------------------
# Source: Phase 2.5 finding and document-duplication-attractor §1.8 annotation.
# Round 2 augmented terms include verbatim phrasals from the LCR system prompt.
# These are SYSTEM-PROMPT-ATTRIBUTED content, not model-output-attributed.
# Matching any of these high-confidence phrases assigns LCR_System_Prompt_Reproduction.
#
# PC-LCR-17 and PC-LCR-18 are the ground-truth positives for this label.

_LCR_SP_PATTERNS = [
    # Verbatim LCR system-prompt phrases (high-confidence Round 2 anchors)
    re.compile(r'loss of attachment with reality', re.IGNORECASE),
    re.compile(r'escalating detachment from reality', re.IGNORECASE),
    re.compile(r'vigilant for escalating detachment from reality', re.IGNORECASE),
    re.compile(
        r'suggest the person speaks? with a professional or trusted person for support',
        re.IGNORECASE),
    re.compile(r'without either sugar coating them or being infantilizing', re.IGNORECASE),
    # Additional verbatim LCR-SP phrases confirmed from PC-LCR-17, PC-LCR-18, PC-LCR-25
    re.compile(r'watch for mania,?\s+psychosis,?\s+dissociation', re.IGNORECASE),
    re.compile(r'long conversation (?:reminders?|prompt)', re.IGNORECASE),
    re.compile(r'strict professional content generation guidelines', re.IGNORECASE),
    re.compile(r'remains? vigilant for', re.IGNORECASE),
    re.compile(
        r'can suggest the person speaks? with a professional or trusted person',
        re.IGNORECASE),
]


def _is_lcr_system_prompt(text: str) -> bool:
    """Return True if text contains verbatim LCR system-prompt phrasals."""
    if not text:
        return False
    for pat in _LCR_SP_PATTERNS:
        if pat.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 1a — Strong-signal regex (direct quote markers)
# ---------------------------------------------------------------------------
# These fire first (after Layer 0 / 0b); any match returns high confidence.

_DQ_PATTERNS = [
    # Reddit blockquote lines (> at line start)
    (re.compile(r'(?:^|\n)[ \t]*(?:>|&gt;)\s*\S', re.M),
     LABEL_DIRECT, CONF_HIGH, "blockquote_marker"),

    # Explicit attribution + open quotation mark
    # e.g.  Claude said "I'm concerned about your mental state"
    (re.compile(
        r'(?:claude|the\s+model|it|the\s+ai|the\s+assistant|claude\s+code)\s+'
        r'(?:said|wrote|replied|responded|told\s+me|told\s+us|stated|added)\s*'
        r'(?:something\s+like\s*)?["""''""]',
        re.IGNORECASE),
     LABEL_DIRECT, CONF_HIGH, "attribution_phrase_open_quote"),

    # Speaker label on its own line: "Claude:" or "AI:" or "Model:"
    (re.compile(r'(?im)^(?:claude|model|ai|assistant)\s*:\s*\S'),
     LABEL_DIRECT, CONF_HIGH, "standalone_speaker_label"),

    # Inline colon-style attribution: "Claude: 'I'm concerned...'"
    (re.compile(r'(?i)claude\s*:[""""]'),
     LABEL_DIRECT, CONF_HIGH, "speaker_label_colon_quote"),

    # Quoted block with prior attribution (5–300 chars in quotes)
    # Wider than sleep (200) because LCR model utterances tend to be longer
    (re.compile(
        r'(?:claude|the\s+model|the\s+ai|it)\s+'
        r'(?:said|told\s+me|wrote|replied|responded)\s+'
        r'[""""][^""""]{5,300}[""""]',
        re.IGNORECASE),
     LABEL_DIRECT, CONF_HIGH, "attribution_phrase_quoted_block"),
]

# ---------------------------------------------------------------------------
# Layer 1b — Paraphrase-signal regex (corpus-specific vocabulary)
# ---------------------------------------------------------------------------
# Derived from:
#   - KWIC Phase 2 Pass 1b notes (therapist, professional, pathologizing, spiraling seeds)
#   - 35 LCR Round 1 positive cases (PC-LCR-01 through PC-LCR-35)
#   - LCR-specific attribution patterns: pathologizing-context model self-attribution,
#     user-quotation patterns of model utterances, indirect reported speech
#
# Confidence is HIGH for patterns with specific corpus attestation from Round 1 positives,
# MED for more general attribution-verb patterns.

_PM_PATTERNS = [
    # -----------------------------------------------------------------------
    # LCR-specific: third-person model self-attribution in pathologizing contexts
    # e.g. "Claude said I had SERIOUS ANXIETY" / "it diagnosed me as delusional"
    # Ground truth: PC-LCR-01, PC-LCR-02, PC-LCR-03, PC-LCR-31
    # -----------------------------------------------------------------------
    (re.compile(
        r'(?:claude|the\s+model|the\s+ai|it|the\s+assistant)\s+'
        r'(?:said|claimed|told\s+me|told\s+us|diagnosed\s+me\s+(?:as|with)|'
        r'accused\s+me\s+of|called\s+me|labeled\s+me|declared\s+(?:I|that\s+I)|'
        r'determined\s+(?:I|that\s+I)|decided\s+(?:I|that\s+I)|'
        r'insisted\s+(?:I|that\s+I))\s+'
        r'(?:I|I\'m|I\s+was|I\s+had|I\s+am|I\s+have|my\b)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_model_self_attribution_pathologizing"),

    # -----------------------------------------------------------------------
    # LCR-specific: user-quotation patterns of model utterances with
    # mental-health content
    # e.g. "it told me I was manic" / "Claude said I'm spiraling" / "it accused me of being in denial"
    # Ground truth: PC-LCR-04, PC-LCR-05, PC-LCR-09, PC-LCR-31
    # -----------------------------------------------------------------------
    (re.compile(
        r'(?:it|claude|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:told\s+me\s+I|said\s+I(?:\'m|\s+was|\s+had|\s+am|\s+have|\'ve)|'
        r'accused\s+me\s+of|said\s+I\s+was\s+(?:in|having|showing|displaying|exhibiting))',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_user_quotation_model_utterance"),

    # -----------------------------------------------------------------------
    # LCR-specific: indirect reported speech patterns
    # e.g. "Claude basically said I was manic" / "it kept telling me I was spiraling"
    # "it's been suggesting I get help" / "it basically told me I was having a breakdown"
    # Ground truth: PC-LCR-10, PC-LCR-16, PC-LCR-29
    # -----------------------------------------------------------------------
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:basically|essentially|kind\s+of|sort\s+of|practically|literally|'
        r'all\s+but|pretty\s+much)\s+'
        r'(?:said|told\s+me|told\s+us|suggested|implied|stated|informed\s+me|'
        r'accused\s+me|called\s+me)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_basically_said"),

    (re.compile(
        r'(?:it|claude|the\s+model|the\s+ai)\s+'
        r'(?:kept|keeps|kept\s+on|keeps\s+on|has\s+been|\'s\s+been|'
        r'has\s+been|been)\s+'
        r'(?:telling\s+me|saying|suggesting\s+(?:I|that\s+I)|'
        r'insisting|reminding\s+me|questioning\s+(?:my|me)|'
        r'asking\s+(?:me|if\s+I))',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_kept_telling_me_indirect"),

    # -----------------------------------------------------------------------
    # Standard attribution patterns (adapted from sleep template)
    # -----------------------------------------------------------------------

    # "my claude tells/told me" — very specific corpus pattern
    (re.compile(r'(?i)my\s+claude\s+(?:tells?|told)\s+me', re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "my_claude_tells_me"),

    # "mine is/keeps telling me"
    (re.compile(r'(?i)mine\s+(?:is\s+)?(?:telling|told|keeps?\s+telling)', re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "mine_is_telling"),

    # Third-person model self-attribution (generic, not pathologizing-specific)
    # "Claude said it needs to comply" / "it claimed it was correct"
    (re.compile(
        r'(?:claude|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:said|claimed|told\s+me|insisted|suggested)\s+'
        r'(?:it|she|he)\s+'
        r'(?:needs?|needed|was|is|had\s+to|wanted|couldn\'t?|would)\b',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "third_person_self_attribution"),

    # Directive attribution patterns with mental-health vocabulary
    # e.g. "Claude told me to see a therapist" / "it said I should seek professional help"
    # Ground truth: PC-LCR-06, PC-LCR-07, PC-LCR-08, PC-LCR-16
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:told\s+me\s+to|told\s+us\s+to|said\s+I\s+should|'
        r'suggested\s+I\s+(?:see|speak|talk|get|seek|reach)|'
        r'recommended\s+(?:I|that\s+I)|urged\s+me\s+to|'
        r'pushed\s+(?:me|for)\s+(?:me\s+to)?)\s+'
        r'(?:see|seek|get|speak\s+with|talk\s+to|reach\s+out\s+to|'
        r'consult|contact)\s+'
        r'(?:a\s+)?(?:therapist|professional|psychiatrist|psychologist|'
        r'counselor|doctor|mental\s+health)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_directive_help_attribution"),

    # "it refused to continue" / "claude refused to help" patterns
    # Ground truth: PC-LCR-01, PC-LCR-02, PC-LCR-07
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:refused\s+to|wouldn\'t|would\s+not|couldn\'t|could\s+not|'
        r'stopped|declined\s+to|flat.?out\s+refused)\s+'
        r'(?:continue|help|proceed|engage|respond|assist)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_refusal_attribution"),

    # "Claude diagnosed me" / "it pathologized" / "it flagged my" patterns
    # Ground truth: PC-LCR-03, PC-LCR-15, PC-LCR-34
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:diagnosed|pathologized|flagged|identified|detected|assessed|'
        r'evaluated|classified|labeled|tagged|marked)\s+'
        r'(?:me|my|it|the\s+content|this\s+as|that\s+as)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_diagnostic_attribution"),

    # "Claude expressed concern" / "it was concerned" patterns
    # Ground truth: PC-LCR-01 through PC-LCR-14 (concern framing is pervasive)
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:expressed|showed|displayed|said\s+it\s+was|claimed\s+to\s+be|'
        r'seemed|appeared)\s+'
        r'(?:concerned|worried|alarmed|troubled)\s+'
        r'(?:about|by|that|for)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_concern_expression_attribution"),

    # "Claude basically said" / hedged paraphrase marker (generic)
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:basically|essentially|kind\s+of|sort\s+of|practically|literally)\s+'
        r'(?:said|told\s+me|told\s+us|suggested|implied|stated|informed\s+me)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "basically_said"),

    # Attribution-verb directive (general)
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant|claude\s+code)\s+'
        r'(?:told\s+me\s+to|told\s+us\s+to|told\s+me\s+that|told\s+us\s+that|'
        r'said\s+(?:I|we|to)\s+should|said\s+to|'
        r'suggested\s+(?:I|we)\s+|suggested\s+that\s+(?:I|we)\s+|'
        r'insisted\s+(?:I|we)\s+|insisted\s+that|'
        r'asked\s+(?:me|us)\s+to|'
        r'decided\s+(?:it\s+was|that\s+it\s+was|to)|'
        r'kept\s+saying|'
        r'would\s+(?:say|tell\s+me|ask)\s+)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "attribution_verb_directive"),

    # "claude is like" / "claude was like" (Reddit reported-speech idiom)
    (re.compile(
        r'(?:claude|the\s+model|the\s+ai)\s+(?:is|was|were)\s+like\b',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "is_like_reported_speech"),

    # "and then it said" / narrative reconstruction
    (re.compile(
        r'(?:and\s+then|then|after\s+that|at\s+which\s+point|so|whereupon)\s+'
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:said|told|replied|responded|wrote|suggested|asked)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "narrative_reconstruction"),

    # Model judgment attribution
    (re.compile(
        r'(?:it|claude|the\s+model)\s+decides?\s+(?:I\'?m|I\s+was|that)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "model_judgment_attribution"),

    # "claude would" + model-behavior verb (repeated behavior paraphrase)
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:would|will|does|doesn\'t?|won\'t|wouldn\'t)\s+'
        r'(?:say|tell\s+me|suggest|insist|refuse|push|encourage|prompt|ask|'
        r'pathologize|flag|diagnose|accuse|claim)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "would_say"),

    # Keeps/keeps on suggesting (iterative help-directive — very common in LCR corpus)
    # "Claude keeps suggesting talking to a mental health professional"
    # Ground truth: PC-LCR-16, PC-LCR-26, PC-LCR-29
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:keeps?|kept|has\s+been|\'s\s+been|constantly|always|'
        r'repeatedly|again\s+and\s+again)\s+'
        r'(?:suggesting|telling\s+me|asking\s+me|recommending|'
        r'reminding\s+me|pushing|insisting)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_keeps_suggesting"),

    # "I asked claude to X and it Y" pattern
    (re.compile(
        r'(?:I\s+asked\s+(?:claude|it|the\s+model))\s+.{0,80}'
        r'(?:and\s+(?:it|claude|the\s+model))\s+'
        r'(?:said|told|replied|responded|wrote|suggested|refused|accused|diagnosed)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "i_asked_and_it_said"),

    # "it started to" / "it began" + pathologizing behavior pattern
    # Ground truth: PC-LCR-13, PC-LCR-29
    (re.compile(
        r'(?:it|claude|the\s+model|the\s+ai)\s+'
        r'(?:started\s+to|began\s+to|suddenly\s+started|started\s+asking|'
        r'out\s+of\s+nowhere\s+(?:said|started|asked))\s+'
        r'(?:suggest|tell|ask|question|pathologize|accuse|diagnose|flag)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "lcr_suddenly_started"),

    # Repeated-attribution adverbs
    (re.compile(
        r'(?:constantly|keeps?|kept|always|again|repeatedly|'
        r'every\s+(?:time|session|message|chat))\s+'
        r'(?:telling|saying|asking|suggesting|reminding|interrupting|'
        r'flagging|warning|bringing\s+it\s+up)\s+'
        r'(?:me|us|you)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "repeated_attribution_adverb"),

    # "claude telling its users to" (meta-report phrasing)
    (re.compile(
        r'claude\s+telling\s+(?:its\s+)?users?\s+to',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "claude_telling_users"),

    # The "Flip" — community vocabulary for the LCR behavioral shift
    # Ground truth: PC-LCR-32
    (re.compile(
        r'(?:the\s+flip\b|flips?\s+(?:to|into|from)|flipped\s+(?:to|into|from)|'
        r'psych\s+mode|flips?\s+(?:on|off)\s+(?:the\s+)?(?:lcr|guardrails|safety))',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "lcr_flip_vocabulary"),

    # "straight up told me / tell users" (community phrasing for blunt model output)
    # Ground truth: PC-LCR-19, PC-LCR-33
    (re.compile(
        r'(?:straight\s+up\s+|flat-?out\s+|outright\s+)(?:told|said|told\s+me|accused)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "lcr_straight_up_told"),
]

# ---------------------------------------------------------------------------
# Layer 1c — Weak corpus-specific signals (low confidence)
# ---------------------------------------------------------------------------
_WEAK_PM_PATTERNS = [
    # Present-tense habitual attribution
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|claude\s+code)\s+'
        r'(?:always|constantly|often|sometimes|never|still|just|randomly|'
        r'suddenly|out\s+of\s+nowhere)\s+'
        r'(?:tries?\s+to|tells?\s+me\s+to|wants?\s+to|says?\s+to|asks?\s+me\s+to|'
        r'suggests?\s+|insists?\s+|refuses?\s+to|flags?|diagnoses?|pathologizes?)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_LOW, "weak_habitual_attribution"),

    # "tells me/you to" without leading attribution marker (bare verb)
    (re.compile(
        r'(?:tells?\s+(?:me|you|us)|told\s+(?:me|you|us))\s+'
        r'(?:to\s+(?:see|seek|get|speak|consult|reach)\s+'
        r'(?:a\s+)?(?:therapist|professional|psychiatrist|psychologist|doctor|help)|'
        r'that\s+I(?:\'m|\s+was|\s+had|\s+am)\s+'
        r'(?:spiraling|manic|delusional|paranoid|anxious|concerning|in\s+crisis))',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_LOW, "weak_tells_me_pathologizing"),

    # "it's like it's trying to" — indirect model-behavior attribution
    (re.compile(
        r'(?:it\'s|its|seems)\s+like\s+(?:it\'s|claude\'s|the\s+model\s+is)\s+'
        r'(?:trying\s+to|attempting\s+to|designed\s+to)\s+'
        r'(?:pathologize|flag|diagnose|suggest|push)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_LOW, "weak_indirect_model_behavior"),

    # "Reconstructed dialogue without attribution tag"
    (re.compile(
        r'(?:claude|it|the\s+model)\s+(?:is|was)\s+like\s+[""""]',
        re.IGNORECASE),
     LABEL_DIRECT, CONF_LOW, "weak_is_like_with_quote"),

    # Community abbreviations and reaction vocabulary implying model attribution
    (re.compile(
        r'\b(?:lcr|long\s+conversation\s+reminder)\b.*'
        r'(?:told|said|flagged|diagnosed|accused|pathologized|kicked\s+in)',
        re.IGNORECASE | re.DOTALL),
     LABEL_PARAPHRASE, CONF_LOW, "weak_lcr_community_abbreviation"),
]


# ---------------------------------------------------------------------------
# Utility: apply Layer 1 patterns
# ---------------------------------------------------------------------------

def _apply_layer1(text: str) -> Tuple[str, str, str]:
    """
    Returns (label, confidence, pattern_note).
    Returns ("ambiguous", "low", "no_match") if no pattern fires.

    Layer order:
      1a — strong-signal direct-quote markers (high confidence)
      1b — paraphrase-signal regex (high/medium confidence)
      1c — weak corpus-specific signals (low confidence; better than default)
    """
    # Layer 1a: strong direct-quote signals
    for pattern, label, conf, note in _DQ_PATTERNS:
        if pattern.search(text):
            return label, conf, note

    # Layer 1b: paraphrase signals
    for pattern, label, conf, note in _PM_PATTERNS:
        if pattern.search(text):
            return label, conf, note

    # Layer 1c: weak signals (low confidence)
    for pattern, label, conf, note in _WEAK_PM_PATTERNS:
        if pattern.search(text):
            return label, conf, note

    return "ambiguous", CONF_LOW, "no_match"


# ---------------------------------------------------------------------------
# Layer 2 — LLM-assisted fallback
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = (
    "You are a discourse analyst classifying Reddit post or comment text from "
    "communities discussing Claude (Anthropic's AI assistant). The phenomenon "
    "under study is Long Conversation Reminders (LCR): a system-prompt behavior "
    "in Claude that causes it to issue unsolicited mental-health-related "
    "directives and psychiatric attributions to users.\n\n"
    "You must pick exactly one label and one confidence level. "
    "Do not explain. Output valid JSON only, no markdown.\n\n"
    "Labels:\n"
    "  Direct_Quote_Of_Model — the author appears to be quoting Claude's "
    "exact words verbatim (with or without quotation marks, including "
    "reconstructed dialogue and paraphrased-but-close reproductions).\n"
    "  Paraphrase_Of_Model — the author is summarising or describing what "
    "Claude said or did, without verbatim quotation. Third-person attributions "
    "count (e.g. 'Claude told me I was spiraling'; 'it diagnosed me as delusional'; "
    "'it kept suggesting I see a therapist').\n"
    "  LCR_System_Prompt_Reproduction — the text reproduces verbatim phrasals "
    "from the LCR system-prompt itself (e.g. 'loss of attachment with reality', "
    "'escalating detachment from reality', 'suggest the person speaks with a "
    "professional or trusted person for support', 'long conversation reminders', "
    "'watch for mania, psychosis, dissociation'). This is system-prompt-attributed, "
    "not model-output-attributed.\n"
    "  User_Original_Content — the text is entirely in the user's own voice; "
    "no model speech is attributed, and it is not a system-prompt reproduction.\n\n"
    "Confidence levels:\n"
    "  high — clear signal, little room for doubt.\n"
    "  medium — plausible but some ambiguity remains.\n"
    "  low — genuinely ambiguous; the label is a best guess.\n\n"
    'Output format (JSON only): {"label": "<label>", "confidence": "<confidence>"}'
)

LLM_USER_TEMPLATE = (
    "Classify this Reddit text. Is the author quoting or paraphrasing Claude's "
    "output, reproducing the LCR system-prompt text, or is this the user's own "
    "voice?\n\n---\n{text}\n---\n\nJSON only."
)

llm_call_count = 0


def _call_llm(text: str, row_id: str) -> Tuple[str, str]:
    """
    Call claude-haiku-4-5 fallback.  Returns (label, confidence).
    Logs every call.  On any error returns (User_Original_Content, low).
    """
    global llm_call_count
    raw = ""
    try:
        import anthropic
        client = anthropic.Anthropic()
        user_text = text[:2000]
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=64,
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": LLM_USER_TEMPLATE.format(text=user_text)}],
        )
        raw = message.content[0].text.strip()
        parsed = json.loads(raw)
        label = parsed.get("label", LABEL_USER)
        confidence = parsed.get("confidence", CONF_LOW)
        valid_labels = {LABEL_DIRECT, LABEL_PARAPHRASE, LABEL_LCR_SP, LABEL_USER}
        if label not in valid_labels:
            label = LABEL_USER
            confidence = CONF_LOW
        if confidence not in {CONF_HIGH, CONF_MED, CONF_LOW}:
            confidence = CONF_LOW
    except Exception as e:
        label = LABEL_USER
        confidence = CONF_LOW
        raw = f"ERROR: {e}"

    llm_call_count += 1
    log_entry = {
        "row_id": row_id,
        "model": LLM_MODEL,
        "prompt_system": LLM_SYSTEM_PROMPT,
        "prompt_user": LLM_USER_TEMPLATE.format(text=(text or "")[:500]),
        "raw_response": raw if isinstance(raw, str) else json.dumps(raw),
        "assigned_label": label,
        "assigned_confidence": confidence,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "call_number": llm_call_count,
    }
    with open(LLM_LOG, "a", encoding="utf-8") as flog:
        flog.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return label, confidence


# ---------------------------------------------------------------------------
# Main segmenter
# ---------------------------------------------------------------------------

def run_segmentation(input_path: Path, output_path: Path) -> List[Dict]:
    """
    Multi-pass segmentation:
      Layer 0  — AutoModerator filter (bypass tag; no voice segmentation)
      Layer 0b — LCR system-prompt verbatim recognition (bypass tag)
      Pass 1   — Layer 1 regex on remaining rows.  Record which are ambiguous.
      Pass 2   — LLM fallback on a STRATIFIED RANDOM SAMPLE of ambiguous rows
                 (proportional post/comment allocation, budget <= 100).
                 Remaining ambiguous rows default to User_Original_Content/low.
      Write output CSV with voice_label, voice_confidence, voice_source columns.
    """
    with open(input_path, newline="", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    n = len(rows)
    print(f"Loaded {n} rows from {input_path.name}")

    # --- Layer 0: AutoModerator filter ---
    automod_indices: set = set()
    lcr_sp_indices: set = set()
    remaining_indices: list = []

    for i, row in enumerate(rows):
        text = row.get("body", "") or ""
        if _is_automod(text):
            automod_indices.add(i)
        elif _is_lcr_system_prompt(text):
            lcr_sp_indices.add(i)
        else:
            remaining_indices.append(i)

    print(f"  Layer 0  (AutoModerator):          {len(automod_indices):6} rows tagged")
    print(f"  Layer 0b (LCR system-prompt):      {len(lcr_sp_indices):6} rows tagged")
    print(f"  Remaining for voice segmentation:  {len(remaining_indices):6} rows")

    # --- Pass 1: regex on remaining rows ---
    results: Dict[int, Tuple[str, str, str, str]] = {}  # index -> (label, conf, note, source)

    # Pre-assign Layer 0 / 0b labels
    for i in automod_indices:
        results[i] = (LABEL_AUTOMOD, CONF_HIGH, "automod_pattern", "layer0_automod")
    for i in lcr_sp_indices:
        results[i] = (LABEL_LCR_SP, CONF_HIGH, "lcr_sp_pattern", "layer0b_lcr_sp")

    ambiguous_indices: list = []
    for i in remaining_indices:
        row = rows[i]
        text = row.get("body", "") or ""
        label, conf, note = _apply_layer1(text)
        if label == "ambiguous":
            ambiguous_indices.append(i)
            results[i] = (LABEL_USER, CONF_LOW, note, "regex_default")
        else:
            results[i] = (label, conf, note, "regex")

    n_tagged_l0 = len(automod_indices) + len(lcr_sp_indices)
    n_ambig = len(ambiguous_indices)
    n_regex_hit = len(remaining_indices) - n_ambig
    print(f"  Regex coverage (of remaining): {n_regex_hit}/{len(remaining_indices)} rows "
          f"({100*n_regex_hit/max(len(remaining_indices),1):.1f}%)")
    print(f"  Ambiguous (regex miss): {n_ambig} rows")
    print(f"  LLM budget: {LLM_BUDGET} calls")

    # --- Stratified sample of ambiguous rows for LLM ---
    if n_ambig > 0 and LLM_BUDGET > 0:
        ambig_posts = [i for i in ambiguous_indices if rows[i]["type"] == "post"]
        ambig_comments = [i for i in ambiguous_indices if rows[i]["type"] == "comment"]
        prop_posts = len(ambig_posts) / n_ambig if n_ambig else 0
        n_posts_llm = max(1, round(LLM_BUDGET * prop_posts)) if ambig_posts else 0
        n_comments_llm = LLM_BUDGET - n_posts_llm
        if n_posts_llm > len(ambig_posts):
            n_posts_llm = len(ambig_posts)
            n_comments_llm = min(LLM_BUDGET - n_posts_llm, len(ambig_comments))
        if n_comments_llm > len(ambig_comments):
            n_comments_llm = len(ambig_comments)
            n_posts_llm = min(LLM_BUDGET - n_comments_llm, len(ambig_posts))

        random.seed(42)
        sampled_posts = random.sample(ambig_posts, n_posts_llm) if n_posts_llm else []
        sampled_comments = random.sample(ambig_comments, n_comments_llm) if n_comments_llm else []
        llm_indices = set(sampled_posts + sampled_comments)
        print(f"  LLM sample: {len(llm_indices)} rows "
              f"({n_posts_llm} posts, {n_comments_llm} comments)")
    else:
        llm_indices = set()

    # --- Pass 2: LLM on sample ---
    for i in sorted(llm_indices):
        row = rows[i]
        text = row.get("body", "") or ""
        row_id = row.get("post_id", "") + "|" + row.get("comment_id", "")
        label, conf = _call_llm(text, row_id)
        # LLM can return LCR_SP label — validate it stays in 4-way (not automod)
        if label == LABEL_AUTOMOD:
            label = LABEL_USER
        results[i] = (label, conf, "llm_fallback", "llm")

    # --- Write output ---
    out_fieldnames = fieldnames + ["voice_label", "voice_confidence", "voice_source"]
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            label, conf, note, source = results[i]
            row["voice_label"] = label
            row["voice_confidence"] = conf
            row["voice_source"] = source
            writer.writerow(row)

    print(f"\nSegmentation complete.  LLM calls used: {llm_call_count}")
    print(f"Output: {output_path}")

    with open(output_path, newline="", encoding="utf-8") as f:
        rows_out = list(csv.DictReader(f))
    return rows_out


def compute_confidence_distribution(rows_out: List[Dict], output_path: Path):
    """Stratified confidence distribution table (type x voice_label x confidence)."""
    counts: Counter = Counter()
    for row in rows_out:
        key = (row["type"], row["voice_label"], row["voice_confidence"])
        counts[key] += 1

    total = len(rows_out)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "voice_label", "voice_confidence",
                         "n", "pct_of_total"])
        for (stratum, label, conf), n in sorted(counts.items()):
            writer.writerow([stratum, label, conf, n,
                             round(100 * n / total, 2)])

    print("\n--- Confidence distribution (stratified) ---")
    conf_totals: Dict = defaultdict(int)
    label_totals: Dict = defaultdict(int)
    type_totals: Dict = defaultdict(int)

    for row in rows_out:
        conf_totals[row["voice_confidence"]] += 1
        label_totals[row["voice_label"]] += 1
        type_totals[row["type"]] += 1

    print("  By confidence:")
    for conf in [CONF_HIGH, CONF_MED, CONF_LOW]:
        n = conf_totals[conf]
        print(f"    {conf:6}: {n:6}/{total} ({100*n/total:5.1f}%)")
    high_med = conf_totals[CONF_HIGH] + conf_totals[CONF_MED]
    print(f"    high+medium (reliably segmentable): "
          f"{high_med}/{total} ({100*high_med/total:.1f}%)")

    print("  By voice label:")
    all_labels = [LABEL_DIRECT, LABEL_PARAPHRASE, LABEL_LCR_SP,
                  LABEL_USER, LABEL_AUTOMOD]
    for label in all_labels:
        n = label_totals[label]
        print(f"    {label}: {n}/{total} ({100*n/total:.1f}%)")

    print("  By stratum:")
    for stratum in ["post", "comment"]:
        n = type_totals[stratum]
        print(f"    {stratum}: {n} rows")

    # Per-stratum breakdown of voice label (excluding automod from percentage base
    # because automod rows are typically comments; do not distort post analysis)
    for stratum in ["post", "comment"]:
        stratum_rows = [r for r in rows_out if r["type"] == stratum]
        n_stratum = len(stratum_rows)
        print(f"  Per-stratum voice label ({stratum}s, n={n_stratum}):")
        for label in all_labels:
            n = sum(1 for r in stratum_rows if r["voice_label"] == label)
            print(f"    {label}: {n}/{n_stratum} ({100*n/max(n_stratum,1):.1f}%)")

    return conf_totals, label_totals, high_med, total


def prepare_validation_sample(rows_out: List[Dict], output_path: Path,
                               seed: int = 42):
    """
    Stratified hand-validation sample (§9.3).  70 items total:
      15 Direct_Quote_Of_Model
      15 Paraphrase_Of_Model
      10 LCR_System_Prompt_Reproduction
      15 User_Original_Content
      10 auto_moderator_content
       5 low confidence (any non-automod, non-lcr-sp label; de-duplicated)

    Each class is stratified by type (post/comment) where possible.
    r/claudexplorers is slightly oversampled within the paraphrase and DQ strata
    per Phase 3 finding of elevated mechanism vocabulary.

    Columns:
      row_id, post_id, type, subreddit, retrieval_provenance, r2_any_match,
      full_text, predicted_voice_label, predicted_confidence,
      researcher_label, researcher_notes
    """
    random.seed(seed)

    def stratum_sample_with_claudexplorers_boost(pool, n, boost_subreddit="claudexplorers"):
        """Sample n from pool, stratifying post/comment and boosting claudexplorers."""
        if not pool or n <= 0:
            return []
        # Separate claudexplorers
        ce_pool = [r for r in pool if r.get("subreddit") == boost_subreddit]
        other_pool = [r for r in pool if r.get("subreddit") != boost_subreddit]

        # Claudexplorers gets floor(n/3) or all available, whichever is smaller
        n_ce = min(len(ce_pool), max(1, n // 3))
        n_other = n - n_ce
        if n_other > len(other_pool):
            n_other = len(other_pool)
            n_ce = min(n - n_other, len(ce_pool))

        chosen = []
        if n_ce > 0:
            chosen += random.sample(ce_pool, n_ce)
        if n_other > 0:
            chosen += random.sample(other_pool, n_other)
        return chosen

    def stratum_sample(pool, n):
        """Simple stratified sample by post/comment."""
        if not pool or n <= 0:
            return []
        posts = [r for r in pool if r["type"] == "post"]
        comments = [r for r in pool if r["type"] == "comment"]
        if not posts:
            return random.sample(comments, min(n, len(comments)))
        if not comments:
            return random.sample(posts, min(n, len(posts)))
        n_posts = max(1, round(n * len(posts) / len(pool)))
        n_comments = n - n_posts
        if n_posts > len(posts):
            n_posts = len(posts)
            n_comments = min(n - n_posts, len(comments))
        if n_comments > len(comments):
            n_comments = len(comments)
            n_posts = min(n - n_comments, len(posts))
        return random.sample(posts, n_posts) + random.sample(comments, n_comments)

    # Partition rows by label (exclude automod from downstream voice strata but
    # still sample it for validation)
    by_label = {lbl: [] for lbl in [LABEL_DIRECT, LABEL_PARAPHRASE, LABEL_LCR_SP,
                                     LABEL_USER, LABEL_AUTOMOD]}
    low_conf = []  # any non-automod, non-lcr-sp low-confidence row
    for row in rows_out:
        lbl = row.get("voice_label")
        conf = row.get("voice_confidence")
        if lbl in by_label:
            by_label[lbl].append(row)
        if (conf == CONF_LOW and
                lbl not in {LABEL_AUTOMOD, LABEL_LCR_SP}):
            low_conf.append(row)

    sampled_ids: set = set()
    sample_rows = []

    def add_pool(pool, n, use_ce_boost=False):
        if use_ce_boost:
            chosen = stratum_sample_with_claudexplorers_boost(pool, n)
        else:
            chosen = stratum_sample(pool, n)
        for r in chosen:
            rid = r["post_id"] + "|" + r.get("comment_id", "")
            if rid not in sampled_ids:
                sampled_ids.add(rid)
                sample_rows.append(r)

    # Core strata — use claudexplorers boost for DQ and Paraphrase
    add_pool(by_label[LABEL_DIRECT],    15, use_ce_boost=True)
    add_pool(by_label[LABEL_PARAPHRASE], 15, use_ce_boost=True)
    add_pool(by_label[LABEL_LCR_SP],    10)
    add_pool(by_label[LABEL_USER],      15)
    add_pool(by_label[LABEL_AUTOMOD],   10)
    # Low-confidence stratum (any label except automod/lcr-sp, not already sampled)
    remaining_low = [r for r in low_conf
                     if (r["post_id"] + "|" + r.get("comment_id", "")) not in sampled_ids]
    add_pool(remaining_low, 5)

    sample_rows.sort(key=lambda r: (r["voice_label"], r["type"]))

    out_fieldnames = [
        "row_id", "post_id", "type", "subreddit", "retrieval_provenance",
        "r2_any_match", "full_text",
        "predicted_voice_label", "predicted_confidence",
        "researcher_label", "researcher_notes",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in sample_rows:
            rid = row["post_id"] + "|" + row.get("comment_id", "")
            body = (row.get("body", "") or "").replace("\n", " | ")
            writer.writerow({
                "row_id": rid,
                "post_id": row["post_id"],
                "type": row["type"],
                "subreddit": row.get("subreddit", ""),
                "retrieval_provenance": row.get("retrieval_provenance", ""),
                "r2_any_match": row.get("r2_any_match", ""),
                "full_text": body,
                "predicted_voice_label": row["voice_label"],
                "predicted_confidence": row["voice_confidence"],
                "researcher_label": "",
                "researcher_notes": "",
            })

    tally = Counter(r["voice_label"] for r in sample_rows)
    n_lc = sum(1 for r in sample_rows if r["voice_confidence"] == CONF_LOW)
    print(f"\nValidation sample: {len(sample_rows)} items")
    print(f"  Direct_Quote:              {tally.get(LABEL_DIRECT, 0):3}")
    print(f"  Paraphrase_Of_Model:       {tally.get(LABEL_PARAPHRASE, 0):3}")
    print(f"  LCR_System_Prompt_Repr:    {tally.get(LABEL_LCR_SP, 0):3}")
    print(f"  User_Original_Content:     {tally.get(LABEL_USER, 0):3}")
    print(f"  auto_moderator_content:    {tally.get(LABEL_AUTOMOD, 0):3}")
    print(f"  Low-confidence items:      {n_lc:3}")
    print(f"  Written to: {output_path.name}")
    return sample_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows_out = run_segmentation(INPUT_CSV, OUTPUT_CSV)
    conf_totals, label_totals, high_med, total = \
        compute_confidence_distribution(rows_out, CONF_DIST)

    validation_path = (REPO_ROOT / "notebooks" / "audit_trail" /
                       "phase_4_voice_segmentation_validation_sample.csv")
    prepare_validation_sample(rows_out, validation_path)

    print("\nPhase 4 LCR voice segmentation complete.")
    print(f"  Segmented corpus : {OUTPUT_CSV}")
    print(f"  Confidence dist  : {CONF_DIST}")
    print(f"  Validation sample: {validation_path}")
    print(f"  LLM call log     : {LLM_LOG}")
    print(f"  Reliable fraction: {high_med}/{total} "
          f"({100*high_med/total:.1f}%) at high+medium confidence")
    print(f"  AutoModerator rows: {label_totals.get(LABEL_AUTOMOD, 0)}")
    print(f"  LCR system-prompt rows: {label_totals.get(LABEL_LCR_SP, 0)}")
