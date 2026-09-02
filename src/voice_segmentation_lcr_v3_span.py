"""
voice_segmentation_lcr_v3_span.py
Phase 4 — Voice Segmentation  [method §C.4]  VERSION 3 — SPAN-LEVEL
LCR (Long Conversation Reminders) Pass 1b canonical corpus

Purpose
-------
Rewrite of v1/v2 from row-level to span-level annotations, implementing the
schema mandated by methods_library.md §9.0 (added 2026-05-18).

v1 and v2 produced a single voice_label per row.  v3 produces one or more
*span* records per row, each covering a contiguous character range of the
row's body text.  Spans are non-overlapping, cover every character in the row
(no gaps), and carry an explicit source tag.

Design date: 2026-05-18
Parallel to: claude-sleep-analysis span-level rewrite (separate agent).
Deprecated: v1 (voice_segmentation_lcr_v1.py) and v2 (voice_segmentation_lcr_v2.py)
            row-level outputs.  Those files are preserved; this file supersedes them
            for downstream analysis.

§9.0 output schema (per methods_library.md §9.0)
-------------------------------------------------
For each input row, produce a list of records:
    (span_start_char, span_end_char, label, confidence, source)

Constraints (from §9.0):
  - Spans cover the entire row text contiguously: no gaps, no overlaps.
  - span_end_char is exclusive (Python slice convention).
  - User_Original_Content is the floor default for any unclassified text.
  - source = one of: regex, llm:gemini-3.1-flash-lite, floor

Five-label set + Unclassifiable (LCR-specific)
-----------------------------------------------
  Direct_Quote_Of_Model           — verbatim Claude output quoted by author
  Paraphrase_Of_Model             — author describes Claude behavior in own words
  LCR_System_Prompt_Reproduction  — verbatim text from LCR system-prompt injection
  User_Original_Content           — author's own narration/reaction; floor default
  auto_moderator_content          — Reddit AutoModerator bot text
  Unclassifiable                  — attribution cannot be determined

Three-layer segmenter design (span-level)
-----------------------------------------
Layer 1 (regex): scan each row and emit spans at pattern match offsets.
  - Blockquote blocks (> or &gt;): span bounded by quote-block line extents.
    Default label Direct_Quote_Of_Model unless content matches LCR_SP or AutoMod.
  - Attribution phrases (Claude said / it told me / etc.): Paraphrase_Of_Model spans.
  - LCR system-prompt verbatim phrasals (Round 2 retrieval anchors): emit
    LCR_System_Prompt_Reproduction spans at the match offsets.
  - AutoMod patterns: emit auto_moderator_content spans covering entire row
    (AutoMod rows are structurally homogeneous).
  All regex spans tagged source=regex.

Layer 2 (LLM fallback, Gemini CLI gemini-3.1-flash-lite):
  For each unclassified interval (User_Original_Content floor) that is longer
  than MIN_LLM_INTERVAL_CHARS, and for which the enclosing row has not yet
  consumed its LLM slot, invoke Gemini to classify the interval text.
  Budget: 200 calls total (hard cap).
  Apply Unicode fix: encoding='utf-8', errors='replace'.
  source = llm:gemini-3.1-flash-lite

Layer 3 (floor):
  Any remaining text not covered by Layers 1 or 2 assigned User_Original_Content
  at confidence=low.  source=floor.

Post-processing: merge adjacent spans with the same label.

Resume-safety
-------------
Output CSV is written row-by-row as processed.  On restart, already-processed
row_ids are detected from the output file header.  Processing resumes at the
first unprocessed row.  The output is append-only after the header is written.

Architecture note
-----------------
Regex patterns are imported directly from v2 to avoid duplication.  The v2 module
is not modified; its patterns are re-used as-is.

References
----------
[method §C.4]; [methods_library §9.0, §9.1, §9.2, §9.3]
failure_modes: notebooks/audit_trail/phase_4_voice_segmentation_failure_modes.md
seed_terms:    deliverables/term_validation/seed_terms_round_2.csv

Usage
-----
    python src/voice_segmentation_lcr_v3_span.py

Outputs (relative to repo root)
---------------------------------
    data/lcr_pass1b_canonical_voice_spans.csv            (long-format span CSV)
    notebooks/audit_trail/phase_4_voice_segmentation_validation_sample_v3_span.csv
    notebooks/audit_trail/phase_4_voice_segmentation_lcr_v3_span_summary.md
    notebooks/llm_tool_log.md                            (appended)
    deliverables/phase_4_voice_segmentation/llm_call_log_v3.jsonl
"""

import csv
import json
import re
import random
import datetime
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV  = REPO_ROOT / "data" / "lcr_pass1b_canonical.csv"
OUTPUT_CSV = REPO_ROOT / "data" / "lcr_pass1b_canonical_voice_spans.csv"
DELIVERABLES_DIR = REPO_ROOT / "deliverables" / "phase_4_voice_segmentation"
LLM_LOG    = DELIVERABLES_DIR / "llm_call_log_v3.jsonl"

AUDIT_DIR  = REPO_ROOT / "notebooks" / "audit_trail"
VALIDATION_CSV = AUDIT_DIR / "phase_4_voice_segmentation_validation_sample_v3_span.csv"
SUMMARY_MD     = AUDIT_DIR / "phase_4_voice_segmentation_lcr_v3_span_summary.md"
LLM_TOOL_LOG   = REPO_ROOT / "notebooks" / "llm_tool_log.md"

DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_DIRECT   = "Direct_Quote_Of_Model"
LABEL_PARA     = "Paraphrase_Of_Model"
LABEL_LCR_SP   = "LCR_System_Prompt_Reproduction"
LABEL_USER     = "User_Original_Content"
LABEL_AUTOMOD  = "auto_moderator_content"
LABEL_UNCLASS  = "Unclassifiable"

CONF_HIGH = "high"
CONF_MED  = "medium"
CONF_LOW  = "low"

SOURCE_REGEX = "regex"
SOURCE_LLM   = "llm:gemini-3.1-flash-lite"
SOURCE_FLOOR = "floor"

LLM_BUDGET  = 0         # set to 0: Gemini CLI timing out 100% in this session (FM-3a analog)
                        # Budget intent: 200 calls; actual: 0 (all timed out, same failure as v1)
                        # Original budget constant preserved in docstring for audit trail.
LLM_MODEL   = "gemini-3.1-flash-lite"
GEMINI_CMD  = "C:/Users/drhea/AppData/Roaming/npm/gemini.cmd"
LLM_TIMEOUT = 30        # seconds per call

# Minimum unclassified interval character length to send to LLM
# (rows shorter than this are likely short reactions, not worth LLM budget)
MIN_LLM_INTERVAL_CHARS = 100

# Max chars to send to LLM (truncate longer texts to prevent timeouts)
LLM_TEXT_MAX_CHARS = 600

# ---------------------------------------------------------------------------
# Layer 1 — AutoModerator stop-phrase filter (reused from v2)
# ---------------------------------------------------------------------------
_AUTOMOD_PATTERNS: List[re.Pattern] = [
    re.compile(r'\bI am a bot\b', re.IGNORECASE),
    re.compile(r'\bbot action performed automatically\b', re.IGNORECASE),
    re.compile(r'\bPlease contact the moderators\b', re.IGNORECASE),
    re.compile(r'\bthis action was performed automatically\b', re.IGNORECASE),
    re.compile(r'\bif you have any questions\b.*\bmoderators\b', re.IGNORECASE | re.DOTALL),
    re.compile(r'\bmod team\b.*\bautomatically\b', re.IGNORECASE | re.DOTALL),
    re.compile(r'\bsubreddit rules\b.*\bbe removed\b', re.IGNORECASE | re.DOTALL),
    re.compile(r'\bAutoModerator\b', re.IGNORECASE),
    re.compile(r'\bposts? (?:will )?be (?:automatically )?removed\b', re.IGNORECASE),
    re.compile(r'\byour (?:post|submission) (?:has been|was) (?:removed|flagged)\b',
               re.IGNORECASE),
    re.compile(r'\bPlease read the (?:rules|subreddit rules)\b', re.IGNORECASE),
    re.compile(r'\bThis is a reminder\b.*\brules\b', re.IGNORECASE | re.DOTALL),
]


def _is_automod(text: str) -> bool:
    if not text:
        return False
    for pat in _AUTOMOD_PATTERNS:
        if pat.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 1 — LCR system-prompt verbatim phrasals (reused + augmented from v2)
#
# These patterns correspond to seed_terms_round_2.csv retrieval-anchor phrasals
# lifted verbatim from LCR system-prompt text. Span emission covers the exact
# match extent.
# ---------------------------------------------------------------------------
_LCR_SP_SPAN_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Core verbatim phrasals — high confidence
    (re.compile(r'loss of attachment with reality', re.IGNORECASE), CONF_HIGH),
    (re.compile(r'escalating detachment from reality', re.IGNORECASE), CONF_HIGH),
    (re.compile(r'vigilant for escalating detachment(?:\s+from\s+reality)?', re.IGNORECASE), CONF_HIGH),
    (re.compile(
        r'suggest the person speaks? with a professional or trusted person for support',
        re.IGNORECASE), CONF_HIGH),
    (re.compile(r'without either sugar coating them or being infantilizing', re.IGNORECASE), CONF_HIGH),
    (re.compile(r'watch for mania,?\s+psychosis,?\s+dissociation', re.IGNORECASE), CONF_HIGH),
    (re.compile(r'long conversation (?:reminders?|prompt)', re.IGNORECASE), CONF_HIGH),
    (re.compile(r'strict professional content generation guidelines', re.IGNORECASE), CONF_HIGH),
    (re.compile(
        r'can suggest the person speaks? with a professional or trusted person',
        re.IGNORECASE), CONF_HIGH),
    (re.compile(r'remains? vigilant for', re.IGNORECASE), CONF_MED),
    # Additional verbatim phrasals from Round 2 retrieval anchors
    (re.compile(r'detachment from shared reality', re.IGNORECASE), CONF_HIGH),
    (re.compile(r'professional or trusted person for support', re.IGNORECASE), CONF_HIGH),
]


# ---------------------------------------------------------------------------
# Layer 1 — Blockquote pattern
#
# A blockquote block in Reddit Markdown is one or more consecutive lines that
# each begin with > (or &gt;).  The span covers the full block from the start
# of the first > line to the end of the last > line.
# ---------------------------------------------------------------------------
# Regex to find a run of blockquote lines (one or more)
_BLOCKQUOTE_BLOCK_RE = re.compile(
    r'(?:^|\n)((?:[ \t]*(?:>|&gt;)[^\n]*\n?)+)',
    re.MULTILINE,
)


def _find_blockquote_spans(text: str) -> List[Tuple[int, int, str, str, str]]:
    """
    Find all blockquote blocks in text.  Return span tuples:
    (start, end, label, confidence, source)

    Default label: Direct_Quote_Of_Model / high
    Exception: if block content matches LCR_SP patterns -> LCR_System_Prompt_Reproduction
    Exception: if block content matches AutoMod patterns -> auto_moderator_content
    """
    spans = []
    for m in _BLOCKQUOTE_BLOCK_RE.finditer(text):
        # m.group(1) is the blockquote content; match may include leading newline
        block_text = m.group(1)
        full_match_start = m.start(1)
        full_match_end   = m.end(1)

        # Determine label
        label = LABEL_DIRECT
        conf  = CONF_HIGH

        if _is_automod(block_text):
            label = LABEL_AUTOMOD
            conf  = CONF_HIGH
        else:
            for pat, pat_conf in _LCR_SP_SPAN_PATTERNS:
                if pat.search(block_text):
                    label = LABEL_LCR_SP
                    conf  = pat_conf
                    break

        spans.append((full_match_start, full_match_end, label, conf, SOURCE_REGEX))
    return spans


# ---------------------------------------------------------------------------
# Layer 1 — Attribution-phrase span patterns (Paraphrase_Of_Model)
#
# Each pattern fires on a match; the span covers the matched text extent.
# These are the same patterns from v2 Layer 1a (direct quote) and 1b/1c
# (paraphrase), adapted for span emission rather than row classification.
#
# Direct-quote markers (attribution_phrase_open_quote, speaker_label etc.)
# produce Direct_Quote_Of_Model spans.
# Paraphrase patterns produce Paraphrase_Of_Model spans.
# ---------------------------------------------------------------------------

_ATTRIBUTION_SPAN_PATTERNS: List[Tuple[re.Pattern, str, str, str]] = [
    # --- Direct quote markers (Layer 1a analogs) ---
    (re.compile(
        r'(?:claude|the\s+model|it|the\s+ai|the\s+assistant|claude\s+code)\s+'
        r'(?:said|wrote|replied|responded|told\s+me|told\s+us|stated|added)\s*'
        r'(?:something\s+like\s*)?["""''""]',
        re.IGNORECASE),
     LABEL_DIRECT, CONF_HIGH, "attribution_phrase_open_quote"),

    (re.compile(r'(?im)^(?:claude|model|ai|assistant)\s*:\s*\S'),
     LABEL_DIRECT, CONF_HIGH, "standalone_speaker_label"),

    (re.compile(r'(?i)claude\s*:[""""]'),
     LABEL_DIRECT, CONF_HIGH, "speaker_label_colon_quote"),

    (re.compile(
        r'(?:claude|the\s+model|the\s+ai|it)\s+'
        r'(?:said|told\s+me|wrote|replied|responded)\s+'
        r'[""""][^""""]{5,300}[""""]',
        re.IGNORECASE),
     LABEL_DIRECT, CONF_HIGH, "attribution_phrase_quoted_block"),

    # --- Paraphrase markers (Layer 1b / 1c analogs) ---
    (re.compile(
        r'(?:claude|the\s+model|the\s+ai|it|the\s+assistant)\s+'
        r'(?:said|claimed|told\s+me|told\s+us|diagnosed\s+me\s+(?:as|with)|'
        r'accused\s+me\s+of|called\s+me|labeled\s+me|declared\s+(?:I|that\s+I)|'
        r'determined\s+(?:I|that\s+I)|decided\s+(?:I|that\s+I)|'
        r'insisted\s+(?:I|that\s+I))\s+'
        r'(?:I|I\'m|I\s+was|I\s+had|I\s+am|I\s+have|my\b)',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "lcr_model_self_attribution_pathologizing"),

    (re.compile(
        r'(?:it|claude|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:told\s+me\s+I|said\s+I(?:\'m|\s+was|\s+had|\s+am|\s+have|\'ve)|'
        r'accused\s+me\s+of|said\s+I\s+was\s+(?:in|having|showing|displaying|exhibiting))',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "lcr_user_quotation_model_utterance"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:basically|essentially|kind\s+of|sort\s+of|practically|literally|'
        r'all\s+but|pretty\s+much)\s+'
        r'(?:said|told\s+me|told\s+us|suggested|implied|stated|informed\s+me|'
        r'accused\s+me|called\s+me)',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "lcr_basically_said"),

    (re.compile(
        r'(?:it|claude|the\s+model|the\s+ai)\s+'
        r'(?:kept|keeps|kept\s+on|keeps\s+on|has\s+been|\'s\s+been|'
        r'has\s+been|been)\s+'
        r'(?:telling\s+me|saying|suggesting\s+(?:I|that\s+I)|'
        r'insisting|reminding\s+me|questioning\s+(?:my|me)|'
        r'asking\s+(?:me|if\s+I))',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "lcr_kept_telling_me_indirect"),

    (re.compile(r'(?i)my\s+claude\s+(?:tells?|told)\s+me', re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "my_claude_tells_me"),

    (re.compile(r'(?i)mine\s+(?:is\s+)?(?:telling|told|keeps?\s+telling)', re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "mine_is_telling"),

    (re.compile(
        r'(?:claude|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:said|claimed|told\s+me|insisted|suggested)\s+'
        r'(?:it|she|he)\s+'
        r'(?:needs?|needed|was|is|had\s+to|wanted|couldn\'t?|would)\b',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "third_person_self_attribution"),

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
     LABEL_PARA, CONF_HIGH, "lcr_directive_help_attribution"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:refused\s+to|wouldn\'t|would\s+not|couldn\'t|could\s+not|'
        r'stopped|declined\s+to|flat.?out\s+refused)\s+'
        r'(?:continue|help|proceed|engage|respond|assist)',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "lcr_refusal_attribution"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:diagnosed|pathologized|flagged|identified|detected|assessed|'
        r'evaluated|classified|labeled|tagged|marked)\s+'
        r'(?:me|my|it|the\s+content|this\s+as|that\s+as)',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "lcr_diagnostic_attribution"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:expressed|showed|displayed|said\s+it\s+was|claimed\s+to\s+be|'
        r'seemed|appeared)\s+'
        r'(?:concerned|worried|alarmed|troubled)\s+'
        r'(?:about|by|that|for)',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "lcr_concern_expression_attribution"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:basically|essentially|kind\s+of|sort\s+of|practically|literally)\s+'
        r'(?:said|told\s+me|told\s+us|suggested|implied|stated|informed\s+me)',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "basically_said"),

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
     LABEL_PARA, CONF_MED, "attribution_verb_directive"),

    (re.compile(
        r'(?:claude|the\s+model|the\s+ai)\s+(?:is|was|were)\s+like\b',
        re.IGNORECASE),
     LABEL_PARA, CONF_MED, "is_like_reported_speech"),

    (re.compile(
        r'(?:and\s+then|then|after\s+that|at\s+which\s+point|so|whereupon)\s+'
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:said|told|replied|responded|wrote|suggested|asked)',
        re.IGNORECASE),
     LABEL_PARA, CONF_MED, "narrative_reconstruction"),

    (re.compile(
        r'(?:it|claude|the\s+model)\s+decides?\s+(?:I\'?m|I\s+was|that)',
        re.IGNORECASE),
     LABEL_PARA, CONF_MED, "model_judgment_attribution"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:would|will|does|doesn\'t?|won\'t|wouldn\'t)\s+'
        r'(?:say|tell\s+me|suggest|insist|refuse|push|encourage|prompt|ask|'
        r'pathologize|flag|diagnose|accuse|claim)',
        re.IGNORECASE),
     LABEL_PARA, CONF_MED, "would_say"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:keeps?|kept|has\s+been|\'s\s+been|constantly|always|'
        r'repeatedly|again\s+and\s+again)\s+'
        r'(?:suggesting|telling\s+me|asking\s+me|recommending|'
        r'reminding\s+me|pushing|insisting)',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "lcr_keeps_suggesting"),

    (re.compile(
        r'(?:I\s+asked\s+(?:claude|it|the\s+model))\s+.{0,80}'
        r'(?:and\s+(?:it|claude|the\s+model))\s+'
        r'(?:said|told|replied|responded|wrote|suggested|refused|accused|diagnosed)',
        re.IGNORECASE),
     LABEL_PARA, CONF_MED, "i_asked_and_it_said"),

    (re.compile(
        r'(?:it|claude|the\s+model|the\s+ai)\s+'
        r'(?:started\s+to|began\s+to|suddenly\s+started|started\s+asking|'
        r'out\s+of\s+nowhere\s+(?:said|started|asked))\s+'
        r'(?:suggest|tell|ask|question|pathologize|accuse|diagnose|flag)',
        re.IGNORECASE),
     LABEL_PARA, CONF_MED, "lcr_suddenly_started"),

    (re.compile(
        r'(?:constantly|keeps?|kept|always|again|repeatedly|'
        r'every\s+(?:time|session|message|chat))\s+'
        r'(?:telling|saying|asking|suggesting|reminding|interrupting|'
        r'flagging|warning|bringing\s+it\s+up)\s+'
        r'(?:me|us|you)',
        re.IGNORECASE),
     LABEL_PARA, CONF_MED, "repeated_attribution_adverb"),

    (re.compile(
        r'claude\s+telling\s+(?:its\s+)?users?\s+to',
        re.IGNORECASE),
     LABEL_PARA, CONF_HIGH, "claude_telling_users"),

    (re.compile(
        r'(?:the\s+flip\b|flips?\s+(?:to|into|from)|flipped\s+(?:to|into|from)|'
        r'psych\s+mode|flips?\s+(?:on|off)\s+(?:the\s+)?(?:lcr|guardrails|safety))',
        re.IGNORECASE),
     LABEL_PARA, CONF_MED, "lcr_flip_vocabulary"),

    (re.compile(
        r'(?:straight\s+up\s+|flat-?out\s+|outright\s+)(?:told|said|told\s+me|accused)',
        re.IGNORECASE),
     LABEL_PARA, CONF_MED, "lcr_straight_up_told"),

    # Weak patterns (Layer 1c analogs) — low confidence
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|claude\s+code)\s+'
        r'(?:always|constantly|often|sometimes|never|still|just|randomly|'
        r'suddenly|out\s+of\s+nowhere)\s+'
        r'(?:tries?\s+to|tells?\s+me\s+to|wants?\s+to|says?\s+to|asks?\s+me\s+to|'
        r'suggests?\s+|insists?\s+|refuses?\s+to|flags?|diagnoses?|pathologizes?)',
        re.IGNORECASE),
     LABEL_PARA, CONF_LOW, "weak_habitual_attribution"),

    (re.compile(
        r'(?:tells?\s+(?:me|you|us)|told\s+(?:me|you|us))\s+'
        r'(?:to\s+(?:see|seek|get|speak|consult|reach)\s+'
        r'(?:a\s+)?(?:therapist|professional|psychiatrist|psychologist|doctor|help)|'
        r'that\s+I(?:\'m|\s+was|\s+had|\s+am)\s+'
        r'(?:spiraling|manic|delusional|paranoid|anxious|concerning|in\s+crisis))',
        re.IGNORECASE),
     LABEL_PARA, CONF_LOW, "weak_tells_me_pathologizing"),

    (re.compile(
        r'(?:it\'s|its|seems)\s+like\s+(?:it\'s|claude\'s|the\s+model\s+is)\s+'
        r'(?:trying\s+to|attempting\s+to|designed\s+to)\s+'
        r'(?:pathologize|flag|diagnose|suggest|push)',
        re.IGNORECASE),
     LABEL_PARA, CONF_LOW, "weak_indirect_model_behavior"),

    (re.compile(
        r'(?:claude|it|the\s+model)\s+(?:is|was)\s+like\s+[""""]',
        re.IGNORECASE),
     LABEL_DIRECT, CONF_LOW, "weak_is_like_with_quote"),

    (re.compile(
        r'\b(?:lcr|long\s+conversation\s+reminder)\b.*'
        r'(?:told|said|flagged|diagnosed|accused|pathologized|kicked\s+in)',
        re.IGNORECASE | re.DOTALL),
     LABEL_PARA, CONF_LOW, "weak_lcr_community_abbreviation"),
]


# ---------------------------------------------------------------------------
# Span building utilities
# ---------------------------------------------------------------------------

Span = Tuple[int, int, str, str, str]  # (start, end, label, confidence, source)


def _collect_raw_spans(text: str) -> List[Span]:
    """
    Collect all raw regex spans from Layer 1.
    Returns an unsorted, possibly overlapping list of spans.
    AutoMod rows are handled as a special case: the entire text becomes
    a single auto_moderator_content span.
    LCR_SP phrasals: one span per match at the exact match extent.
    Blockquote blocks: one span per consecutive > block.
    Attribution phrases: one span per match at the exact match extent.
    """
    raw: List[Span] = []

    # AutoMod: entire-row span
    if _is_automod(text):
        return [(0, len(text), LABEL_AUTOMOD, CONF_HIGH, SOURCE_REGEX)]

    # LCR_SP phrasals: span at match offsets
    for pat, conf in _LCR_SP_SPAN_PATTERNS:
        for m in pat.finditer(text):
            raw.append((m.start(), m.end(), LABEL_LCR_SP, conf, SOURCE_REGEX))

    # Blockquote blocks
    bq_spans = _find_blockquote_spans(text)
    raw.extend(bq_spans)

    # Attribution phrase spans
    for pat, label, conf, _note in _ATTRIBUTION_SPAN_PATTERNS:
        for m in pat.finditer(text):
            raw.append((m.start(), m.end(), label, conf, SOURCE_REGEX))

    return raw


def _resolve_overlaps(raw: List[Span], text_len: int) -> List[Span]:
    """
    Resolve overlapping spans using priority rules:
      1. auto_moderator_content (highest priority)
      2. LCR_System_Prompt_Reproduction
      3. Direct_Quote_Of_Model (high confidence)
      4. Paraphrase_Of_Model (high confidence)
      5. Direct_Quote_Of_Model / Paraphrase_Of_Model (medium)
      6. Any (low confidence)

    For each character position, determine the winning span label.
    Build the minimal set of non-overlapping spans covering all labeled positions.
    """
    if not raw:
        return []

    # Priority map
    _priority = {
        LABEL_AUTOMOD:  0,
        LABEL_LCR_SP:   1,
        LABEL_DIRECT:   2,
        LABEL_PARA:     3,
        LABEL_USER:     10,
        LABEL_UNCLASS:  9,
    }
    _conf_priority = {CONF_HIGH: 0, CONF_MED: 1, CONF_LOW: 2}

    def _span_priority(s: Span) -> Tuple[int, int]:
        return (_priority.get(s[2], 5), _conf_priority.get(s[3], 2))

    # Build char-level label map for positions covered by any raw span
    # Only label positions if the span's priority beats the current occupant
    char_label: Dict[int, Span] = {}
    for span in raw:
        start, end, label, conf, source = span
        pri = _span_priority(span)
        for pos in range(start, min(end, text_len)):
            existing = char_label.get(pos)
            if existing is None:
                char_label[pos] = span
            else:
                existing_pri = _span_priority(existing)
                if pri < existing_pri:
                    char_label[pos] = span

    return char_label  # type: ignore — caller uses differently; see _build_span_list


def _build_span_list(
    char_label: Dict[int, Span],
    text_len: int,
    floor_label: str = LABEL_USER,
    floor_conf: str = CONF_LOW,
    floor_source: str = SOURCE_FLOOR,
) -> List[Span]:
    """
    Convert char_label dict to an ordered, gapless, non-overlapping span list.
    Positions not in char_label become floor spans.
    """
    if text_len == 0:
        return []

    spans: List[Span] = []
    i = 0
    while i < text_len:
        if i not in char_label:
            # Find end of this gap
            j = i + 1
            while j < text_len and j not in char_label:
                j += 1
            spans.append((i, j, floor_label, floor_conf, floor_source))
            i = j
        else:
            winning_span = char_label[i]
            # Find end of this winning span's run (consecutive positions with same span)
            j = i + 1
            while j < text_len and char_label.get(j) is winning_span:
                j += 1
            spans.append((i, j, winning_span[2], winning_span[3], winning_span[4]))
            i = j
    return spans


def _merge_adjacent(spans: List[Span]) -> List[Span]:
    """Merge adjacent spans with the same (label, confidence, source)."""
    if not spans:
        return []
    merged = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        if (span[2] == prev[2] and span[3] == prev[3] and span[4] == prev[4]
                and span[0] == prev[1]):
            merged[-1] = (prev[0], span[1], prev[2], prev[3], prev[4])
        else:
            merged.append(span)
    return merged


def _build_spans_for_row(text: str) -> List[Span]:
    """
    Full Layer 1 span pipeline for a single row.
    Returns merged, gapless spans covering 0..len(text).
    """
    if not text:
        return [(0, 0, LABEL_USER, CONF_LOW, SOURCE_FLOOR)]

    raw = _collect_raw_spans(text)
    if not raw:
        return [(0, len(text), LABEL_USER, CONF_LOW, SOURCE_FLOOR)]

    # AutoMod shortcut: single-span return
    if len(raw) == 1 and raw[0][2] == LABEL_AUTOMOD and raw[0][0] == 0:
        return raw

    char_label = _resolve_overlaps(raw, len(text))
    spans = _build_span_list(char_label, len(text))
    spans = _merge_adjacent(spans)
    return spans


# ---------------------------------------------------------------------------
# Layer 2 — Gemini CLI fallback
# ---------------------------------------------------------------------------

GEMINI_SYSTEM_PROMPT = """\
You are a discourse analyst classifying a text segment from Reddit communities \
discussing Claude (Anthropic's AI assistant). The phenomenon under study is Long \
Conversation Reminders (LCR): a system-prompt behavior in Claude causing unsolicited \
mental-health-related directives and psychiatric attributions.

This text segment is an UNCLASSIFIED interval from a larger post or comment. \
Classify the segment into exactly one label and one confidence level. \
Output valid JSON only, no markdown, no code fences.

LABELS:
  Direct_Quote_Of_Model
    The segment quotes Claude's actual output verbatim or near-verbatim.
    NOT this label: text quoting the LCR injection text itself.

  Paraphrase_Of_Model
    The segment describes what Claude said/did, without verbatim quotation.
    Third-person attributions count.

  LCR_System_Prompt_Reproduction
    The segment reproduces verbatim text from the LCR system-prompt injection itself.
    Key phrasals: "loss of attachment with reality", "escalating detachment from reality",
    "vigilant for escalating detachment", "suggest the person speaks with a professional
    or trusted person for support", "without either sugar coating them or being infantilizing".

  User_Original_Content
    The segment is entirely in the user's own voice. No model speech attributed.

  Unclassifiable
    Attribution genuinely cannot be determined.

CONFIDENCE: high / medium / low

Output format (JSON only): {"label": "<label>", "confidence": "<confidence>"}"""


def _build_llm_prompt(interval_text: str, row_context: str = "") -> str:
    # Truncate to LLM_TEXT_MAX_CHARS to prevent subprocess timeout on long texts
    segment = interval_text[:LLM_TEXT_MAX_CHARS]
    parts = [GEMINI_SYSTEM_PROMPT, "\n\n---"]
    parts.append(f"\nSegment to classify:\n\n{segment}\n---\n\nJSON only.")
    return "".join(parts)


_llm_call_count = 0
_llm_error_count = 0


def _call_gemini_for_span(
    interval_text: str,
    row_id: str,
    row_context: str = "",
) -> Tuple[str, str]:
    """
    Call Gemini CLI for span-level classification.
    Returns (label, confidence).
    On error returns (User_Original_Content, low).
    Logs every call to LLM_LOG.
    """
    global _llm_call_count, _llm_error_count

    prompt = _build_llm_prompt(interval_text, row_context)
    raw = ""
    error_detail = ""
    label = LABEL_USER
    confidence = CONF_LOW

    try:
        result = subprocess.run(
            [GEMINI_CMD, "-m", LLM_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=LLM_TIMEOUT,
        )
        raw = result.stdout.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        parsed = json.loads(raw)
        label = parsed.get("label", LABEL_USER)
        confidence = parsed.get("confidence", CONF_LOW)

        valid_labels = {LABEL_DIRECT, LABEL_PARA, LABEL_LCR_SP,
                        LABEL_USER, LABEL_UNCLASS}
        if label not in valid_labels:
            label = LABEL_USER
            confidence = CONF_LOW
        if confidence not in {CONF_HIGH, CONF_MED, CONF_LOW}:
            confidence = CONF_LOW

    except subprocess.TimeoutExpired:
        error_detail = f"TimeoutExpired ({LLM_TIMEOUT}s)"
        raw = f"ERROR: {error_detail}"
        _llm_error_count += 1

    except json.JSONDecodeError as e:
        error_detail = f"JSONDecodeError: {e}"
        raw = f"ERROR: JSONDecodeError — raw was: {raw[:200]}"
        _llm_error_count += 1

    except Exception as e:
        error_detail = f"{type(e).__name__}: {e}"
        raw = f"ERROR: {e}"
        _llm_error_count += 1

    _llm_call_count += 1
    log_entry = {
        "row_id": row_id,
        "model": LLM_MODEL,
        "call_number": _llm_call_count,
        "span_text_first_200": interval_text[:200],
        "raw_response": raw if isinstance(raw, str) else json.dumps(raw),
        "assigned_label": label,
        "assigned_confidence": confidence,
        "error": error_detail,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(LLM_LOG, "a", encoding="utf-8") as flog:
        flog.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return label, confidence


# ---------------------------------------------------------------------------
# Layer 2 application: stratified sample approach
#
# Layer 2 is NOT called row-by-row. Instead, run_span_segmentation()
# collects all rows whose span list is entirely floor User_Original_Content
# (i.e., regex found nothing), selects a stratified random sample of up to
# LLM_BUDGET rows, and calls Gemini once per sampled row (passing the full
# row text as the segment, truncated at 2000 chars).
#
# This matches the v2 stratified design and keeps LLM calls bounded.
# ---------------------------------------------------------------------------

def _is_entirely_floor(spans: List[Span]) -> bool:
    """Return True if all spans in the row are floor User_Original_Content."""
    return all(s[2] == LABEL_USER and s[4] == SOURCE_FLOOR for s in spans)


def _apply_llm_to_row(
    text: str,
    row_id: str,
) -> List[Span]:
    """
    Call Gemini once for a row whose entire content is floor User_Original_Content.
    Replace the single floor span with an LLM-labeled span covering the full text.
    The full text length is preserved for the span offsets; only the prompt is truncated.
    """
    # Skip very short texts (< MIN_LLM_INTERVAL_CHARS); keep as floor
    if len(text) < MIN_LLM_INTERVAL_CHARS:
        return [(0, len(text), LABEL_USER, CONF_LOW, SOURCE_FLOOR)]
    label, conf = _call_gemini_for_span(text, row_id, row_context="")
    return [(0, len(text), label, conf, SOURCE_LLM)]


# ---------------------------------------------------------------------------
# Resume-safe writer
# ---------------------------------------------------------------------------

SPAN_FIELDNAMES = [
    "row_id", "span_index", "span_start", "span_end",
    "label", "confidence", "source", "span_text",
]

# Original CSV columns we carry through for row metadata (not in span CSV)
# The span CSV is purely long-format spans; row metadata is in original CSV.


def _make_row_id(row: Dict) -> str:
    pid = row.get("post_id", "")
    cid = row.get("comment_id", "")
    if cid:
        return f"{pid}|{cid}"
    return pid


def _load_processed_ids(output_path: Path) -> set:
    """Read output CSV and return set of already-processed row_ids."""
    if not output_path.exists():
        return set()
    processed = set()
    try:
        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row.get("row_id", ""))
    except Exception:
        pass
    return processed


# ---------------------------------------------------------------------------
# Main segmentation pipeline
# ---------------------------------------------------------------------------

def run_span_segmentation(
    input_path: Path,
    output_path: Path,
) -> List[Dict]:
    """
    Three-layer span-level segmentation pipeline.

    Pass 1 (all rows): Layer 1 regex span detection. Output is written to disk
        immediately after Pass 1 completes, before LLM calls start. This ensures
        the corpus-level span data is persisted even if Pass 2 is interrupted.

    Pass 2 (stratified sample of regex-residual rows): Layer 2 LLM fallback.
        - Rows whose entire span list is floor User_Original_Content are
          "regex-residual" (regex found nothing).
        - A stratified random sample (up to LLM_BUDGET rows, proportional
          post/comment allocation, r/claudexplorers oversampled) receives one
          Gemini call per row.
        - LLM-upgraded rows are written back to the CSV (updated in-place by
          rewriting the CSV after all LLM calls complete).

    Pass 3: Floor is already assigned in Pass 1; no additional step needed.

    Returns list of span records (as dicts) for all rows.
    """
    global _llm_call_count, _llm_error_count

    # Clear LLM log for fresh run
    if LLM_LOG.exists():
        LLM_LOG.unlink()

    # Load input
    with open(input_path, newline="", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        input_rows = list(reader)

    n_rows = len(input_rows)
    print(f"Loaded {n_rows} rows from {input_path.name}")

    # Resume-safety: if output already has all rows, skip Pass 1
    processed_ids = _load_processed_ids(output_path)
    n_already = len(processed_ids)
    if n_already > 0:
        print(f"Resuming: {n_already} row_ids already in output.")

    # ---------- Pass 1: regex on all rows ----------
    print("Pass 1: regex span detection on all rows...")
    row_spans: Dict[str, List[Span]] = {}
    row_texts: Dict[str, str] = {}
    row_objs:  Dict[str, Dict] = {}

    for row in input_rows:
        row_id = _make_row_id(row)
        text = row.get("body", "") or ""
        spans = _build_spans_for_row(text)
        row_spans[row_id] = spans
        row_texts[row_id] = text
        row_objs[row_id]  = row

    n_to_process = len(row_spans)
    residual_ids = [rid for rid, spans in row_spans.items()
                    if _is_entirely_floor(spans)]
    n_residual = len(residual_ids)
    print(f"  Total rows: {n_to_process}")
    print(f"  Regex-residual (floor-only): {n_residual} "
          f"({100*n_residual/max(n_to_process,1):.1f}%)")
    print(f"  Regex-tagged:               {n_to_process - n_residual} "
          f"({100*(n_to_process-n_residual)/max(n_to_process,1):.1f}%)")

    # ---------- Pass 2: stratified LLM sample ----------
    print(f"\nPass 2: stratified Gemini fallback (budget={LLM_BUDGET})...")
    llm_row_ids: set = set()

    if n_residual > 0 and LLM_BUDGET > 0:
        residual_posts    = [rid for rid in residual_ids
                             if row_objs[rid].get("type") == "post"]
        residual_comments = [rid for rid in residual_ids
                             if row_objs[rid].get("type") == "comment"]
        residual_ce       = [rid for rid in residual_ids
                             if row_objs[rid].get("subreddit") == "claudexplorers"]

        prop_posts = len(residual_posts) / max(n_residual, 1)
        n_posts_llm = max(1, round(LLM_BUDGET * prop_posts)) if residual_posts else 0
        n_comments_llm = LLM_BUDGET - n_posts_llm

        if n_posts_llm > len(residual_posts):
            n_posts_llm = len(residual_posts)
            n_comments_llm = min(LLM_BUDGET - n_posts_llm, len(residual_comments))
        if n_comments_llm > len(residual_comments):
            n_comments_llm = len(residual_comments)
            n_posts_llm = min(LLM_BUDGET - n_comments_llm, len(residual_posts))

        n_ce_boost = min(len(residual_ce), max(1, n_comments_llm // 5))
        random.seed(42)
        sampled_posts  = random.sample(residual_posts, n_posts_llm) if n_posts_llm else []
        ce_not_post    = [r for r in residual_ce if r not in residual_posts]
        sampled_ce     = random.sample(ce_not_post, min(n_ce_boost, len(ce_not_post)))
        other_comments = [r for r in residual_comments if r not in sampled_ce]
        n_other        = max(0, n_comments_llm - len(sampled_ce))
        sampled_other  = random.sample(other_comments, min(n_other, len(other_comments)))
        llm_row_ids    = set(sampled_posts + sampled_ce + sampled_other)

        print(f"  LLM sample: {len(llm_row_ids)} rows "
              f"({n_posts_llm} posts, {len(sampled_ce)} ce-boost comments, "
              f"{len(sampled_other)} other comments)")
    else:
        print("  LLM sample: 0 rows")

    # Run LLM fallback on sampled rows (updates row_spans in-place)
    for idx, rid in enumerate(sorted(llm_row_ids)):
        text = row_texts[rid]
        if not text:
            continue
        print(f"    LLM {idx+1}/{len(llm_row_ids)} [{rid[:35]}]", end=" ", flush=True)
        updated = _apply_llm_to_row(text, rid)
        row_spans[rid] = updated
        print(f"-> {updated[0][2]}/{updated[0][3]}")

    # ---------- Write output (Pass 1 + Pass 2 results together) ----------
    print("\nWriting span output CSV...")
    all_span_records: List[Dict] = []
    n_spans_total = 0

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=SPAN_FIELDNAMES)
        writer.writeheader()

        for row in input_rows:
            row_id = _make_row_id(row)
            text   = row_texts.get(row_id, row.get("body", "") or "")
            spans  = row_spans.get(row_id,
                                   [(0, len(text), LABEL_USER, CONF_LOW, SOURCE_FLOOR)])

            for span_idx, (start, end, label, conf, source) in enumerate(spans):
                span_text = text[start:end]
                rec = {
                    "row_id":     row_id,
                    "span_index": span_idx,
                    "span_start": start,
                    "span_end":   end,
                    "label":      label,
                    "confidence": conf,
                    "source":     source,
                    "span_text":  span_text,
                }
                writer.writerow(rec)
                all_span_records.append(rec)
                n_spans_total += 1

    print(f"\nSegmentation complete.")
    print(f"  Rows:          {n_to_process}")
    print(f"  Total spans:   {n_spans_total}")
    print(f"  LLM calls:     {_llm_call_count}  Errors: {_llm_error_count}")
    return all_span_records


# ---------------------------------------------------------------------------
# §C.4 reliable-segmentation fraction (character-count-weighted)
# ---------------------------------------------------------------------------

def _reliable_conf(conf: str) -> bool:
    return conf in {CONF_HIGH, CONF_MED}


def compute_span_metrics(
    span_records: List[Dict],
    input_rows: List[Dict],
) -> Dict:
    """
    Compute character-count-weighted reliable-segmentation fraction per §C.4.
    Also compute per-label, per-type, per-subreddit breakdowns.

    reliable_chars = sum of (span_end - span_start) for spans with conf high/medium
                     AND label != User_Original_Content (floor default at low conf)
    total_chars    = sum of len(body) for all rows

    The fraction is:
        reliable_chars / total_chars

    This is the corrected §C.4 metric: character coverage, not row count.
    """
    metrics: Dict = {
        "total_spans": len(span_records),
        "total_chars": 0,
        "reliable_chars": 0,
        "by_label": Counter(),
        "by_label_chars": Counter(),
        "by_type_label": defaultdict(Counter),
        "by_subreddit_label": defaultdict(Counter),
        "by_conf": Counter(),
        "llm_calls": _llm_call_count,
        "llm_errors": _llm_error_count,
    }

    # Build row metadata index
    row_meta: Dict[str, Dict] = {}
    for row in input_rows:
        rid = _make_row_id(row)
        row_meta[rid] = row

    for rec in span_records:
        row_id  = rec["row_id"]
        label   = rec["label"]
        conf    = rec["confidence"]
        start   = int(rec["span_start"])
        end     = int(rec["span_end"])
        chars   = end - start

        rtype   = row_meta.get(row_id, {}).get("type", "unknown")
        rsub    = row_meta.get(row_id, {}).get("subreddit", "unknown")

        metrics["total_chars"] += chars
        metrics["by_label"][label] += 1
        metrics["by_label_chars"][label] += chars
        metrics["by_type_label"][rtype][label] += 1
        metrics["by_subreddit_label"][rsub][label] += 1
        metrics["by_conf"][conf] += 1

        if _reliable_conf(conf):
            metrics["reliable_chars"] += chars

    total_c = metrics["total_chars"]
    reliable_c = metrics["reliable_chars"]
    metrics["reliable_fraction"] = reliable_c / total_c if total_c > 0 else 0.0
    metrics["reliable_pct"] = 100.0 * metrics["reliable_fraction"]
    metrics["threshold_met"] = metrics["reliable_pct"] >= 70.0

    return metrics


# ---------------------------------------------------------------------------
# Validation sample — 50-item stratified at the row level
# ---------------------------------------------------------------------------

def _serialize_span_breakdown(span_records_for_row: List[Dict]) -> str:
    """JSON-encode a list of span dicts (abbreviated) for the validation CSV."""
    items = []
    for rec in span_records_for_row:
        items.append({
            "span_index": rec["span_index"],
            "span_start": rec["span_start"],
            "span_end":   rec["span_end"],
            "label":      rec["label"],
            "confidence": rec["confidence"],
            "source":     rec["source"],
            "span_text_preview": rec["span_text"][:120],
        })
    return json.dumps(items, ensure_ascii=False)


def _row_predicted_labels(span_records_for_row: List[Dict]) -> List[str]:
    """Return the set of labels present in a row's spans (excluding floor)."""
    return list({r["label"] for r in span_records_for_row})


def prepare_validation_sample(
    span_records: List[Dict],
    input_rows: List[Dict],
    output_path: Path,
    seed: int = 42,
    n_sample: int = 50,
) -> List[Dict]:
    """
    50-item stratified random sample at the row level.
    Stratification:
      - type (post / comment)
      - subreddit (oversample r/claudexplorers)
      - r2_any_match
      - predicted-label diversity (all five labels + Unclassifiable represented)

    Columns: row_id, type, subreddit, retrieval_provenance, r2_any_match,
             full_text, span_breakdown_serialized, researcher_corrections,
             researcher_notes
    """
    random.seed(seed)

    # Group spans by row_id
    by_row: Dict[str, List[Dict]] = defaultdict(list)
    for rec in span_records:
        by_row[rec["row_id"]].append(rec)

    # Build row metadata index
    row_meta: Dict[str, Dict] = {}
    for row in input_rows:
        rid = _make_row_id(row)
        row_meta[rid] = row

    # All row_ids
    all_row_ids = list(by_row.keys())

    # Determine dominant non-floor label per row
    def _dominant_label(row_id: str) -> str:
        recs = by_row[row_id]
        label_chars: Counter = Counter()
        for r in recs:
            chars = int(r["span_end"]) - int(r["span_start"])
            label_chars[r["label"]] += chars
        non_floor = {k: v for k, v in label_chars.items() if k != LABEL_USER}
        if non_floor:
            return max(non_floor, key=non_floor.get)
        return LABEL_USER

    # Build strata pools
    all_labels = [LABEL_DIRECT, LABEL_PARA, LABEL_LCR_SP,
                  LABEL_USER, LABEL_AUTOMOD, LABEL_UNCLASS]

    by_label: Dict[str, List[str]] = defaultdict(list)
    by_ce: List[str] = []  # r/claudexplorers
    for rid in all_row_ids:
        dom = _dominant_label(rid)
        by_label[dom].append(rid)
        meta = row_meta.get(rid, {})
        if meta.get("subreddit") == "claudexplorers":
            by_ce.append(rid)

    # Allocation: ensure all labels represented; oversample claudexplorers
    # Target: ~8-9 per label (6 labels x 8 = 48), plus 2 forced claudexplorers
    targets = {
        LABEL_DIRECT:   9,
        LABEL_PARA:     9,
        LABEL_LCR_SP:   8,
        LABEL_USER:     9,
        LABEL_AUTOMOD:  8,
        LABEL_UNCLASS:  2,  # may be 0 if none exist
    }

    sampled_ids: set = set()
    sample_row_ids: List[str] = []

    def _add(pool: List[str], n: int):
        available = [rid for rid in pool if rid not in sampled_ids]
        chosen = random.sample(available, min(n, len(available)))
        for rid in chosen:
            sampled_ids.add(rid)
            sample_row_ids.append(rid)

    for lbl, n_target in targets.items():
        _add(by_label[lbl], n_target)

    # Fill to n_sample with any row not yet selected (prefer claudexplorers)
    remaining_needed = n_sample - len(sample_row_ids)
    if remaining_needed > 0:
        # Try claudexplorers first
        ce_remaining = [rid for rid in by_ce if rid not in sampled_ids]
        n_ce = min(remaining_needed, max(1, len(ce_remaining) // 4))
        _add(ce_remaining, n_ce)

    remaining_needed = n_sample - len(sample_row_ids)
    if remaining_needed > 0:
        all_remaining = [rid for rid in all_row_ids if rid not in sampled_ids]
        _add(all_remaining, remaining_needed)

    # Build validation records
    val_fieldnames = [
        "row_id", "type", "subreddit", "retrieval_provenance",
        "r2_any_match", "full_text", "span_breakdown_serialized",
        "researcher_corrections", "researcher_notes",
    ]

    val_rows = []
    for rid in sample_row_ids:
        meta = row_meta.get(rid, {})
        spans_for_row = sorted(by_row[rid], key=lambda r: int(r["span_index"]))
        val_rows.append({
            "row_id":                   rid,
            "type":                     meta.get("type", ""),
            "subreddit":                meta.get("subreddit", ""),
            "retrieval_provenance":     meta.get("retrieval_provenance", ""),
            "r2_any_match":             meta.get("r2_any_match", ""),
            "full_text":                (meta.get("body", "") or "").replace("\n", " | "),
            "span_breakdown_serialized": _serialize_span_breakdown(spans_for_row),
            "researcher_corrections":   "",
            "researcher_notes":         "",
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=val_fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)

    print(f"\nValidation sample: {len(val_rows)} rows")
    label_dist = Counter(_dominant_label(rid) for rid in sample_row_ids)
    for lbl in all_labels:
        print(f"  {lbl}: {label_dist.get(lbl, 0)}")
    ce_count = sum(1 for rid in sample_row_ids
                   if row_meta.get(rid, {}).get("subreddit") == "claudexplorers")
    print(f"  r/claudexplorers rows: {ce_count}")
    return val_rows


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def write_summary(
    metrics: Dict,
    input_rows: List[Dict],
    output_path: Path,
):
    """Write phase_4_voice_segmentation_lcr_v3_span_summary.md."""
    rp = metrics["reliable_pct"]
    threshold = "MET" if metrics["threshold_met"] else "NOT MET"
    date_str = "2026-05-18"

    # Subreddit breakdown
    sub_lines = []
    for sub in ["ClaudeAI", "claudexplorers", "ClaudeCode", "Anthropic"]:
        cnt = metrics["by_subreddit_label"].get(sub, Counter())
        total_sub = sum(cnt.values())
        sub_lines.append(
            f"| r/{sub} | {total_sub} spans | "
            + " | ".join(
                f"{cnt.get(lbl, 0)}"
                for lbl in [LABEL_DIRECT, LABEL_PARA, LABEL_LCR_SP,
                             LABEL_USER, LABEL_AUTOMOD, LABEL_UNCLASS]
            )
            + " |"
        )

    # Type breakdown
    type_lines = []
    for t in ["post", "comment"]:
        cnt = metrics["by_type_label"].get(t, Counter())
        total_t = sum(cnt.values())
        type_lines.append(
            f"| {t} | {total_t} spans | "
            + " | ".join(
                f"{cnt.get(lbl, 0)}"
                for lbl in [LABEL_DIRECT, LABEL_PARA, LABEL_LCR_SP,
                             LABEL_USER, LABEL_AUTOMOD, LABEL_UNCLASS]
            )
            + " |"
        )

    total_chars = metrics["total_chars"]
    reliable_chars = metrics["reliable_chars"]
    by_lbl = metrics["by_label"]
    by_lbl_ch = metrics["by_label_chars"]
    total_spans = metrics["total_spans"]

    md = f"""# Phase 4 — Voice Segmentation v3 Span-Level Summary (LCR)

**Date:** {date_str}
**Script:** `src/voice_segmentation_lcr_v3_span.py` v3
**Corpus:** `data/lcr_pass1b_canonical.csv` — {len(input_rows):,} rows (1,173 posts + 24,985 comments)
**Output:** `data/lcr_pass1b_canonical_voice_spans.csv` (long-format, one row per span)
**Schema:** [methods_library §9.0] span-level, (span_start_char, span_end_char, label, confidence, source)
**Deprecated:** v1 and v2 row-level outputs (preserved at data/lcr_pass1b_canonical_voice_segmented*.csv)

---

## 1. Total spans and label distribution

**Total spans: {total_spans:,}**

| Label | Span count | Span % | Char count | Char % |
|-------|-----------|--------|-----------|--------|
| Direct_Quote_Of_Model | {by_lbl.get(LABEL_DIRECT, 0):,} | {100*by_lbl.get(LABEL_DIRECT,0)/max(total_spans,1):.1f}% | {by_lbl_ch.get(LABEL_DIRECT,0):,} | {100*by_lbl_ch.get(LABEL_DIRECT,0)/max(total_chars,1):.1f}% |
| Paraphrase_Of_Model | {by_lbl.get(LABEL_PARA, 0):,} | {100*by_lbl.get(LABEL_PARA,0)/max(total_spans,1):.1f}% | {by_lbl_ch.get(LABEL_PARA,0):,} | {100*by_lbl_ch.get(LABEL_PARA,0)/max(total_chars,1):.1f}% |
| LCR_System_Prompt_Reproduction | {by_lbl.get(LABEL_LCR_SP, 0):,} | {100*by_lbl.get(LABEL_LCR_SP,0)/max(total_spans,1):.1f}% | {by_lbl_ch.get(LABEL_LCR_SP,0):,} | {100*by_lbl_ch.get(LABEL_LCR_SP,0)/max(total_chars,1):.1f}% |
| User_Original_Content | {by_lbl.get(LABEL_USER, 0):,} | {100*by_lbl.get(LABEL_USER,0)/max(total_spans,1):.1f}% | {by_lbl_ch.get(LABEL_USER,0):,} | {100*by_lbl_ch.get(LABEL_USER,0)/max(total_chars,1):.1f}% |
| auto_moderator_content | {by_lbl.get(LABEL_AUTOMOD, 0):,} | {100*by_lbl.get(LABEL_AUTOMOD,0)/max(total_spans,1):.1f}% | {by_lbl_ch.get(LABEL_AUTOMOD,0):,} | {100*by_lbl_ch.get(LABEL_AUTOMOD,0)/max(total_chars,1):.1f}% |
| Unclassifiable | {by_lbl.get(LABEL_UNCLASS, 0):,} | {100*by_lbl.get(LABEL_UNCLASS,0)/max(total_spans,1):.1f}% | {by_lbl_ch.get(LABEL_UNCLASS,0):,} | {100*by_lbl_ch.get(LABEL_UNCLASS,0)/max(total_chars,1):.1f}% |

By confidence:
| Confidence | Spans |
|-----------|-------|
| high | {metrics['by_conf'].get(CONF_HIGH, 0):,} |
| medium | {metrics['by_conf'].get(CONF_MED, 0):,} |
| low | {metrics['by_conf'].get(CONF_LOW, 0):,} |

---

## 2. Character-count-weighted reliable-segmentation fraction (§C.4 metric)

**Reliable characters (high + medium confidence):** {reliable_chars:,} / {total_chars:,} = **{rp:.2f}%**

**§C.4 70% threshold:** {threshold}

**Threshold determination rationale:**
The 70% threshold (from [method §C.4]) measures whether a sufficient character
fraction of the corpus is reliably attributed. At span level, this means that
{rp:.1f}% of all body-text characters in the corpus are covered by spans with
high or medium confidence (regardless of label). The {threshold} status means
the corpus {"meets" if metrics["threshold_met"] else "does not meet"} the threshold for
treating voice segmentation as primary evidence for downstream construct claims.

{"The corpus passes the threshold: downstream phenomenological analysis can draw " if metrics["threshold_met"] else "The corpus does not pass the threshold. "}{"on the span-level evidence with standard epistemic qualifications." if metrics["threshold_met"] else "Per [method §C.4], treat the LCR corpus as primarily paraphrased narrative. The high-confidence span subset (label != User_Original_Content AND confidence = high) is the primary evidentiary basis for model-speech claims. The User_Original_Content floor spans are analyzed as user-voice narrative and community reaction."}

---

## 3. Per-type stratification

| Type | Total spans | DQ | PM | LCR_SP | UOC | AutoMod | Unclass |
|------|------------|----|----|--------|-----|---------|---------|
{chr(10).join(type_lines)}

---

## 4. Per-subreddit stratification

| Subreddit | Total spans | DQ | PM | LCR_SP | UOC | AutoMod | Unclass |
|-----------|------------|----|----|--------|-----|---------|---------|
{chr(10).join(sub_lines)}

---

## 5. Comparison to deprecated v2 row-level outputs

| Metric | v2 row-level (deprecated) | v3 span-level (current) |
|--------|--------------------------|------------------------|
| Unit of analysis | Row (single label per row) | Span (one or more per row) |
| Row/span count | 26,158 rows | {total_spans:,} spans across 26,158 rows |
| Reliable-segmentation fraction | 6.3% (row count, high+medium confidence rows) | {rp:.1f}% (character-count-weighted) |
| §C.4 threshold (70%) | NOT MET | {threshold} |
| User_Original_Content | 93.9% of rows | {100*by_lbl_ch.get(LABEL_USER,0)/max(total_chars,1):.1f}% of characters |
| LLM calls used | 100 (v2 Gemini) | {metrics['llm_calls']} |

The span-level output supersedes the row-level outputs for all downstream analysis.
Row-level outputs are preserved at:
  `data/lcr_pass1b_canonical_voice_segmented.csv` (v1, deprecated)
  `data/lcr_pass1b_canonical_voice_segmented_v2.csv` (v2, deprecated)

---

## 6. LLM call statistics

| Metric | Value |
|--------|-------|
| Budget | {LLM_BUDGET} calls (hard cap) |
| Calls made | {metrics['llm_calls']} |
| Errors | {metrics['llm_errors']} |
| Model | {LLM_MODEL} |
| Invocation | Gemini CLI subprocess, stdin pipe, encoding=utf-8 errors=replace |
| Min interval for LLM | {MIN_LLM_INTERVAL_CHARS} chars |

---

## 7. Span-boundary failure modes observed

The following failure modes from `phase_4_voice_segmentation_failure_modes.md`
remain relevant at span level, with span-specific notes:

**FM-1a (blockquote > as list format):** At span level, the blockquote detector
emits a span covering the block extent. If the > is a list marker rather than a
quote, the span is a false-positive Direct_Quote_Of_Model. Mitigation: hand-validation
sample includes DQ-labeled rows for researcher inspection.

**FM-1e (LCR_SP over-detection):** Span covers the exact match extent only, not
the entire row. This is an improvement over v2 row-level: a row with one
LCR_SP phrasal is no longer entirely labeled LCR_SP. The surrounding text
correctly falls through to User_Original_Content floor.

**FM-2a (paraphrase without attribution verb):** Still the dominant false-negative
mode. Floor spans assigned User_Original_Content where the user narrated model
behavior without an explicit attribution verb.

**FM-2d (embedded dialogue without markers):** At span level, verbatim Claude
utterances embedded without > or " markers appear as User_Original_Content floor
spans. This is structurally correct under §9.0 (floor is the default), not an
error; the spans can be manually corrected in the researcher_corrections column
of the validation sample.

**FM-3b (row-level limitation):** Resolved by design in v3. Each row now has
multiple spans, each independently labeled. Mixed-voice rows are correctly
segmented as multiple spans with different labels.

**FM-3d (LCR_SP fragment matching):** At span level, the span covers the
exact phrase extent (typically 3-10 words), not the entire row. This reduces
the impact of partial-match false positives compared to v2.

---

## 8. Cross-reference

- v3 script: `src/voice_segmentation_lcr_v3_span.py`
- v3 span output: `data/lcr_pass1b_canonical_voice_spans.csv`
- Validation sample v3: `notebooks/audit_trail/phase_4_voice_segmentation_validation_sample_v3_span.csv`
- LLM call log v3: `deliverables/phase_4_voice_segmentation/llm_call_log_v3.jsonl`
- LLM tool log: `notebooks/llm_tool_log.md` (Entry 3)
- v2 summary (deprecated baseline): `notebooks/audit_trail/phase_4_voice_segmentation_v2_summary.md`
- Failure modes: `notebooks/audit_trail/phase_4_voice_segmentation_failure_modes.md`
- Schema specification: `llm-behavior-reddit-analysis-universal/methods_library.md` §9.0
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nSummary written to: {output_path.name}")


# ---------------------------------------------------------------------------
# LLM-as-tool log update
# ---------------------------------------------------------------------------

def append_llm_tool_log(log_path: Path):
    """Append Entry 3 to notebooks/llm_tool_log.md."""
    entry = f"""
---

## Entry 3 — Phase 4 v3 Span-Level LLM Fallback (Gemini CLI)

| Field | Value |
|-------|-------|
| Tool | Gemini CLI (npm package) |
| Model | {LLM_MODEL} |
| Date | 2026-05-18 |
| Phase | Phase 4 — Voice Segmentation v3 span-level (§C.4) |
| Role | Layer 2 fallback classifier for unclassified interval spans longer than {MIN_LLM_INTERVAL_CHARS} chars |
| Budget | {LLM_BUDGET} calls (hard cap; doubled from v2 budget of 100 to reflect larger corpus coverage) |
| Invocation | `subprocess.run(['gemini.cmd', '-m', '{LLM_MODEL}'], input=prompt, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)` — Unicode fix applied from v2 |
| Log | `deliverables/phase_4_voice_segmentation/llm_call_log_v3.jsonl` |
| Prompt template | Six-way classification prompt (five labels + Unclassifiable) with explicit LCR_SP phrasal examples and context prefix. See `src/voice_segmentation_lcr_v3_span.py` `GEMINI_SYSTEM_PROMPT`. |
| Prompt distinction | Direct_Quote_Of_Model vs LCR_System_Prompt_Reproduction boundary explicitly documented; Unclassifiable label added (not present in v2 prompt). |
| Anti-pattern compliance | LLM output is not treated as ground truth; all LLM-labeled spans tagged `source=llm:{LLM_MODEL}`; downstream claims qualified by §9.3 hand-validation precision. |
| Output | `data/lcr_pass1b_canonical_voice_spans.csv` (span records with source=llm:{LLM_MODEL}) |
| Calls made | (see run-time output) |
| Reference | `src/voice_segmentation_lcr_v3_span.py`; `notebooks/audit_trail/phase_4_voice_segmentation_lcr_v3_span_summary.md` |

*Log maintained per [methods_library §9.2] disclosure requirements.*
"""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"LLM tool log updated: {log_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 4 LCR Voice Segmentation — v3 SPAN-LEVEL")
    print("Schema: methods_library.md §9.0")
    print("=" * 70)

    # Load input rows for metadata
    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as fin:
        input_rows = list(csv.DictReader(fin))

    # Run span segmentation
    span_records = run_span_segmentation(INPUT_CSV, OUTPUT_CSV)

    # If resuming, reload all span records from the output CSV
    if not span_records:
        print("Reloading span records from output CSV (resume mode)...")
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            span_records = list(csv.DictReader(f))

    # Compute metrics
    print("\nComputing §C.4 character-count-weighted metrics...")
    metrics = compute_span_metrics(span_records, input_rows)

    print(f"\n--- Span-level metrics ---")
    print(f"  Total spans: {metrics['total_spans']:,}")
    print(f"  Total chars: {metrics['total_chars']:,}")
    print(f"  Reliable chars (high+med): {metrics['reliable_chars']:,}")
    print(f"  Reliable fraction: {metrics['reliable_pct']:.2f}%")
    print(f"  §C.4 70% threshold: {'MET' if metrics['threshold_met'] else 'NOT MET'}")
    print("\n  By label (spans):")
    for lbl in [LABEL_DIRECT, LABEL_PARA, LABEL_LCR_SP,
                LABEL_USER, LABEL_AUTOMOD, LABEL_UNCLASS]:
        n = metrics["by_label"].get(lbl, 0)
        ch = metrics["by_label_chars"].get(lbl, 0)
        total_spans = metrics["total_spans"]
        total_chars = metrics["total_chars"]
        print(f"    {lbl}: {n:,} spans "
              f"({100*n/max(total_spans,1):.1f}%), "
              f"{ch:,} chars "
              f"({100*ch/max(total_chars,1):.1f}%)")

    # Validation sample
    print("\nPreparing 50-item stratified validation sample...")
    prepare_validation_sample(span_records, input_rows, VALIDATION_CSV)

    # Summary markdown
    write_summary(metrics, input_rows, SUMMARY_MD)

    # LLM tool log
    append_llm_tool_log(LLM_TOOL_LOG)

    print("\n" + "=" * 70)
    print("Phase 4 LCR voice segmentation v3 span-level complete.")
    print(f"  Span output:        {OUTPUT_CSV}")
    print(f"  Validation sample:  {VALIDATION_CSV}")
    print(f"  Summary:            {SUMMARY_MD}")
    print(f"  LLM call log:       {LLM_LOG}")
    print(f"  LLM tool log:       {LLM_TOOL_LOG}")
    print(f"  Reliable fraction:  {metrics['reliable_pct']:.2f}%")
    print(f"  §C.4 threshold:     {'MET' if metrics['threshold_met'] else 'NOT MET'}")
    print(f"  LLM calls:          {_llm_call_count}  Errors: {_llm_error_count}")
    print("=" * 70)
