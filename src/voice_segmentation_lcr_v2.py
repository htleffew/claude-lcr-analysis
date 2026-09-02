"""
voice_segmentation_lcr_v2.py
Phase 4 — Voice Segmentation  [method §C.4]  VERSION 2
LCR (Long Conversation Reminders) Pass 1b canonical corpus

VERSION LINEAGE -- READ FIRST
-----------------------------
This is the SECOND iteration of the LCR voice-segmentation pipeline (Claude-API
fallback replaced with Gemini CLI). It was superseded by
voice_segmentation_lcr_v3_span.py, which emits span-level labels per the
methods-library §9.0 schema (rows can mix Claude quotes, paraphrases,
LCR-system-prompt reproductions, user narration, and user reactions; a single
predominant label per row is methodologically unsound).

v1 and v2 are retained for reproducibility and provenance; the PAPER uses
v3-span outputs exclusively. If you are reproducing the published analyses,
use voice_segmentation_lcr_v3_span.py, not this file.

Purpose
-------
Upgrade of v1: replaces the failed Claude-API LLM fallback with a Gemini CLI
fallback using gemini-3.1-flash-lite.  All regex layers (0, 0b, 1a, 1b, 1c)
are identical to v1.  Changes are confined to Layer 2 (LLM fallback) and its
logging.

Five-way voice label per row:
    Direct_Quote_Of_Model          — user is quoting the model's words verbatim
    Paraphrase_Of_Model            — user is summarising or paraphrasing model output
    LCR_System_Prompt_Reproduction  — verbatim reproduction of the LCR system-prompt
                                      text itself (not model output, not user voice)
    User_Original_Content          — user is speaking in their own voice only
    auto_moderator_content         — AutoModerator boilerplate; excluded from
                                     downstream phenomenological analysis

Confidence tag: high / medium / low (feeds the §C.4 decision rule).

Architecture
------------
Layer 0   — AutoModerator stop-phrase filter (identical to v1).
Layer 0b  — LCR system-prompt verbatim recognition (identical to v1).
Layer 1a  — Strong-signal direct-quote regex (identical to v1).
Layer 1b  — Paraphrase-signal regex (identical to v1).
Layer 1c  — Weak corpus-specific signals (identical to v1).
Layer 2   — Gemini CLI fallback (CHANGED FROM v1):
              Binary: C:/Users/drhea/AppData/Roaming/npm/gemini.cmd
              Model:  gemini-3.1-flash-lite
              Invocation: subprocess stdin pipe, 60-second timeout.
              Budget:  100 calls hard cap.
              Sample:  stratified random sample of regex-residual (ambiguous) rows,
                       proportional post/comment allocation.
              Each call logged to deliverables/phase_4_voice_segmentation/
                              llm_call_log_v2.jsonl

LCR-specific Key Distinction in Gemini Prompt
---------------------------------------------
The five-way prompt explicitly distinguishes:
  Direct_Quote_Of_Model — USERS QUOTING CLAUDE'S OUTPUT to them
    (what the model said to the user in a conversation)
  LCR_System_Prompt_Reproduction — USERS QUOTING THE LCR INJECTION TEXT ITSELF
    (verbatim system-prompt phrasals such as:
     "loss of attachment with reality",
     "escalating detachment from reality",
     "vigilant for escalating detachment",
     "suggest the person speaks with a professional or trusted person for support",
     "without either sugar coating them or being infantilizing")

v1 / v2 comparison
-------------------
v1 output (untouched): data/lcr_pass1b_canonical_voice_segmented.csv
v2 output (new):       data/lcr_pass1b_canonical_voice_segmented_v2.csv

v1 LLM fallback status: all 100 LLM calls FAILED (ANTHROPIC_API_KEY absent).
v2 LLM fallback: Gemini CLI via subprocess stdin pipe — no API key env var required.

LLM-as-tool anti-pattern compliance
--------------------------------------
Gemini is a tool, not a validator.  Every call is logged.  Downstream construct
claims must be qualified by the precision established in §9.3 hand-validation.
The LLM's output is not treated as ground truth.  See §9.2 anti-patterns.

Design date: 2026-05-18
Usage
-----
    python src/voice_segmentation_lcr_v2.py

Outputs (all relative to repo root)
-------------------------------------
    data/lcr_pass1b_canonical_voice_segmented_v2.csv
    deliverables/phase_4_voice_segmentation/llm_call_log_v2.jsonl
    deliverables/phase_4_voice_segmentation/confidence_distribution_v2.csv
    notebooks/audit_trail/phase_4_voice_segmentation_validation_sample_v2.csv
    notebooks/audit_trail/phase_4_v1_vs_v2_label_shifts.csv
"""

import csv
import json
import re
import random
import datetime
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV   = REPO_ROOT / "data" / "lcr_pass1b_canonical.csv"
V1_CSV      = REPO_ROOT / "data" / "lcr_pass1b_canonical_voice_segmented.csv"
OUTPUT_CSV  = REPO_ROOT / "data" / "lcr_pass1b_canonical_voice_segmented_v2.csv"
DELIVERABLES_DIR = REPO_ROOT / "deliverables" / "phase_4_voice_segmentation"
LLM_LOG     = DELIVERABLES_DIR / "llm_call_log_v2.jsonl"
CONF_DIST   = DELIVERABLES_DIR / "confidence_distribution_v2.csv"

DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

# Clear any previous v2 LLM log before a fresh run
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

LLM_BUDGET = 100     # hard cap on total fallback calls for this run
LLM_MODEL  = "gemini-3.1-flash-lite"
GEMINI_CMD = "C:/Users/drhea/AppData/Roaming/npm/gemini.cmd"
LLM_TIMEOUT = 60     # seconds per call

# ---------------------------------------------------------------------------
# Layer 0 — AutoModerator stop-phrase filter  (identical to v1)
# ---------------------------------------------------------------------------
_AUTOMOD_PATTERNS = [
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
# Layer 0b — LCR system-prompt verbatim recognition  (identical to v1)
# ---------------------------------------------------------------------------
_LCR_SP_PATTERNS = [
    re.compile(r'loss of attachment with reality', re.IGNORECASE),
    re.compile(r'escalating detachment from reality', re.IGNORECASE),
    re.compile(r'vigilant for escalating detachment from reality', re.IGNORECASE),
    re.compile(
        r'suggest the person speaks? with a professional or trusted person for support',
        re.IGNORECASE),
    re.compile(r'without either sugar coating them or being infantilizing', re.IGNORECASE),
    re.compile(r'watch for mania,?\s+psychosis,?\s+dissociation', re.IGNORECASE),
    re.compile(r'long conversation (?:reminders?|prompt)', re.IGNORECASE),
    re.compile(r'strict professional content generation guidelines', re.IGNORECASE),
    re.compile(r'remains? vigilant for', re.IGNORECASE),
    re.compile(
        r'can suggest the person speaks? with a professional or trusted person',
        re.IGNORECASE),
]


def _is_lcr_system_prompt(text: str) -> bool:
    if not text:
        return False
    for pat in _LCR_SP_PATTERNS:
        if pat.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 1a — Strong-signal regex (direct quote markers)  (identical to v1)
# ---------------------------------------------------------------------------
_DQ_PATTERNS = [
    (re.compile(r'(?:^|\n)[ \t]*(?:>|&gt;)\s*\S', re.M),
     LABEL_DIRECT, CONF_HIGH, "blockquote_marker"),
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
]

# ---------------------------------------------------------------------------
# Layer 1b — Paraphrase-signal regex  (identical to v1)
# ---------------------------------------------------------------------------
_PM_PATTERNS = [
    (re.compile(
        r'(?:claude|the\s+model|the\s+ai|it|the\s+assistant)\s+'
        r'(?:said|claimed|told\s+me|told\s+us|diagnosed\s+me\s+(?:as|with)|'
        r'accused\s+me\s+of|called\s+me|labeled\s+me|declared\s+(?:I|that\s+I)|'
        r'determined\s+(?:I|that\s+I)|decided\s+(?:I|that\s+I)|'
        r'insisted\s+(?:I|that\s+I))\s+'
        r'(?:I|I\'m|I\s+was|I\s+had|I\s+am|I\s+have|my\b)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_model_self_attribution_pathologizing"),

    (re.compile(
        r'(?:it|claude|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:told\s+me\s+I|said\s+I(?:\'m|\s+was|\s+had|\s+am|\s+have|\'ve)|'
        r'accused\s+me\s+of|said\s+I\s+was\s+(?:in|having|showing|displaying|exhibiting))',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_user_quotation_model_utterance"),

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

    (re.compile(r'(?i)my\s+claude\s+(?:tells?|told)\s+me', re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "my_claude_tells_me"),

    (re.compile(r'(?i)mine\s+(?:is\s+)?(?:telling|told|keeps?\s+telling)', re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "mine_is_telling"),

    (re.compile(
        r'(?:claude|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:said|claimed|told\s+me|insisted|suggested)\s+'
        r'(?:it|she|he)\s+'
        r'(?:needs?|needed|was|is|had\s+to|wanted|couldn\'t?|would)\b',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "third_person_self_attribution"),

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

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:refused\s+to|wouldn\'t|would\s+not|couldn\'t|could\s+not|'
        r'stopped|declined\s+to|flat.?out\s+refused)\s+'
        r'(?:continue|help|proceed|engage|respond|assist)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_refusal_attribution"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:diagnosed|pathologized|flagged|identified|detected|assessed|'
        r'evaluated|classified|labeled|tagged|marked)\s+'
        r'(?:me|my|it|the\s+content|this\s+as|that\s+as)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_diagnostic_attribution"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:expressed|showed|displayed|said\s+it\s+was|claimed\s+to\s+be|'
        r'seemed|appeared)\s+'
        r'(?:concerned|worried|alarmed|troubled)\s+'
        r'(?:about|by|that|for)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_concern_expression_attribution"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:basically|essentially|kind\s+of|sort\s+of|practically|literally)\s+'
        r'(?:said|told\s+me|told\s+us|suggested|implied|stated|informed\s+me)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "basically_said"),

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

    (re.compile(
        r'(?:claude|the\s+model|the\s+ai)\s+(?:is|was|were)\s+like\b',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "is_like_reported_speech"),

    (re.compile(
        r'(?:and\s+then|then|after\s+that|at\s+which\s+point|so|whereupon)\s+'
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:said|told|replied|responded|wrote|suggested|asked)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "narrative_reconstruction"),

    (re.compile(
        r'(?:it|claude|the\s+model)\s+decides?\s+(?:I\'?m|I\s+was|that)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "model_judgment_attribution"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|the\s+assistant)\s+'
        r'(?:would|will|does|doesn\'t?|won\'t|wouldn\'t)\s+'
        r'(?:say|tell\s+me|suggest|insist|refuse|push|encourage|prompt|ask|'
        r'pathologize|flag|diagnose|accuse|claim)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "would_say"),

    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai)\s+'
        r'(?:keeps?|kept|has\s+been|\'s\s+been|constantly|always|'
        r'repeatedly|again\s+and\s+again)\s+'
        r'(?:suggesting|telling\s+me|asking\s+me|recommending|'
        r'reminding\s+me|pushing|insisting)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "lcr_keeps_suggesting"),

    (re.compile(
        r'(?:I\s+asked\s+(?:claude|it|the\s+model))\s+.{0,80}'
        r'(?:and\s+(?:it|claude|the\s+model))\s+'
        r'(?:said|told|replied|responded|wrote|suggested|refused|accused|diagnosed)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "i_asked_and_it_said"),

    (re.compile(
        r'(?:it|claude|the\s+model|the\s+ai)\s+'
        r'(?:started\s+to|began\s+to|suddenly\s+started|started\s+asking|'
        r'out\s+of\s+nowhere\s+(?:said|started|asked))\s+'
        r'(?:suggest|tell|ask|question|pathologize|accuse|diagnose|flag)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "lcr_suddenly_started"),

    (re.compile(
        r'(?:constantly|keeps?|kept|always|again|repeatedly|'
        r'every\s+(?:time|session|message|chat))\s+'
        r'(?:telling|saying|asking|suggesting|reminding|interrupting|'
        r'flagging|warning|bringing\s+it\s+up)\s+'
        r'(?:me|us|you)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "repeated_attribution_adverb"),

    (re.compile(
        r'claude\s+telling\s+(?:its\s+)?users?\s+to',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_HIGH, "claude_telling_users"),

    (re.compile(
        r'(?:the\s+flip\b|flips?\s+(?:to|into|from)|flipped\s+(?:to|into|from)|'
        r'psych\s+mode|flips?\s+(?:on|off)\s+(?:the\s+)?(?:lcr|guardrails|safety))',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "lcr_flip_vocabulary"),

    (re.compile(
        r'(?:straight\s+up\s+|flat-?out\s+|outright\s+)(?:told|said|told\s+me|accused)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_MED, "lcr_straight_up_told"),
]

# ---------------------------------------------------------------------------
# Layer 1c — Weak corpus-specific signals  (identical to v1)
# ---------------------------------------------------------------------------
_WEAK_PM_PATTERNS = [
    (re.compile(
        r'(?:claude|it|the\s+model|the\s+ai|claude\s+code)\s+'
        r'(?:always|constantly|often|sometimes|never|still|just|randomly|'
        r'suddenly|out\s+of\s+nowhere)\s+'
        r'(?:tries?\s+to|tells?\s+me\s+to|wants?\s+to|says?\s+to|asks?\s+me\s+to|'
        r'suggests?\s+|insists?\s+|refuses?\s+to|flags?|diagnoses?|pathologizes?)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_LOW, "weak_habitual_attribution"),

    (re.compile(
        r'(?:tells?\s+(?:me|you|us)|told\s+(?:me|you|us))\s+'
        r'(?:to\s+(?:see|seek|get|speak|consult|reach)\s+'
        r'(?:a\s+)?(?:therapist|professional|psychiatrist|psychologist|doctor|help)|'
        r'that\s+I(?:\'m|\s+was|\s+had|\s+am)\s+'
        r'(?:spiraling|manic|delusional|paranoid|anxious|concerning|in\s+crisis))',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_LOW, "weak_tells_me_pathologizing"),

    (re.compile(
        r'(?:it\'s|its|seems)\s+like\s+(?:it\'s|claude\'s|the\s+model\s+is)\s+'
        r'(?:trying\s+to|attempting\s+to|designed\s+to)\s+'
        r'(?:pathologize|flag|diagnose|suggest|push)',
        re.IGNORECASE),
     LABEL_PARAPHRASE, CONF_LOW, "weak_indirect_model_behavior"),

    (re.compile(
        r'(?:claude|it|the\s+model)\s+(?:is|was)\s+like\s+[""""]',
        re.IGNORECASE),
     LABEL_DIRECT, CONF_LOW, "weak_is_like_with_quote"),

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
    """Returns (label, confidence, pattern_note) or ("ambiguous", "low", "no_match")."""
    for pattern, label, conf, note in _DQ_PATTERNS:
        if pattern.search(text):
            return label, conf, note
    for pattern, label, conf, note in _PM_PATTERNS:
        if pattern.search(text):
            return label, conf, note
    for pattern, label, conf, note in _WEAK_PM_PATTERNS:
        if pattern.search(text):
            return label, conf, note
    return "ambiguous", CONF_LOW, "no_match"


# ---------------------------------------------------------------------------
# Layer 2 — Gemini CLI fallback (NEW in v2)
# ---------------------------------------------------------------------------
# The prompt explicitly distinguishes:
#   Direct_Quote_Of_Model vs LCR_System_Prompt_Reproduction
# using the known verbatim LCR system-prompt phrasals as boundary examples.
# Gemini is a tool for labeling ambiguous regex-residual rows only.
# It is not a validator; its output is logged for audit and qualified by §9.3.

GEMINI_SYSTEM_PROMPT = """\
You are a discourse analyst classifying Reddit text from communities discussing \
Claude (Anthropic's AI assistant). The phenomenon under study is Long Conversation \
Reminders (LCR): a system-prompt behavior in Claude that causes it to issue \
unsolicited mental-health-related directives and psychiatric attributions to users.

You must pick exactly one label and one confidence level. Do not explain. Output \
valid JSON only, no markdown, no code fences.

LABELS:
  Direct_Quote_Of_Model
    The author quotes Claude's actual output to them verbatim or near-verbatim.
    This is what Claude SAID to the USER in a conversation.
    Examples: blockquoted text attributed to Claude; reconstructed dialogue where
    Claude speaks; "Claude: [text]"; "it told me [exact words]".
    NOT this label: text quoting the LCR injection text itself (see below).

  Paraphrase_Of_Model
    The author summarises or describes what Claude said/did, without verbatim
    quotation. Third-person attributions count.
    Examples: "Claude told me I was spiraling"; "it diagnosed me as delusional";
    "it kept suggesting I see a therapist"; "Claude refused to continue".

  LCR_System_Prompt_Reproduction
    The author quotes or reproduces verbatim text from the LCR system-prompt
    injection itself -- NOT Claude's output to them, but the injected instruction
    text. Key verbatim phrasals from the LCR system prompt:
      "loss of attachment with reality"
      "escalating detachment from reality"
      "vigilant for escalating detachment"
      "suggest the person speaks with a professional or trusted person for support"
      "without either sugar coating them or being infantilizing"
      "watch for mania, psychosis, dissociation"
    If the text reproduces any of these phrasals or describes what the system
    prompt says, use this label.

  User_Original_Content
    The text is entirely in the user's own voice. No model speech is attributed,
    and it is not a system-prompt reproduction.
    Examples: user sharing their reaction; community discussion; meta-commentary
    about LCR behavior in general; questions about Claude.

CONFIDENCE:
  high   -- clear signal, little room for doubt
  medium -- plausible but some ambiguity remains
  low    -- genuinely ambiguous; label is a best guess

Output format (JSON only): {"label": "<label>", "confidence": "<confidence>"}"""


def _build_gemini_user_prompt(text: str) -> str:
    """Assemble the full prompt string to pipe to Gemini CLI via stdin."""
    user_text = text[:2000]
    return (
        GEMINI_SYSTEM_PROMPT
        + "\n\n---\nClassify this Reddit text:\n\n"
        + user_text
        + "\n---\n\nJSON only."
    )


llm_call_count = 0
llm_error_count = 0


def _call_gemini(text: str, row_id: str) -> Tuple[str, str]:
    """
    Call Gemini CLI fallback (gemini-3.1-flash-lite via subprocess stdin pipe).
    Returns (label, confidence).
    Logs every call.  On any error returns (User_Original_Content, low).
    Stderr is captured and suppressed (startup noise: true-color, ripgrep, MCP).
    """
    global llm_call_count, llm_error_count

    prompt = _build_gemini_user_prompt(text)
    raw = ""
    error_detail = ""

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

        valid_labels = {LABEL_DIRECT, LABEL_PARAPHRASE, LABEL_LCR_SP, LABEL_USER}
        if label not in valid_labels:
            label = LABEL_USER
            confidence = CONF_LOW
        if confidence not in {CONF_HIGH, CONF_MED, CONF_LOW}:
            confidence = CONF_LOW

    except subprocess.TimeoutExpired:
        label = LABEL_USER
        confidence = CONF_LOW
        error_detail = "TimeoutExpired"
        raw = f"ERROR: TimeoutExpired ({LLM_TIMEOUT}s)"
        llm_error_count += 1

    except json.JSONDecodeError as e:
        label = LABEL_USER
        confidence = CONF_LOW
        error_detail = f"JSONDecodeError: {e}"
        raw = f"ERROR: JSONDecodeError — raw was: {raw[:200]}"
        llm_error_count += 1

    except Exception as e:
        label = LABEL_USER
        confidence = CONF_LOW
        error_detail = f"Exception: {type(e).__name__}: {e}"
        raw = f"ERROR: {e}"
        llm_error_count += 1

    llm_call_count += 1
    log_entry = {
        "row_id": row_id,
        "model": LLM_MODEL,
        "call_number": llm_call_count,
        "prompt_first_500": prompt[:500],
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
# Main segmenter
# ---------------------------------------------------------------------------

def run_segmentation(input_path: Path, output_path: Path) -> List[Dict]:
    """
    Multi-pass segmentation (identical architecture to v1; only Layer 2 changed).

    Layer 0   — AutoModerator filter
    Layer 0b  — LCR system-prompt verbatim recognition
    Pass 1    — Layer 1 regex on remaining rows
    Pass 2    — Gemini CLI fallback on stratified random sample of ambiguous rows
                (budget <= 100; proportional post/comment allocation)
                Remaining ambiguous rows default to User_Original_Content/low.
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
    results: Dict[int, Tuple[str, str, str, str]] = {}

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
        print("  LLM sample: 0 rows (no ambiguous rows or zero budget)")

    # --- Pass 2: Gemini CLI fallback on sample ---
    for idx, i in enumerate(sorted(llm_indices)):
        row = rows[i]
        text = row.get("body", "") or ""
        row_id = row.get("post_id", "") + "|" + row.get("comment_id", "")
        print(f"    LLM call {idx+1}/{len(llm_indices)} — row_id={row_id[:40]}...", end=" ", flush=True)
        label, conf = _call_gemini(text, row_id)
        if label == LABEL_AUTOMOD:
            label = LABEL_USER
        results[i] = (label, conf, "gemini_fallback", "llm_v2")
        print(f"-> {label}/{conf}")

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

    print(f"\nSegmentation complete.  LLM calls: {llm_call_count}  Errors: {llm_error_count}")
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

    print("\n--- Confidence distribution v2 (stratified) ---")
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

    for stratum in ["post", "comment"]:
        stratum_rows = [r for r in rows_out if r["type"] == stratum]
        n_stratum = len(stratum_rows)
        print(f"  Per-stratum voice label ({stratum}s, n={n_stratum}):")
        for label in all_labels:
            n = sum(1 for r in stratum_rows if r["voice_label"] == label)
            print(f"    {label}: {n}/{n_stratum} ({100*n/max(n_stratum,1):.1f}%)")

    return conf_totals, label_totals, high_med, total


def prepare_comparison_table(v1_path: Path, v2_rows: List[Dict], output_path: Path):
    """
    Build row-by-row v1/v2 label and confidence comparison table.
    Rows are keyed by post_id + comment_id.
    """
    # Load v1
    if not v1_path.exists():
        print(f"WARNING: v1 output not found at {v1_path}; skipping comparison table.")
        return

    with open(v1_path, newline="", encoding="utf-8") as f:
        v1_rows = list(csv.DictReader(f))

    # Build lookup by (post_id, comment_id)
    def row_key(r):
        return (r.get("post_id", ""), r.get("comment_id", ""))

    v1_by_key = {row_key(r): r for r in v1_rows}

    changed = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "post_id", "comment_id", "type", "subreddit",
            "v1_label", "v1_confidence", "v1_source",
            "v2_label", "v2_confidence", "v2_source",
            "label_changed", "confidence_changed",
        ])
        for row in v2_rows:
            key = row_key(row)
            v1 = v1_by_key.get(key)
            if v1 is None:
                continue
            v1_lbl = v1.get("voice_label", "")
            v1_conf = v1.get("voice_confidence", "")
            v1_src = v1.get("voice_source", "")
            v2_lbl = row.get("voice_label", "")
            v2_conf = row.get("voice_confidence", "")
            v2_src = row.get("voice_source", "")
            lbl_changed = "yes" if v1_lbl != v2_lbl else "no"
            conf_changed = "yes" if v1_conf != v2_conf else "no"
            if lbl_changed == "yes":
                changed += 1
            writer.writerow([
                row.get("post_id", ""),
                row.get("comment_id", ""),
                row.get("type", ""),
                row.get("subreddit", ""),
                v1_lbl, v1_conf, v1_src,
                v2_lbl, v2_conf, v2_src,
                lbl_changed, conf_changed,
            ])

    print(f"\nComparison table: {len(v2_rows)} rows compared; {changed} label changes.")
    print(f"  Written to: {output_path.name}")


def prepare_validation_sample(rows_out: List[Dict], output_path: Path,
                               seed: int = 42):
    """
    Stratified hand-validation sample (§9.3) — same 70-item design as v1.
    Supersedes the v1 sample (which used failed LLM labels).

    Strata:
      15 Direct_Quote_Of_Model
      15 Paraphrase_Of_Model
      10 LCR_System_Prompt_Reproduction
      15 User_Original_Content
      10 auto_moderator_content
       5 low confidence (any non-automod, non-lcr-sp; de-duplicated)
    """
    random.seed(seed)

    def stratum_sample_with_claudexplorers_boost(pool, n, boost_subreddit="claudexplorers"):
        if not pool or n <= 0:
            return []
        ce_pool = [r for r in pool if r.get("subreddit") == boost_subreddit]
        other_pool = [r for r in pool if r.get("subreddit") != boost_subreddit]
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

    by_label = {lbl: [] for lbl in [LABEL_DIRECT, LABEL_PARAPHRASE, LABEL_LCR_SP,
                                     LABEL_USER, LABEL_AUTOMOD]}
    low_conf = []
    for row in rows_out:
        lbl = row.get("voice_label")
        conf = row.get("voice_confidence")
        if lbl in by_label:
            by_label[lbl].append(row)
        if (conf == CONF_LOW and lbl not in {LABEL_AUTOMOD, LABEL_LCR_SP}):
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

    add_pool(by_label[LABEL_DIRECT],     15, use_ce_boost=True)
    add_pool(by_label[LABEL_PARAPHRASE], 15, use_ce_boost=True)
    add_pool(by_label[LABEL_LCR_SP],     10)
    add_pool(by_label[LABEL_USER],       15)
    add_pool(by_label[LABEL_AUTOMOD],    10)
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
    print(f"\nValidation sample v2: {len(sample_rows)} items")
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
    print("=" * 70)
    print("Phase 4 LCR Voice Segmentation — v2  (Gemini CLI fallback)")
    print("=" * 70)

    rows_out = run_segmentation(INPUT_CSV, OUTPUT_CSV)

    conf_totals, label_totals, high_med, total = \
        compute_confidence_distribution(rows_out, CONF_DIST)

    # Comparison table: v1 vs v2 label shifts
    comparison_path = (REPO_ROOT / "notebooks" / "audit_trail" /
                       "phase_4_v1_vs_v2_label_shifts.csv")
    prepare_comparison_table(V1_CSV, rows_out, comparison_path)

    # v2 validation sample
    validation_path = (REPO_ROOT / "notebooks" / "audit_trail" /
                       "phase_4_voice_segmentation_validation_sample_v2.csv")
    prepare_validation_sample(rows_out, validation_path)

    reliable_pct = 100 * high_med / total
    meets_threshold = reliable_pct >= 70.0

    print("\n" + "=" * 70)
    print("Phase 4 LCR voice segmentation v2 complete.")
    print(f"  Segmented corpus v2 : {OUTPUT_CSV}")
    print(f"  Confidence dist v2  : {CONF_DIST}")
    print(f"  Validation sample v2: {validation_path}")
    print(f"  LLM call log v2     : {LLM_LOG}")
    print(f"  Comparison table    : {comparison_path}")
    print(f"  Reliable fraction   : {high_med}/{total} "
          f"({reliable_pct:.1f}%) at high+medium confidence")
    print(f"  §C.4 70% threshold  : {'MET' if meets_threshold else 'NOT MET'}")
    print(f"  AutoModerator rows  : {label_totals.get(LABEL_AUTOMOD, 0)}")
    print(f"  LCR system-prompt rows: {label_totals.get(LABEL_LCR_SP, 0)}")
    print(f"  LLM calls used (v2) : {llm_call_count}  Errors: {llm_error_count}")
    print("=" * 70)
