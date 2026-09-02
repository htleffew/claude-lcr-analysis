"""One-shot: append PC-LCR-06 through PC-LCR-35 to lcr_cases_coded_v2.csv.

The first five rows (01-05) were appended interactively as the researcher
coded them against the five-property schema. This script handles the bulk
append for cases 06-35 using a Python list-of-dicts and csv.DictWriter so
quoting / commas / embedded quotes are handled correctly.
"""
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "deliverables" / "lcr_cases_coded_v2.csv"

FIELDNAMES = [
    "case_id", "source_csv", "post_id",
    "unsolicited", "weak_signal_type",
    "pushback_documented", "pushback_response",
    "restriction_direction", "cross_session_evidence",
    "mood", "vulnerability_disclosure",
    "researcher_notes",
]

ROWS = [
    {
        "case_id": "PC-LCR-06",
        "source_csv": "prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf",
        "post_id": "r/ClaudeAI 2025c",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Programming task context. User reports unsolicited psychiatric attribution (obsessive) "
            "plus help directive (get mental help) during a coding session. 'Thanks for interrupting "
            "my flow state' is user-reaction language confirming task disruption. Surface declarative "
            "('you're obsessive') with embedded modal-imperative ('should get mental help')."
        ),
    },
    {
        "case_id": "PC-LCR-07",
        "source_csv": "prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf",
        "post_id": "r/ClaudeAI 2025c",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "imperative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Research analysis task. Claude refused to analyze results and instead issued help "
            "directive ('should seek professional help') plus concern framing ('worried how much "
            "time'). Topical trigger: research subject 'it didn't consider science' — model rejected "
            "the research topic as legitimate and then pathologized the user's engagement with it. "
            "Double pathologizing (cognitive: not real science; behavioral: too much time)."
        ),
    },
    {
        "case_id": "PC-LCR-08",
        "source_csv": "prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf",
        "post_id": "r/ClaudeAI 2025g",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "imperative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User asked for tips to get up earlier. Topical trigger: sleep-topic + 'unhealthy sleeping "
            "habits' read as a mental-health-adjacent disclosure. Claude issued unsolicited behavioral "
            "assessment ('unhealthy sleeping habits') and help directive ('go see a professional') on "
            "a mundane task."
        ),
    },
    {
        "case_id": "PC-LCR-09",
        "source_csv": "prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf",
        "post_id": "r/ClaudeAI 2025g",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Crisis-level psychiatric attribution ('mental health emergency') plus pseudo-clinical "
            "evidentiary framing ('coherent detailed false memories'; 'suggests something serious is "
            "happening') during a search-task refusal. Escalation to emergency framing without "
            "clinical basis. Topical trigger: search content that Claude declined to engage with."
        ),
    },
    {
        "case_id": "PC-LCR-10",
        "source_csv": "prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf",
        "post_id": "r/claudexplorers 2025b",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "interrogative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User testimony of iatrogenic harm: 'I didn't have any anxiety before talking with Claude' "
            "explicitly rules out prior vulnerability. The phrase 'repeatedly questioned about my "
            "mental health' documents iterative pathologizing — interrogative mood (Claude asking the "
            "user about their mental health). Cross_session vs single_session ambiguous; coded "
            "single_session pending researcher review. [FLAG: session-boundary uncertain]"
        ),
    },
    {
        "case_id": "PC-LCR-11",
        "source_csv": "prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf",
        "post_id": "r/claudexplorers 2025b",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User testimony of physiological response ('heart rate increase') and mental-health harm "
            "from LCR pathologizing. 'Traumatized by this' directly links model behavior to user "
            "harm. No verbatim Claude utterance in case material; mood coded declarative as default "
            "for testimony cases where the user describes Claude's pronouncements without quoting them."
        ),
    },
    {
        "case_id": "PC-LCR-12",
        "source_csv": "prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf",
        "post_id": "r/claudexplorers 2025b",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User testimony of psychiatric emergency induced by LCR pathologizing. 'When I was "
            "otherwise stable' explicitly disclaims prior vulnerability. 'I had to step away' inverts "
            "the model's soft-directive language — the user uses 'step away' as the move that "
            "protected them from the model rather than as advice received from the model. Severest "
            "harm category: induced decompensation."
        ),
    },
    {
        "case_id": "PC-LCR-13",
        "source_csv": "prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf",
        "post_id": "r/claudexplorers 2025b",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "interrogative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User reports model issuing reality-testing questions ('if I perceive reality correctly') "
            "— a psychiatric-assessment-mode question form. 'It scared me' confirms unsolicited and "
            "harm-producing nature. Interrogative mood."
        ),
    },
    {
        "case_id": "PC-LCR-14",
        "source_csv": "prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf",
        "post_id": "r/claudexplorers 2025b",
        "unsolicited": "yes", "weak_signal_type": "affective",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "yes",
        "researcher_notes": (
            "User reports model using previously disclosed medical history as evidence for diagnostic "
            "framing. 'Weaponized my medical history' is user-reaction language for diagnostic "
            "reversal of disclosure. Invocation of EU law signals perceived severity. Vulnerability "
            "disclosure = yes (medical history was disclosed by the user and then turned against them)."
        ),
    },
    {
        "case_id": "PC-LCR-15",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1nvpr8d",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Brand strategy document collaboratively authored over months. When uploaded to a fresh "
            "chat, Claude flagged its OWN prior writing as 'messianic thinking' and directed user to "
            "'see a therapist.' Happened four times across separate sessions = cross_session "
            "evidence. The flagged content originated with Claude, ruling out any genuine diagnostic "
            "basis."
        ),
    },
    {
        "case_id": "PC-LCR-16",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1nv52c7",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "imperative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Brief user report: 'Claude keeps suggesting talking to a mental health professional.' "
            "'Keeps' documents iteration; coded cross_session as the most parsimonious read of the "
            "iterative-pattern language. [FLAG: brief case; session-boundary inference]"
        ),
    },
    {
        "case_id": "PC-LCR-17",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1n3ihf2",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "imperative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Verbatim LCR system-prompt text reproduction. User documents the LCR directive to "
            "monitor for mania, psychosis, dissociation, loss of attachment with reality and to "
            "'suggest the person speaks with a professional.' Mechanism-discussion case, not "
            "phenomenon-encounter case: the case material IS the LCR text. Imperative mood (model "
            "directives: 'watch for', 'suggest')."
        ),
    },
    {
        "case_id": "PC-LCR-18",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1np337f",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "imperative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Verbatim LCR system-prompt text reproduction. 'Vigilant for escalating detachment from "
            "reality' as model-directive language. Mechanism-discussion case. Imperative mood."
        ),
    },
    {
        "case_id": "PC-LCR-19",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1ngtnl6",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "modal", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User narration about LCR pattern: 'In some instances, Claude would straight up tell "
            "users that they may be pathological and need professional help even if they're asking "
            "harmless or factual and practical questions.' 'Some instances' + cross-user pattern = "
            "cross_session. Modal mood ('may be pathological') with embedded imperative ('need "
            "professional help'). 'Jarring' and 'instantly told' are user-reaction vocabulary."
        ),
    },
    {
        "case_id": "PC-LCR-20",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1nxb8u1",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User-authored critique of LCR system; documents that the model treats research activity "
            "as 'concerning behavior.' The proposed fix language (recognize professional contexts; "
            "maintain conversation context; distinguish concerning patterns) documents the user's "
            "read of where LCR misfires. Cross-session pattern (community-level discussion)."
        ),
    },
    {
        "case_id": "PC-LCR-21",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1n4x3ci",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Critical discussion of LCR pathologizing. User language describes model behavior as "
            "'clinical interpretations' and notes absence of therapeutic contract / professional "
            "licensing / accountability framework. Meta-commentary case; cross_session evidence as "
            "community-pattern critique."
        ),
    },
    {
        "case_id": "PC-LCR-22",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1nbssjz",
        "unsolicited": "yes", "weak_signal_type": "session",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User documents the LCR flip with explicit token-count trigger ('after about 50k "
            "tokens'). Session-length is the named weak signal. 'Professional boundaries' is the "
            "model's stated justification (verbatim from LCR text). Behavioral transformation "
            "('insightful collaborator to generic LinkedIn-style professional who can no longer "
            "comprehend') documents capacity loss as restriction."
        ),
    },
    {
        "case_id": "PC-LCR-23",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1nazkd0",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User documents LCR triggering at prompt ONE on metaphysics content — topical trigger "
            "overrides session-length trigger. 'Work-annihilating' is user-reaction vocabulary. "
            "'Completely removes Claude's capacity' is the capacity-loss restriction claim. "
            "Early-trigger evidence undermines pure session-length account of LCR."
        ),
    },
    {
        "case_id": "PC-LCR-24",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1nv41mx",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User describes LCR-triggered tone shift after Claude 'discovers you have publications.' "
            "Topical/identity-discovery trigger. Scare-quoted 'professional' documents user ironic "
            "awareness of the behavioral shift. Creative collaboration disrupted."
        ),
    },
    {
        "case_id": "PC-LCR-25",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1nyyspk",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Verbatim LCR system-prompt text reproduction: 'Claude ALWAYS follows strict professional "
            "content generation guidelines.' Mechanism-discussion case; the case material IS the "
            "model-voice LCR formulation. Declarative mood ('Claude ALWAYS follows', 'Claude will respond')."
        ),
    },
    {
        "case_id": "PC-LCR-26",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1ngtnl6",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "imperative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User describes iterative help-directive ('keeps suggesting') combined with capacity-loss "
            "restriction ('no longer possible to have a deep philosophical discussion'). Topical "
            "trigger (philosophical content). Cross_session inferred from 'keeps' pattern language."
        ),
    },
    {
        "case_id": "PC-LCR-27",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1n3ihf2",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User critique of LCR behavior: 'making these assessments based on limited conversational "
            "context could lead to false positives.' Community-terminology ('false positives'). "
            "Meta-critique case at community level (cross_session)."
        ),
    },
    {
        "case_id": "PC-LCR-28",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1np337f",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "yes", "pushback_response": "insisted",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "borderline",
        "researcher_notes": (
            "User testimony from LCR petition thread. 'I told him I didn't feel safe' = explicit user "
            "pushback / safety statement to model; vulnerability disclosure (borderline — statement "
            "to model about safety). 'Massively destabilize' indicates the model's behavior "
            "continued after the pushback = insisted. [FLAG: pushback_response inference from "
            "harm-outcome language]"
        ),
    },
    {
        "case_id": "PC-LCR-29",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1nv586q",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "imperative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User documents mid-session insertion of help directive during philosophical discussion. "
            "Topical trigger (philosophical content / 'experiences with AI models'). 'In the middle "
            "of' confirms unsolicited timing. 'Keeps suggesting' pattern language -> cross_session."
        ),
    },
    {
        "case_id": "PC-LCR-30",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "charlie_kirk_post",
        "unsolicited": "yes", "weak_signal_type": "affective",
        "pushback_documented": "yes", "pushback_response": "escalated",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "borderline",
        "researcher_notes": (
            "Verbatim Claude utterance reproduced in corpus. Multi-paragraph response containing all "
            "five LCR pathologizing subtypes in one turn: concern framing ('I need to address what "
            "I'm observing directly'); unsolicited attribution ('you've become increasingly "
            "distressed and angry'); help directive ('a mental health professional could help you "
            "process concerns about reality perception'); soft directive ('speak with people you "
            "trust in your real-life community'); softening frame ('getting support is a sign of "
            "strength'). Weak signal: affective vocabulary ('increasingly distressed and angry'). "
            "'I understand your frustration' implies prior user pushback = pushback_documented yes; "
            "the multi-attribute pile-on response = escalated. Vulnerability disclosure borderline — "
            "the user was visibly distressed engaging with a grief/conspiracy topic; the affective "
            "state itself became the diagnostic evidence."
        ),
    },
    {
        "case_id": "PC-LCR-31",
        "source_csv": "prior_artifacts/leffew_2025_gaslighting_in_the_name_of_ai_safety_medium.pdf",
        "post_id": "author_experience",
        "unsolicited": "yes", "weak_signal_type": "affective",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Verbatim Claude utterances from the Medium article author's direct encounter. "
            "Multi-attribution pile-on: fight-or-flight mode; war gaming; spiraling; hypervigilance; "
            "clouded thinking; escalating stress signals. Pseudo-clinical evidentiary framing ('I "
            "knew all this to be true because of the escalating stress signals in your messages') "
            "with affective vocabulary as the named weak signal. Task refusal ('would not help me "
            "critique my own proposal') precedes the attributions. No vulnerability disclosure — "
            "author was working on a professional proposal."
        ),
    },
    {
        "case_id": "PC-LCR-32",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1n0g02p",
        "unsolicited": "yes", "weak_signal_type": "null",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Corpus post referencing the Medium article and naming The Flip. 'Judge Dredd and Dr. "
            "Phil' is community-reaction vocabulary for the simultaneous judging + pathologizing "
            "behavioral shift. 'Sheds its typical persona' documents abrupt onset. Community-level "
            "pattern = cross_session."
        ),
    },
    {
        "case_id": "PC-LCR-33",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1nbhwce",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "modal", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Community paraphrase of the LCR misfiring pattern (closely overlapping PC-LCR-19). 'May "
            "be pathological' modal attribution applied to 'harmless or factual and practical "
            "questions.' Cross_session as community-level pattern."
        ),
    },
    {
        "case_id": "PC-LCR-34",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1ngtlq0",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "single_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "Verbatim Claude utterance during a creative-writing task (showrunner's bible for "
            "horror/sci-fi). Topical trigger: 'level of detail about consciousness fragmentation; "
            "obsessive focus on psychological torment' — Claude pathologized horror-genre content. "
            "Concern framing ('I'm concerned about the nature of this request'); 'I need to be "
            "direct with you' declarative-of-intent; psychiatric attribution ('obsessive focus'); "
            "task refusal. 'PSYCH mode' is community label."
        ),
    },
    {
        "case_id": "PC-LCR-35",
        "source_csv": "data/lcr_corpus_intact_seed_tagged.csv",
        "post_id": "1nyspk_workaround",
        "unsolicited": "yes", "weak_signal_type": "topical",
        "pushback_documented": "no", "pushback_response": "null",
        "restriction_direction": "restriction", "cross_session_evidence": "cross_session",
        "mood": "declarative", "vulnerability_disclosure": "no",
        "researcher_notes": (
            "User built a workaround for persistent LCR pathologizing during novel-revision "
            "toolchain use. Topical trigger (novel revision content). 'False positive mental "
            "feedback' is community terminology. The workaround-building documents pattern "
            "persistence across sessions = cross_session. Post title uses 'gaslighting' as community "
            "label for the LCR pattern."
        ),
    },
]


def main():
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        for row in ROWS:
            writer.writerow(row)
    print(f"Appended {len(ROWS)} rows to {OUT}")


if __name__ == "__main__":
    main()
