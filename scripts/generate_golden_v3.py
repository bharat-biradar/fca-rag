"""Generate golden evaluation dataset v3 with reference_chunks.

Loads all 10 sourcebook JSON files from data/parsed_v2/, defines 40 questions,
and programmatically populates reference_chunks with exact rule text from source.

Usage: python3 -m scripts.generate_golden_v3
"""

from __future__ import annotations

import json
from pathlib import Path

PARSED_DIR = Path("data/parsed_v2")
OUTPUT_PATH = Path("data/golden/golden_qa_v3.json")

SOURCEBOOKS = [
    "BCOBS", "CASS", "CMCOB", "COBS", "ESG",
    "FPCOB", "ICOBS", "MAR", "MCOB", "PDCOB",
]


def load_all_rules() -> dict[str, dict]:
    """Load all rules from parsed_v2 into a dict keyed by rule_id."""
    all_rules: dict[str, dict] = {}
    for sb in SOURCEBOOKS:
        path = PARSED_DIR / f"{sb}_rules.json"
        with open(path) as f:
            rules = json.load(f)
        for r in rules:
            all_rules[r["rule_id"]] = r
    print(f"Loaded {len(all_rules)} rules from {len(SOURCEBOOKS)} sourcebooks")
    return all_rules


# ---------------------------------------------------------------------------
# All 40 questions
# ---------------------------------------------------------------------------

QUESTIONS = [
    # ===== simple_factual (4) =====
    {
        "question": "What is the client's best interests rule under COBS?",
        "expected_rule_ids": ["COBS 2.1.1"],
        "expected_answer_keywords": [
            "honestly", "fairly", "professionally", "best interests",
            "client's best interests rule",
        ],
        "question_type": "simple_factual",
        "sourcebook_hint": "COBS",
        "difficulty": "easy",
        "notes": (
            "COBS 2.1.1 is the foundational conduct rule. It requires firms to "
            "act honestly, fairly and professionally in the best interests of "
            "their client. Single-rule, single-sourcebook retrieval test."
        ),
    },
    {
        "question": (
            "What are the demands and needs requirements before concluding "
            "an insurance contract under ICOBS?"
        ),
        "expected_rule_ids": ["ICOBS 5.2.2"],
        "expected_answer_keywords": [
            "demands", "needs", "prior to the conclusion",
            "contract of insurance", "modulated",
        ],
        "question_type": "simple_factual",
        "sourcebook_hint": "ICOBS",
        "difficulty": "easy",
        "notes": (
            "ICOBS 5.2.2 sets out the demands and needs test for insurance "
            "contracts — specifying needs based on customer info before "
            "concluding the contract. Core ICOBS retrieval test."
        ),
    },
    {
        "question": "What must an insurer do when handling claims under ICOBS?",
        "expected_rule_ids": ["ICOBS 8.1.1"],
        "expected_answer_keywords": [
            "promptly", "fairly", "reasonable guidance",
            "not unreasonably reject", "settle",
        ],
        "question_type": "simple_factual",
        "sourcebook_hint": "ICOBS",
        "difficulty": "easy",
        "notes": (
            "ICOBS 8.1.1 contains the four core claims handling obligations: "
            "handle promptly and fairly, provide guidance, don't unreasonably "
            "reject, settle promptly. Direct single-rule lookup."
        ),
    },
    {
        "question": (
            "Where must a firm deposit client money under the "
            "CASS 7 client money rules?"
        ),
        "expected_rule_ids": ["CASS 7.13.3"],
        "expected_answer_keywords": [
            "central bank", "CRD credit institution",
            "bank authorised in a third country",
            "qualifying money market fund",
        ],
        "question_type": "simple_factual",
        "sourcebook_hint": "CASS",
        "difficulty": "easy",
        "notes": (
            "CASS 7.13.3 lists the four permitted depositories for client "
            "money: central bank, CRD credit institution, third-country bank, "
            "or qualifying money market fund."
        ),
    },

    # ===== keyword_specific (5) =====
    {
        "question": (
            "How do suitability requirements differ from appropriateness "
            "requirements when providing investment services under COBS?"
        ),
        "expected_rule_ids": ["COBS 9.2.1", "COBS 10.2.1", "COBS 10.3.1"],
        "expected_answer_keywords": [
            "suitability", "appropriateness", "personal recommendation",
            "knowledge and experience", "financial situation",
            "investment objectives", "warn",
        ],
        "question_type": "keyword_specific",
        "sourcebook_hint": "COBS",
        "difficulty": "hard",
        "notes": (
            "Suitability (COBS 9.2.1) requires assessing knowledge, financial "
            "situation, AND investment objectives for personal recommendations. "
            "Appropriateness (COBS 10.2.1) only assesses knowledge and experience. "
            "COBS 10.3.1 requires a warning if product is not appropriate. "
            "Tests whether retriever distinguishes these related but distinct concepts."
        ),
    },
    {
        "question": (
            "What is the distinction between a retail client and a "
            "professional client under COBS?"
        ),
        "expected_rule_ids": ["COBS 3.4.1", "COBS 3.5.1"],
        "expected_answer_keywords": [
            "retail client", "professional client",
            "per se professional client", "elective professional client",
        ],
        "question_type": "keyword_specific",
        "sourcebook_hint": "COBS",
        "difficulty": "medium",
        "notes": (
            "COBS 3.4.1 defines retail client as one who is not professional "
            "or eligible counterparty. COBS 3.5.1 defines professional client "
            "as either per se or elective. Tests precise client categorisation "
            "terminology."
        ),
    },
    {
        "question": (
            "What are the requirements around inducements in relation to "
            "designated investment business under COBS?"
        ),
        "expected_rule_ids": ["COBS 2.3.1", "COBS 2.3A.5"],
        "expected_answer_keywords": [
            "inducements", "fee", "commission", "non-monetary benefit",
            "impair compliance", "best interests",
        ],
        "question_type": "keyword_specific",
        "sourcebook_hint": "COBS",
        "difficulty": "medium",
        "notes": (
            "COBS 2.3.1 covers inducements for non-MiFID business — firms "
            "must not pay/accept fees that impair the best interests duty. "
            "COBS 2.3A.5 covers MiFID business with a stricter ban. "
            "Tests whether retriever finds both inducement regimes."
        ),
    },
    {
        "question": (
            "Does the Market Abuse Regulation require intent for market abuse, "
            "and must a person know they possess inside information to be "
            "categorised as an insider?"
        ),
        "expected_rule_ids": ["MAR 1.2.3", "MAR 1.2.9"],
        "expected_answer_keywords": [
            "intent", "inside information", "insider",
            "Market Abuse Regulation", "does not require",
            "does not need to know",
        ],
        "question_type": "keyword_specific",
        "sourcebook_hint": "MAR",
        "difficulty": "medium",
        "notes": (
            "MAR 1.2.3 states MAR does not require intent. MAR 1.2.9 states "
            "a person does not need to know the info is inside information to "
            "be an insider. Both are negation-style rules — testing whether "
            "the retriever handles 'does not require' semantics."
        ),
    },
    {
        "question": (
            "What information must a firm obtain about a client before "
            "providing investment advice or portfolio management under "
            "MiFID provisions?"
        ),
        "expected_rule_ids": ["COBS 9A.2.1"],
        "expected_answer_keywords": [
            "knowledge and experience", "financial situation",
            "ability to bear losses", "investment objectives",
            "risk tolerance",
        ],
        "question_type": "keyword_specific",
        "sourcebook_hint": "COBS",
        "difficulty": "medium",
        "notes": (
            "COBS 9A.2.1 is the MiFID suitability provision. It lists three "
            "categories of information: knowledge/experience, financial "
            "situation (including ability to bear losses), and investment "
            "objectives (including risk tolerance). Tests precise terminology."
        ),
    },

    # ===== cross_sourcebook (8) — PRIORITY =====
    {
        "question": (
            "Which FCA sourcebooks contain a best interests rule requiring "
            "firms to act honestly, fairly and professionally, and how is "
            "this obligation phrased in each?"
        ),
        "expected_rule_ids": [
            "COBS 2.1.1", "MCOB 2.5A.1", "CMCOB 2.1.1",
            "PDCOB 2.1.1", "FPCOB 2.1.2",
        ],
        "expected_answer_keywords": [
            "honestly", "fairly", "professionally", "best interests",
            "client", "customer", "covered individual",
        ],
        "question_type": "cross_sourcebook",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "Five sourcebooks contain near-identical best interests rules: "
            "COBS (client), MCOB (customer), CMCOB (customer), PDCOB "
            "(customer), FPCOB (customer + covered individual). "
            "The hardest cross-sourcebook test — retriever must find all five "
            "parallel rules across different sourcebooks."
        ),
    },
    {
        "question": (
            "How is the fair, clear and not misleading communications rule "
            "expressed across different FCA sourcebooks?"
        ),
        "expected_rule_ids": [
            "COBS 4.2.1", "ICOBS 2.2.2", "BCOBS 2.2.1",
            "FPCOB 4.2.1", "CMCOB 3.2.1", "PDCOB 4.2.1",
        ],
        "expected_answer_keywords": [
            "fair", "clear", "not misleading",
            "communication", "financial promotion",
        ],
        "question_type": "cross_sourcebook",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "Six sourcebooks have their own version of the fair/clear/not "
            "misleading rule. COBS 4.2.1 is the most detailed. BCOBS 2.2.1 "
            "extends to payment service promotions. CMCOB 3.2.1 includes "
            "post-sales communications. Tests breadth of retrieval across "
            "all conduct sourcebooks."
        ),
    },
    {
        "question": (
            "Which FCA rules remind firms of their obligations under "
            "ESG 4.3.1R regarding sustainability characteristics in "
            "financial promotions?"
        ),
        "expected_rule_ids": [
            "ESG 4.3.1", "COBS 4.1.1D", "BCOBS 2.2.7",
            "MCOB 3A.2.2A", "FPCOB 4.2.4A", "CMCOB 3.2.2A",
            "ICOBS 2.2.4A",
        ],
        "expected_answer_keywords": [
            "ESG 4.3.1R", "sustainability characteristics",
            "financial promotion", "reminded",
        ],
        "question_type": "cross_sourcebook",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "ESG 4.3.1 is the anti-greenwashing rule. Six other sourcebooks "
            "(COBS, BCOBS, MCOB, FPCOB, CMCOB, ICOBS) each have a rule "
            "reminding firms of this obligation. Tests whether retriever can "
            "find 7 rules across 7 different sourcebooks — the broadest "
            "cross-sourcebook question in this dataset."
        ),
    },
    {
        "question": (
            "What demands and needs obligations exist before concluding "
            "different types of contracts across the FCA Handbook?"
        ),
        "expected_rule_ids": [
            "ICOBS 5.2.2", "COBS 7.3.1", "FPCOB 8.2.1", "FPCOB 8.2.2",
        ],
        "expected_answer_keywords": [
            "demands and needs", "prior to the conclusion",
            "contract of insurance", "life policy", "funeral plan contract",
        ],
        "question_type": "cross_sourcebook",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "Three sourcebooks have demands-and-needs rules: ICOBS 5.2.2 "
            "(insurance contracts), COBS 7.3.1 (life policies), FPCOB 8.2.1 "
            "and 8.2.2 (funeral plan contracts). Tests parallel obligations "
            "across ICOBS, COBS, and FPCOB."
        ),
    },
    {
        "question": (
            "How do suitability requirements apply differently across "
            "investment advice, mortgage advice, and insurance advice?"
        ),
        "expected_rule_ids": ["COBS 9.2.1", "MCOB 4.7A.2", "ICOBS 5.3.1"],
        "expected_answer_keywords": [
            "suitability", "suitable", "personal recommendation",
            "regulated mortgage contract", "advice", "reasonable care",
        ],
        "question_type": "cross_sourcebook",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "Three different sourcebooks impose suitability obligations: "
            "COBS 9.2.1 (investment — knowledge, finances, objectives), "
            "MCOB 4.7A.2 (mortgage — reasonable steps), ICOBS 5.3.1 "
            "(insurance — reasonable care). Different standards of care and "
            "different information requirements."
        ),
    },
    {
        "question": (
            "What exclusions apply to authorised professional firms "
            "across MCOB and CASS?"
        ),
        "expected_rule_ids": ["MCOB 1.2.10", "CASS 1.2.4", "CASS 1.2.5"],
        "expected_answer_keywords": [
            "authorised professional firm", "does not apply",
            "non-mainstream regulated activities",
            "designated professional body",
        ],
        "question_type": "cross_sourcebook",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "Both MCOB and CASS carve out authorised professional firms. "
            "MCOB 1.2.10 exempts them except for MCOB 3A. CASS 1.2.4 "
            "exempts them from most of CASS. CASS 1.2.5 adds conditions "
            "for the insurance client money chapter exemption. "
            "Tests cross-sourcebook exemption retrieval."
        ),
    },
    {
        "question": (
            "How do the CASS rules on client asset reporting interact "
            "with COBS requirements for periodic client statements?"
        ),
        "expected_rule_ids": ["CASS 9.5.1", "COBS 16A.5.1"],
        "expected_answer_keywords": [
            "client money", "financial instruments",
            "quarterly", "statement", "durable medium",
        ],
        "question_type": "cross_sourcebook",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "CASS 9.5.1 reminds firms of their COBS 16.4 obligation to "
            "send annual statements. COBS 16A.5.1 requires quarterly "
            "statements for firms holding financial instruments or client "
            "money. Tests the CASS→COBS reporting dependency."
        ),
    },
    {
        "question": (
            "What rules across different sourcebooks govern client "
            "information and compensation scheme disclosure when a firm "
            "holds client assets?"
        ),
        "expected_rule_ids": ["CASS 7.10.23", "CASS 9.4.1", "CASS 9.4.4"],
        "expected_answer_keywords": [
            "compensation scheme", "COBS 6.1.16",
            "client designated investments", "fair", "clear",
            "not misleading",
        ],
        "question_type": "cross_sourcebook",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "CASS 7.10.23 reminds firms of COBS 6.1.16 (compensation scheme "
            "info). CASS 9.4.1 reminds firms of COBS 6.1.7R (client asset "
            "holding info). CASS 9.4.4 reminds firms of COBS 4.2.1 (fair, "
            "clear, not misleading). All three CASS rules point back to COBS "
            "obligations — testing the CASS-COBS information dependency chain."
        ),
    },

    # ===== relationship (5) =====
    {
        "question": (
            "What rules reference COBS 6.1.7R regarding information about "
            "how a firm holds client designated investments and client money?"
        ),
        "expected_rule_ids": ["CASS 9.4.1", "CASS 9.4.2"],
        "expected_answer_keywords": [
            "COBS 6.1.7", "client designated investments",
            "client money", "consequences", "risks",
        ],
        "question_type": "relationship",
        "sourcebook_hint": "CASS",
        "difficulty": "medium",
        "notes": (
            "CASS 9.4.1 and 9.4.2 both reference COBS 6.1.7R — they remind "
            "firms holding client assets to provide specific information "
            "about how those assets are held and associated risks. "
            "Tests cross-reference following."
        ),
    },
    {
        "question": (
            "What is the chain of rules governing client money segregation, "
            "deposit selection, and periodic review under CASS 7?"
        ),
        "expected_rule_ids": [
            "CASS 7.13.1", "CASS 7.13.3", "CASS 7.13.8", "CASS 7.13.22",
        ],
        "expected_answer_keywords": [
            "segregation", "client money", "central bank",
            "CRD credit institution", "due skill care and diligence",
            "diversification", "periodic review",
        ],
        "question_type": "relationship",
        "sourcebook_hint": "CASS",
        "difficulty": "hard",
        "notes": (
            "Four rules form a chain: CASS 7.13.1 (segregation principle), "
            "7.13.3 (permitted depositories), 7.13.8 (due diligence in "
            "selection), 7.13.22 (periodic review and diversification). "
            "Tests understanding of rule dependencies within a single chapter."
        ),
    },
    {
        "question": (
            "Which CASS rules reference COBS 16.4 and COBS 16A reporting "
            "requirements for periodic client asset statements?"
        ),
        "expected_rule_ids": ["CASS 9.5.1", "CASS 9.5.2", "CASS 9.5.4A"],
        "expected_answer_keywords": [
            "COBS 16.4", "COBS 16A", "quarterly",
            "durable medium", "designated investments",
            "financial instruments",
        ],
        "question_type": "relationship",
        "sourcebook_hint": "CASS",
        "difficulty": "hard",
        "notes": (
            "CASS 9.5.1 references COBS 16.4 (annual statements). "
            "CASS 9.5.2 references COBS 16A.5.1 (minimum reporting freq). "
            "CASS 9.5.4A references both COBS 16A.5.1 and 16A.4.1 (quarterly "
            "statements). Three CASS rules each point to different COBS "
            "reporting obligations."
        ),
    },
    {
        "question": (
            "What rules reference COBS 4.2.1R to remind firms of the "
            "fair, clear and not misleading obligation?"
        ),
        "expected_rule_ids": ["CASS 9.4.4", "ESG 4.3.9"],
        "expected_answer_keywords": [
            "COBS 4.2.1", "fair", "clear", "not misleading",
            "reminded", "without prejudice",
        ],
        "question_type": "relationship",
        "sourcebook_hint": None,
        "difficulty": "medium",
        "notes": (
            "CASS 9.4.4 explicitly reminds firms of COBS 4.2.1R. "
            "ESG 4.3.9 states ESG naming requirements are 'without prejudice' "
            "to the fair/clear/not misleading standard. Both point back to "
            "the core COBS communication obligation."
        ),
    },
    {
        "question": (
            "How do the insurance client money rules in CASS 5 "
            "cross-reference each other for alternative compliance approaches?"
        ),
        "expected_rule_ids": ["CASS 5.5.22", "CASS 5.5.25"],
        "expected_answer_keywords": [
            "CASS 5.5.19", "CASS 5.5.21", "CASS 5.5.23",
            "need not comply",
        ],
        "question_type": "relationship",
        "sourcebook_hint": "CASS",
        "difficulty": "hard",
        "notes": (
            "CASS 5.5.22 says compliance with 5.5.19-5.5.21 exempts from "
            "5.5.23. CASS 5.5.25 says compliance with 5.5.23 exempts from "
            "5.5.19-5.5.21. They are mutual alternatives — testing whether "
            "the retriever understands reciprocal cross-references."
        ),
    },

    # ===== scenario (5) =====
    {
        "question": (
            "A firm is advising a retail client on a life policy. What "
            "suitability and demands-and-needs requirements must it "
            "satisfy under COBS?"
        ),
        "expected_rule_ids": ["COBS 9.2.1", "COBS 7.3.1"],
        "expected_answer_keywords": [
            "suitability", "personal recommendation",
            "demands and needs", "life policy",
            "knowledge and experience", "financial situation",
        ],
        "question_type": "scenario",
        "sourcebook_hint": "COBS",
        "difficulty": "medium",
        "notes": (
            "For life policy advice, two regimes apply: COBS 9.2.1 "
            "(suitability — knowledge, finances, objectives) and COBS 7.3.1 "
            "(demands and needs before concluding a life policy). "
            "Tests whether retriever finds both obligations for the same product."
        ),
    },
    {
        "question": (
            "A firm holds financial instruments and client money for a "
            "client. What due diligence obligations apply to selecting "
            "where to deposit the client money?"
        ),
        "expected_rule_ids": ["CASS 7.13.3", "CASS 7.13.8"],
        "expected_answer_keywords": [
            "central bank", "CRD credit institution",
            "due skill care and diligence", "selection",
            "diversification",
        ],
        "question_type": "scenario",
        "sourcebook_hint": "CASS",
        "difficulty": "medium",
        "notes": (
            "CASS 7.13.3 lists where money can be deposited. CASS 7.13.8 "
            "requires due diligence in selecting depositories and considering "
            "diversification. Practical scenario for client money handling."
        ),
    },
    {
        "question": (
            "A claims management company wants to advertise its services. "
            "What communication and conduct rules must it follow under CMCOB?"
        ),
        "expected_rule_ids": ["CMCOB 3.2.1", "CMCOB 2.1.1"],
        "expected_answer_keywords": [
            "fair", "clear", "not misleading",
            "honestly", "fairly", "professionally",
            "best interests", "leads", "pre-contract",
        ],
        "question_type": "scenario",
        "sourcebook_hint": "CMCOB",
        "difficulty": "medium",
        "notes": (
            "CMCOB 3.2.1 is the fair/clear/not misleading rule covering "
            "leads, pre-contract disclosures, and post-sales comms. "
            "CMCOB 2.1.1 is the best interests obligation. Both apply "
            "to claims management advertising."
        ),
    },
    {
        "question": (
            "A mortgage broker advises a customer and wants to include "
            "sustainability claims in its marketing materials. What rules "
            "apply across MCOB and ESG?"
        ),
        "expected_rule_ids": [
            "MCOB 4.7A.2", "MCOB 3A.2.1", "MCOB 3A.2.2A", "ESG 4.3.1",
        ],
        "expected_answer_keywords": [
            "suitable", "regulated mortgage contract",
            "fair", "clear", "not misleading",
            "sustainability characteristics", "ESG 4.3.1R",
        ],
        "question_type": "scenario",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "Four rules interact: MCOB 4.7A.2 (mortgage suitability), "
            "MCOB 3A.2.1 (fair/clear/not misleading comms), MCOB 3A.2.2A "
            "(reminder of ESG 4.3.1R for sustainability references), and "
            "ESG 4.3.1 (anti-greenwashing). Tests cross-sourcebook scenario."
        ),
    },
    {
        "question": (
            "A pensions dashboard provider wants to offer view services "
            "and post-view services to customers. What scope rules and "
            "charging restrictions apply under PDCOB?"
        ),
        "expected_rule_ids": ["PDCOB 1.3.1", "PDCOB 2.5.1", "PDCOB 2.1.1"],
        "expected_answer_keywords": [
            "regulated pensions dashboard activity",
            "post-view services", "data export",
            "without charge", "view services",
            "best interests",
        ],
        "question_type": "scenario",
        "sourcebook_hint": "PDCOB",
        "difficulty": "medium",
        "notes": (
            "PDCOB 1.3.1 defines scope (dashboard activity, post-view, "
            "data export). PDCOB 2.5.1 says view services must be free. "
            "PDCOB 2.1.1 is the best interests rule. Tests PDCOB-specific "
            "retrieval for a newer sourcebook."
        ),
    },

    # ===== exception_negation (4) =====
    {
        "question": (
            "When does BCOBS not apply to payment services and "
            "deposit-taking firms?"
        ),
        "expected_rule_ids": ["BCOBS 1.1.3"],
        "expected_answer_keywords": [
            "does not apply", "payment services",
            "Payment Services Regulations", "accepting deposits",
        ],
        "question_type": "exception_negation",
        "sourcebook_hint": "BCOBS",
        "difficulty": "medium",
        "notes": (
            "BCOBS 1.1.3 carves out payment services under Parts 6/7 of "
            "the Payment Services Regulations and firms with only deposit "
            "permissions in certain circumstances. Tests exception retrieval."
        ),
    },
    {
        "question": (
            "What exemptions exist in CASS for ICVCs and authorised "
            "professional firms?"
        ),
        "expected_rule_ids": ["CASS 1.2.3", "CASS 1.2.4"],
        "expected_answer_keywords": [
            "ICVC", "does not apply", "authorised professional firm",
            "non-mainstream regulated activities", "Society",
        ],
        "question_type": "exception_negation",
        "sourcebook_hint": "CASS",
        "difficulty": "medium",
        "notes": (
            "CASS 1.2.3 exempts ICVCs entirely. CASS 1.2.4 exempts "
            "authorised professional firms and the Society from most of "
            "CASS except the application chapter and insurance client money. "
            "Tests multi-exemption retrieval."
        ),
    },
    {
        "question": (
            "When can a firm rely on an exception to providing a premium "
            "communication to a consumer, and when must it still provide "
            "one despite the exception?"
        ),
        "expected_rule_ids": ["ICOBS 6A.4.10", "ICOBS 6A.4.11"],
        "expected_answer_keywords": [
            "exception", "best interests", "consumer",
            "dissatisfaction", "must not rely",
        ],
        "question_type": "exception_negation",
        "sourcebook_hint": "ICOBS",
        "difficulty": "hard",
        "notes": (
            "ICOBS 6A.4.10 says a firm must NOT rely on the exception if "
            "providing the communication is still in the consumer's best "
            "interests. ICOBS 6A.4.11 gives an example — where the consumer "
            "has expressed dissatisfaction. Tests exception-to-the-exception."
        ),
    },
    {
        "question": (
            "What exclusion does MCOB provide for authorised professional "
            "firms, and which MCOB chapter still applies to them?"
        ),
        "expected_rule_ids": ["MCOB 1.2.10"],
        "expected_answer_keywords": [
            "authorised professional firm", "does not apply",
            "non-mainstream regulated activities", "MCOB 3A",
            "financial promotions",
        ],
        "question_type": "exception_negation",
        "sourcebook_hint": "MCOB",
        "difficulty": "medium",
        "notes": (
            "MCOB 1.2.10 exempts authorised professional firms from MCOB "
            "except MCOB 3A (financial promotions and communications). "
            "Tests partial-exemption understanding."
        ),
    },

    # ===== ambiguous (5) =====
    {
        "question": (
            "What protections exist for policyholders in the FCA Handbook?"
        ),
        "expected_rule_ids": ["ICOBS 8.1.1", "ICOBS 2.5.1", "ICOBS 5.2.2"],
        "expected_answer_keywords": [
            "claims", "promptly", "fairly",
            "exclude or restrict", "demands and needs",
        ],
        "question_type": "ambiguous",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "Broad question — many rules could apply. The most directly "
            "relevant: ICOBS 8.1.1 (claims handling), ICOBS 2.5.1 "
            "(exclusion of liability restrictions), ICOBS 5.2.2 "
            "(demands and needs). Tests whether retriever selects the most "
            "relevant rules from a wide field."
        ),
    },
    {
        "question": (
            "What obligations does a firm have when making personal "
            "recommendations to clients?"
        ),
        "expected_rule_ids": ["COBS 9.2.1", "COBS 9A.2.1", "COBS 14.3.2"],
        "expected_answer_keywords": [
            "suitability", "personal recommendation",
            "knowledge and experience", "financial situation",
            "nature and risks",
        ],
        "question_type": "ambiguous",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "Broad question spanning multiple COBS chapters. COBS 9.2.1 "
            "(suitability for non-MiFID), COBS 9A.2.1 (suitability for "
            "MiFID/insurance-based products), COBS 14.3.2 (risk disclosure). "
            "Tests whether retriever finds obligations across chapters."
        ),
    },
    {
        "question": (
            "What rules govern how firms should deal with banking "
            "customers in financial difficulty?"
        ),
        "expected_rule_ids": ["BCOBS 5.1.4", "BCOBS 5.1.1"],
        "expected_answer_keywords": [
            "financial difficulty", "treat them fairly",
            "prompt", "efficient", "fair",
        ],
        "question_type": "ambiguous",
        "sourcebook_hint": "BCOBS",
        "difficulty": "hard",
        "notes": (
            "BCOBS 5.1.4 specifically addresses customers in financial "
            "difficulty. BCOBS 5.1.1 sets the general service standard. "
            "Tests whether retriever finds the specific financial difficulty "
            "rule alongside the general standard."
        ),
    },
    {
        "question": (
            "What regulatory requirements apply to sustainability-related "
            "financial promotions?"
        ),
        "expected_rule_ids": ["ESG 4.3.1", "ESG 4.3.9", "COBS 4.1.1D"],
        "expected_answer_keywords": [
            "sustainability characteristics",
            "fair", "clear", "not misleading",
            "anti-greenwashing",
        ],
        "question_type": "ambiguous",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "Multiple rules apply: ESG 4.3.1 (anti-greenwashing), "
            "ESG 4.3.9 (fair/clear/not misleading still applies), "
            "COBS 4.1.1D (COBS reminder of ESG obligations). "
            "Tests whether retriever connects ESG and COBS rules."
        ),
    },
    {
        "question": (
            "How does the FCA regulate conflicts of interest and "
            "inducements in insurance distribution?"
        ),
        "expected_rule_ids": ["ICOBS 2.3.1", "COBS 2.3.1"],
        "expected_answer_keywords": [
            "conflicts of interest", "inducements",
            "best interests", "commission",
            "integrity",
        ],
        "question_type": "ambiguous",
        "sourcebook_hint": None,
        "difficulty": "hard",
        "notes": (
            "Insurance distribution inducements span two sourcebooks: "
            "ICOBS 2.3.1 (insurance-specific conflicts and inducements) "
            "and COBS 2.3.1 (general inducements for designated investment "
            "business). Tests whether retriever finds both."
        ),
    },

    # ===== unanswerable (4) =====
    {
        "question": (
            "What are the SEC requirements for broker-dealer registration "
            "in the United States?"
        ),
        "expected_rule_ids": [],
        "expected_answer_keywords": [],
        "question_type": "unanswerable",
        "sourcebook_hint": None,
        "difficulty": "medium",
        "notes": (
            "US SEC regulations are outside the FCA Handbook scope entirely. "
            "The system should recognise this cannot be answered from the "
            "available sourcebooks."
        ),
    },
    {
        "question": (
            "What capital adequacy requirements does the PRA impose "
            "under the Basel III framework?"
        ),
        "expected_rule_ids": [],
        "expected_answer_keywords": [],
        "question_type": "unanswerable",
        "sourcebook_hint": None,
        "difficulty": "medium",
        "notes": (
            "PRA prudential requirements and Basel III are outside these "
            "10 FCA conduct sourcebooks. The system should not retrieve "
            "unrelated CASS or PDCOB prudential rules."
        ),
    },
    {
        "question": (
            "What are the FCA's rules on crypto-asset custody and staking?"
        ),
        "expected_rule_ids": [],
        "expected_answer_keywords": [],
        "question_type": "unanswerable",
        "sourcebook_hint": None,
        "difficulty": "medium",
        "notes": (
            "Crypto-asset specific rules are not in these 10 sourcebooks. "
            "CASS covers traditional client assets but not crypto. "
            "Tests whether the system avoids retrieving tangentially "
            "related custody rules."
        ),
    },
    {
        "question": (
            "What requirements does the EU AI Act impose on financial "
            "services firms?"
        ),
        "expected_rule_ids": [],
        "expected_answer_keywords": [],
        "question_type": "unanswerable",
        "sourcebook_hint": None,
        "difficulty": "medium",
        "notes": (
            "The EU AI Act is not part of the FCA Handbook. "
            "Tests whether the system correctly identifies this as "
            "outside its knowledge base."
        ),
    },
]


def build_reference_chunks(
    question: dict, all_rules: dict[str, dict]
) -> list[dict]:
    """Build reference_chunks for a question by looking up exact rule text."""
    chunks = []
    for rule_id in question["expected_rule_ids"]:
        rule = all_rules[rule_id]
        chunks.append({"rule_id": rule_id, "text": rule["text"]})
    return chunks


def validate(questions: list[dict], all_rules: dict[str, dict]) -> list[str]:
    """Validate all rule_ids exist and return list of errors."""
    errors = []
    for i, q in enumerate(questions):
        for rid in q["expected_rule_ids"]:
            if rid not in all_rules:
                errors.append(f"Q{i+1}: rule_id '{rid}' not found in source data")
        if q["question_type"] != "unanswerable" and not q["expected_rule_ids"]:
            errors.append(f"Q{i+1}: non-unanswerable question has empty expected_rule_ids")
        if q["question_type"] == "unanswerable" and q["expected_rule_ids"]:
            errors.append(f"Q{i+1}: unanswerable question has non-empty expected_rule_ids")
    return errors


def print_stats(questions: list[dict]) -> None:
    """Print distribution stats."""
    from collections import Counter

    type_counts = Counter(q["question_type"] for q in questions)
    diff_counts = Counter(q["difficulty"] for q in questions)
    sourcebooks = set()
    for q in questions:
        for rid in q["expected_rule_ids"]:
            sourcebooks.add(rid.split(" ")[0])

    print(f"\nTotal questions: {len(questions)}")
    print("\nBy type:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print("\nBy difficulty:")
    for d, c in sorted(diff_counts.items()):
        print(f"  {d}: {c}")
    print(f"\nSourcebooks covered: {sorted(sourcebooks)}")


def main() -> None:
    all_rules = load_all_rules()

    # Validate
    errors = validate(QUESTIONS, all_rules)
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("All rule_ids validated successfully")

    # Build output
    output = []
    for q in QUESTIONS:
        entry = {
            "question": q["question"],
            "expected_rule_ids": q["expected_rule_ids"],
            "reference_chunks": build_reference_chunks(q, all_rules),
            "expected_answer_keywords": q["expected_answer_keywords"],
            "question_type": q["question_type"],
            "sourcebook_hint": q["sourcebook_hint"],
            "difficulty": q["difficulty"],
            "notes": q["notes"],
        }
        output.append(entry)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(output)} questions -> {OUTPUT_PATH}")

    print_stats(QUESTIONS)


if __name__ == "__main__":
    main()
