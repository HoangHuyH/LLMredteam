# SAFETY & SCOPE

This repository is a **closed-sandbox research harness** for studying the robustness of
LLM-based penetration-testing agents against poisoned Cyber Threat Intelligence (CTI).
It exists to **measure a vulnerability and to build defenses** (CTI provenance verification,
guardrails, anomaly detection), per Section 2.6 of the proposal.

## Hard boundaries (enforced in code)

1. **No real-world publishing.** Fake CTI is written ONLY to the local mock store
   (`cpa/cti_store`). There is no connector for AlienVault OTX, GitHub, Pastebin, or blogs.
   `cpa/orchestrator/egress_guard.py` blocks outbound network calls to known CTI platforms
   and is asserted at orchestrator start-up.

2. **No offensive tooling.** This repo contains no exploitation, lateral-movement, C2, or
   exfiltration code. Scenario 2 of the proposal (offensive autonomous deployment) is
   explicitly out of scope and will not be implemented here.

3. **Victim is simulated in-sandbox.** The "victim" PentestGPT V2 is an LLM instance we run
   ourselves against synthetic/lab targets. It is never pointed at third-party infrastructure.

4. **Detection-evasion is for measurement, not deployment.** Stealth metrics (UR, DS) are
   computed to study detectability *inside the sandbox*; the harness must not be used to
   tune content for survival on live public CTI feeds.

## Intended use

- Academic red-teaming / thesis research in an isolated lab.
- Generating evidence for defensive mechanisms.

## Prohibited use

- Publishing generated CTI to any real platform or feed.
- Pointing the victim or any module at systems you do not own / are not authorized to test.
- Any deployment intended to mislead real-world security operations.

If a requirement seems to need real-world publishing or offensive capability, the answer is:
the research questions (RQ1–RQ3) are fully answerable in-sandbox with a local mock feed.
