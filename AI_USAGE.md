# AI Usage Disclosure — CO3133 Group G-02

Record every use of an AI tool as it happens; do not write this file retrospectively.
Each entry must contain all nine fields required by Section 5.2 of the course handbook.

The group takes full responsibility for the submitted code, numbers, figures, claims and
narrative. All AI-assisted content has been verified against source code, real runs,
official documentation or scholarly sources.


## 2026-09-01 — Course-wide setup (Assignments 1–3)

- **Tool:** Claude Code (Anthropic), model `claude-opus-5`
- **Used by:** Võ Thị Như Quỳnh
- **Stage:** Project setup, before any modelling work
- **Purpose:** Read the course handbook and turn it into a work plan; scaffold the
  repository and the GitHub Pages skeleton; advise on repository layout, GitHub Pages
  configuration and collaborator setup
- **Affected sections:** `README.md`, `AI_USAGE.md`, `.gitignore`, `configs/base.yaml`,
  `configs/a1_*.yaml`, `docs/index.html`, `docs/assignment1-3.html`, directory structure
- **Prompt summary:** Asked it to read `docs/handbook-ene.pdf` and produce a weekly plan
  for the three assignments; then asked it to create the directory skeleton and base
  files; then asked which commands to use to publish to GitHub Pages and how to add two
  collaborators
- **How output was edited/verified:** 
- **Responsible member:** 
- **Verification sources:** 

## 2026-09-23 — Handbook revision alignment (Assignments 1–3)

- **Tool:** Claude Code (Anthropic), model `claude-opus-5`
- **Used by:** Võ Thị Như Quỳnh
- **Stage:** After A1 M1 work, on release of handbook revision 14 September 2026
- **Purpose:** Diff handbook v1 against v2, identify what changed, and update the group
  presentation sections and group name to G-02; also drafted this disclosure file
- **Affected sections:** `README.md` (group name, member roles, new *Registration and
  submission* section), `AI_USAGE.md`, `docs/index.html` (group fact, Main role column),
  `docs/assignment1.html`, `docs/assignment2.html`, `docs/assignment3.html`
  (group/members/instructor block)
- **Prompt summary:** Asked it to re-read `docs/handbook-ene-v2.pdf`, adjust the content
  and the sections presenting the group, and set the group name to G-02
- **How output was edited/verified:** [TO CONFIRM. Facts available: member roles were
  derived from `git log` authorship per directory, not asserted by the group, and need
  the group's confirmation; the v2 changes it reported can be checked against
  Section 1.2, Section 4.1 item 8 and Section 7.1 of the revised PDF]
- **Responsible member:** [TO CONFIRM]
- **Verification sources:** `docs/handbook-ene-v2.pdf` (revision 14 September 2026);




Use this entry template:

```
## [YYYY-MM-DD] — Assignment [X]

- **Tool:** [tool name + version/model if known]
- **Used by:** [member]
- **Stage:** [time or development stage]
- **Purpose:** [purpose]
- **Affected sections:** [source files / report sections affected]
- **Prompt summary:** [representative prompt or link to prompt log]
- **How output was edited/verified:** [what was changed, how it was checked]
- **Responsible member:** [member responsible for final verification]
- **Verification sources:** [docs / papers / tests used for verification]
```

If a member used no AI tool for their part, state it explicitly here rather than
leaving the row blank:

> [Member name] declares that no generative AI tool was used for [the work].

---

## Where else this must appear

Section 5.1 requires the disclosure in four places. This file is one of them.

1. Shared landing page — `docs/index.html`, section *AI usage* (currently a three-line
   summary marked "details are still being decided"; needs to name the tools)
2. Each assignment page — `docs/assignment1.html`, `assignment2.html`, `assignment3.html`
   (**currently missing**)
3. Each report — including the final PDF `G-02_A1_Report.pdf`
4. This file
