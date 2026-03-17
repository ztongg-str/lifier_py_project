# Multi-Agent NLP Triage Project — Setup Guide

## Folder Structure
```
your-project/
  agents/
    agent1_knowledge.md     ← Agent 1 system prompt
    agent2_oop.md           ← Agent 2 system prompt
    agent3_notebook.md      ← Agent 3 system prompt
    agent4_validator.md     ← Validator system prompt
  data/
    your_dataset.csv
  
  modules/
     ESI.py
     other files for using in the oop strucutre notebook
  y2-t2-final-project.ipynb (old notebook)
  knowledge.md              ← created by Agent 1
  ooptriage.ipynb      ← created by Agent 2
  normal_triage.ipynb     ← created by Agent 3
```

---

## How to Run in VS Code

### Step 1 — Install Claude Code
```bash
npm install -g @anthropic-ai/claude-code
claude --version   # verify install
```

### Step 2 — Open VS Code split terminals
Press `Ctrl+Shift+5` (or `Cmd+Shift+5` on Mac) to split your terminal panel.
You'll run one agent per terminal.

### Step 3 — Run Agent 1 first (knowledge extraction)
```bash
cd your-project
claude \
  --system-prompt "$(cat agents/agent1_knowledge.md)" \
  "Read ESI.py, the ESI PDF handbook, and y2-t2-final-project.ipynb. Extract all domain knowledge and write knowledge.md exactly as specified in your instructions."
```
Wait for `knowledge.md` to be written before continuing.

### Step 4 — Run Agent 2 and Agent 3 in parallel
Open two more terminal splits. Run both at the same time:

**Terminal 2 — OOP notebook:**
```bash
claude \
  --system-prompt "$(cat agents/agent2_oop.md)" \
  "Read knowledge.md and the old notebook. Build oop_nlp_triage.ipynb with the full OOP pipeline as specified."
```

**Terminal 3 — Flat notebook:**
```bash
claude \
  --system-prompt "$(cat agents/agent3_notebook.md)" \
  "Read knowledge.md. Build flat_nlp_triage.ipynb — the same pipeline without OOP."
```

### Step 5 — Run Validator after Agent 2 finishes
```bash
claude \
  --system-prompt "$(cat agents/agent4_validator.md)" \
  "Audit oop_nlp_triage.ipynb and produce a completeness report."
```
Fix any ❌ gaps before finalising.

---

## Tips
- If Claude Code loses context mid-session, re-run with `--continue` flag
- Each session has its own context window — agents don't share memory, only files
- `knowledge.md` is the critical handoff file — review it manually before running Agents 2 and 3
- If a session times out, check what files were written and resume with a narrower task

## Expected Outputs
| File | Created by |
|---|---|
| `knowledge.md` | Agent 1 |
| `oop_nlp_triage.ipynb` | Agent 2 |
| `flat_nlp_triage.ipynb` | Agent 3 |
| `model_1st_best.pkl` | Agent 2 + 3 |
| `model_2nd_best.pkl` | Agent 2 + 3 |
| `model_3rd_best.pkl` | Agent 2 + 3 |
| `validator_report.md` | Validator |
