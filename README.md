# 如是 (rushi)

> "If it were them, they would surely have done the same."
> 如是我闻 ("Thus have I heard"): write down what you heard, as it was.

rushi takes the part of a person that stays with you and organizes it into a memory graph with a timeline. You can then chat with "them as you remember them", or ask "what would they do in this situation?"

The name 如是 combines 如果是ta ("if it were them") with 如是我闻 ("thus have I heard"). The idea comes from *Frieren: Beyond Journey's End*: "Himmel would have done the same."

**Zero dependencies**: pure Python standard library (SQLite + urllib). Set `ANTHROPIC_API_KEY` to use Claude. You can also point it at a local model such as Ollama or vLLM. With no key at all, the built-in `mock` model runs the whole pipeline offline.

The tool is built for Chinese-language material. Prompts, sample data, and generated persona text are in Chinese.

## Install

```bash
cd rushi
pip install -e .          # or skip installing and run python -m rushi.cli ...
export ANTHROPIC_API_KEY=sk-...        # use Claude
# or a local model: export RUSHI_LLM=openai RUSHI_OPENAI_BASE=http://localhost:11434/v1 RUSHI_MODEL=qwen3:8b
```

You can also put these settings in a config file: `cp .env.example .env` and fill it in. `.env` is git-ignored, and `~/.rushi/.env` works too. Real environment variables take precedence. Any OpenAI-compatible API works, for example MiniMax or DeepSeek.

## Keeping data out of the repo

All persona data lives **outside the repository**, under `$RUSHI_HOME` (default `~/.rushi`). That includes `persona.db`, `speakers.json`, sandboxes, exports, and raw chat logs.

```
~/.rushi/
  raw/                     raw chat logs / narrative material
  workspaces/<name>/       one workspace per persona: persona.db, speakers.json, sandboxes/, exports/
```

- `-w name` resolves to `$RUSHI_HOME/workspaces/name`. `-w ./path` or an absolute path is used as given.
- The CLI **refuses** to create a workspace inside the code repository. Set `RUSHI_ALLOW_IN_REPO=1` if you really need to.
- As a safety net, `.gitignore` blocks `*.db`, `speakers.json` (except in `examples/`), `exports/`, and similar files.
- The only persona in the repo is 阿泽 in `examples/`, a fictional character used for tests.

## Stages 0–4: building a single persona end to end

```bash
rushi -w ta init                                 # 0. create the workspace (~/.rushi/workspaces/ta) and a speakers.json template
rushi -w ta inspect-chat chat.txt                # list the speakers in a chat log
#    edit ~/.rushi/workspaces/ta/speakers.json: target = their name; others = other people and their relation to them; aliases = anonymized names
rushi -w ta ingest-chat chat.txt                 # 1. import a chat (split sessions, merge bursts, anonymize, mark which side is them)
rushi -w ta ingest-text about.md --reporter user # 1. import third-person descriptions / interviews / diaries (format below)
rushi -w ta add-fact "他其实很怕狗" --dimension I --year 2018   # add a single fact at any time
rushi -w ta style                                # 2. style card: statistics + rules (versioned)
rushi -w ta extract                              # 3. extract chunk by chunk → evidence-backed observations, decision cases, graph
rushi -w ta coverage                             # which dimensions have thin evidence
rushi -w ta review list                          # human review queue (sensitive / ungrounded / contradictory first)
rushi -w ta review confirmed clm_xxx             #    confirmed | rejected | edited --text "rewritten claim"
rushi -w ta reflect                              # 3. reflect: derive traits across observations + record contradictions (never deletes)
rushi -w ta persona                              # 3. core persona card (always in context; sensitive dimensions excluded by default)
rushi -w ta chat --relation close_friend --period 2018-2020     # 4. chat as them
rushi -w ta ask "外地 offer 薪资高40% 但妈妈身体不好。选项: 接受/拒绝/拖着"   # 4. what would they do
rushi -w ta export-graph                         # export graph.json → open it in viewer/index.html
rushi -w ta eval blind -n 20                     # style blind test sheet; fill it in, then run rushi eval score
rushi -w ta eval backtest                        # decision backtest (hide known decisions and try to predict them)
```

Offline trial run: `RUSHI_LLM=mock python -m unittest tests.test_pipeline -v`, or walk through the commands above with the sample data in `examples/`.

## Narrative material format

Each section starts with a header line. The sample text below is the kind of Chinese material the tool expects.

```
--- year: 2016 ; relation: close_friend ; reporter: user ---
那年他刚搬去上海……

--- reporter: third:老张 ---
老张说他压力大的时候会消失两天……

--- reporter: self ---
我这个人吧，从小就不太会表达……
```

`reporter` sets the evidence type. `self` may produce self_report or behavior; `user` and `third:*` can only produce other_report. This is enforced in code, so whatever the model claims is ignored.

## Design principles and how the code implements them

| Principle | Implementation |
|---|---|
| Know which side is them | `speakers.json` says so explicitly; nothing is guessed, and unknown speakers are flagged |
| Evidence first | Every claim must carry a quote, which is checked against the source (`extract.ground`). A mismatch marks it `ungrounded`, halves its confidence, and sends it to the review queue |
| Time is first-class | Claims and edges have `valid_from/valid_to` (when it held in their life) plus `recorded_at` (when we learned it). A change closes the old edge; nothing is deleted |
| Contradictions are allowed | A `contradictions` table plus `contradicts` on claims. Reflection looks for contradictions explicitly, and the persona card is written as "on one hand… on the other…" |
| People change | Style statistics are split by year; chat and ask accept `--period` |
| No idealizing | The extract, reflect, persona-card, and decision prompts all require keeping unflattering traits. The structured decision path weighs evidence only |
| Networked graph | Five layers: source text → entities/facts → observations → traits → themes. Traits have cross-dimension relations (causes / manifests as / trades off / reinforces), following Graphiti's episodic/entity/community structure |
| Traceable derivation | Every trait records `based_on`, `reasoning`, and an inference type (generalization / causal / contrast). Clicking a trait in the viewer expands its chain down to the source text |
| Entity resolution | 妈妈 / 我妈 / 母亲 (all "mom") merge into one node and keep the aliases. The same decision found in different materials is recorded as one case |
| Confidence | Computed in code from the number of supports, number of source passages, evidence types, periods, and contradictions. Material from a single passage is capped at moderate |
| Sensitive dimension | J (childhood / formative experiences) stays out of chat and decisions by default and is visible only in the backstage views (`--include-sensitive` turns it on) |
| Three memory tiers | ① a resident persona card + style rules ② per-turn retrieval of quotes, memories, and observations (char-bigram BM25, pluggable embeddings) ③ the model can reply `[search: …]` to look things up itself |
| Fiction never pollutes the persona | `sandbox create` copies a snapshot into a separate db. Inside the sandbox the base is opened read-only and any write raises an error. There is no automatic write-back; `promote()` is manual and tags results `origin:fiction` |
| Evaluation | Blind test (real vs generated; 50% means indistinguishable) and decision backtest (accuracy, mean probability given to the real choice, random baseline) |

## Layout

```
rushi/
├── rushi/
│   ├── schema.py      data model, graph layers, and the 17 persona dimensions
│   ├── store.py       SQLite (never deletes; read-only mode)
│   ├── config.py      .env loading
│   ├── ingest/chat.py chat parsing (csv/json/txt), session and burst splitting, speaker mapping
│   ├── ingest/narrative.py narrative chunking, anonymization, reporter rules
│   ├── llm.py         Anthropic / OpenAI-compatible (local) / Mock
│   ├── extract.py     per-chunk extraction + grounding check + graph writes
│   ├── resolve.py     entity resolution and dedup
│   ├── scoring.py     confidence computed from evidence
│   ├── themes.py      theme clustering (label propagation)
│   ├── style.py       style statistics + style card
│   ├── retrieve.py    offline hybrid retrieval (pluggable embeddings, RRF fusion) + graph derivation chains
│   ├── persona.py     reflection, contradictions, persona card, review queue, coverage
│   ├── chat.py        chat as them (three memory tiers)
│   ├── decide.py      what would they do (structured + narrative paths)
│   ├── sandbox.py     fiction sandbox (one-way isolation)
│   ├── evaluate.py    blind test, backtest
│   └── cli.py
├── prompts/           every prompt is an editable .md file
├── viewer/index.html  local-only graph / observation browser
├── examples/          sample chat, narrative, speakers.json
└── tests/test_pipeline.py
```

## Not done yet

- Vector retrieval: implement `retrieve.Embedder` (e.g. bge-m3), then `Retriever(store, embedder=...)` gives hybrid retrieval
- Dedicated parsers for WeChat and other platforms (currently generic csv/json/txt)
- A backstage web UI: currently CLI + static viewer
- Value weights in the structured decision path come from claim strength × confidence; there is no conjoint-style active questioning to calibrate them yet
- The multi-persona sandbox is a minimal scene loop ("them + one described character"); the director and narrative layers are only sketched
