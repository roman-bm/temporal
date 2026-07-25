# Council

**One model you choose chairs a panel of frontier models from rival providers.
They propose, cross-examine, and negotiate over a shared claim ledger. The chair
synthesises the result into analysis plus an executable plan — and reports what
the panel never agreed on instead of averaging it away.**

Most "multi-model" tools fan a prompt out to N models and concatenate the
answers. That gives you N opinions and no resolution. This does something
different: it runs a structured negotiation with an explicit ledger, weighted
voting, and a measurable convergence signal, so the output is a *position the
council arrived at* rather than a pile of parallel monologues.

---

## The protocol

```
                    ┌──────────────────────────────────────────────┐
   TASK ───────────►│ 1. FRAMING           (chair, alone)          │
                    │    objective · success criteria · briefs     │
                    └───────────────────┬──────────────────────────┘
                                        │  one tailored brief per panelist
                    ┌───────────────────▼──────────────────────────┐
                    │ 2. PROPOSALS         (panel, in parallel)    │
                    │    each model → 3-5 falsifiable claims       │
                    └───────────────────┬──────────────────────────┘
                                        │  claims pooled + deduped → LEDGER
                    ┌───────────────────▼──────────────────────────┐
                    │ 3. CROSS-EXAMINATION (panel, in parallel)    │
                    │    every model votes on every claim          │
                    └───────────────────┬──────────────────────────┘
                                        │  support · participation · status
                    ┌───────────────────▼──────────────────────────┐
                    │ 4. NEGOTIATION       (N rounds)              │◄──┐
                    │    chair writes a steering note on the crux  │   │
                    │    panel re-votes on contested claims only   │───┘
                    └───────────────────┬──────────────────────────┘
                       convergence ≥ target, or nothing contested
                    ┌───────────────────▼──────────────────────────┐
                    │ 5. SYNTHESIS         (chair, alone)          │
                    │    analysis · action plan · DISSENT REGISTER │
                    └──────────────────────────────────────────────┘
```

### Why this beats a fan-out

| Mechanism | What it prevents |
|---|---|
| **Assigned lenses.** Each panelist is briefed to argue from one angle (formal verification, cost, red team, regulatory…). The chair is told to point at least two models at the *most attractive* answer. | Twelve models producing the same centrist take. |
| **Claims, not essays.** Every substantive statement becomes an ID'd, falsifiable claim. Models vote on claims, not vibes. | Disagreement hiding inside prose nobody reconciles. |
| **Abstention ≠ agreement.** Abstaining lowers a claim's *participation* rather than its support, and a claim below 50% participation can never read as settled. | A claim nobody engaged with being reported as consensus. |
| **Amendments are mandatory on dispute.** A model that disputes must supply the rewrite that would move it to endorse. | Deadlock with no path out. |
| **Reliability weighting, graded on a curve.** Models that land where the council settles gain voting weight; persistent outliers lose it. Scored against the panel's *own mean* accuracy, not a fixed bar, so total voting mass stays constant and uniform agreement moves nobody. Capped at ±10% per round. | One confident model dominating — and, in the other direction, weight inflation quietly pushing the whole panel to the ceiling until the mechanism stops discriminating. |
| **The dissent register is a first-class output.** Contested claims ship with who held out and why. | False confidence, the most dangerous failure mode of ensemble methods. |

### The consensus math

For each claim, every voter contributes weight `w = model_weight × vote_confidence`
and a stance value (`endorse +1`, `dispute −1`, `abstain 0`):

```
support       = Σ(w·v) / Σ(w·|v|)      ∈ [−1, +1]
participation = Σ(w·|v|) / Σ(w_all)    ∈ [0, 1]
```

A claim is **accepted** at `support ≥ 0.5` (a 3:1 weighted supermajority),
**rejected** at `≤ −0.5`, and **contested** otherwise — including whenever
participation is below 0.5, regardless of support.

Convergence is the mean `|support|` across the ledger, discounted by
participation. It rises as the council settles, and is the stopping signal for
negotiation.

---

## Quick start

```bash
pip install -e .
cp .env.example .env          # add whichever provider keys you have

orchestra-server              # http://127.0.0.1:8000
```

**No keys?** It still runs. Every model without a key falls back to a
deterministic simulator so you can inspect the whole protocol offline. Simulated
content is labelled `SIMULATED` in the UI, the CLI, the markdown export and the
JSON — there is no silent substitution, and simulated output is **not analysis**.

### CLI

```bash
orchestra models                        # who's live, who's simulated

orchestra run "Should we migrate the billing monolith to event sourcing? \
  40 engineers, flat budget, weekly releases." \
  --orchestrator claude-opus-5 \
  --panel gpt-5,gemini-2.5-pro,grok-4,deepseek-reasoner,mistral-large \
  --rounds 3 \
  --out decision.md
```

| Flag | Meaning |
|---|---|
| `-m, --orchestrator` | the chair (default `claude-opus-5`) |
| `-p, --panel` | comma-separated keys (default: every other model, capped by `-n`) |
| `-r, --rounds` | negotiation rounds after cross-examination (default 2) |
| `-t, --target` | stop early at this convergence level (default 0.78) |
| `--simulate` | force the simulator for every model |
| `-o` / `--json-out` | write the markdown / full JSON report |

### Running it from your phone

The UI is responsive and the whole flow — including the live SSE stream — works
at phone width. There's no app to install; you point the phone's browser at a
machine running the server.

**⚠ Read this first.** Binding past localhost so a phone can reach the UI also
exposes it to everyone else on that network, and `POST /api/sessions` spends
real money against whatever provider keys are loaded. Always set a token when
you expose it:

```bash
export ORCHESTRA_TOKEN=$(openssl rand -hex 16)
HOST=0.0.0.0 orchestra-server
```

The server prints the exact URL to open, token included. The API rejects any
request without it; the page itself loads but can't do anything, and says so.
The browser stashes the token in `sessionStorage` and strips it from the visible
URL so it doesn't leak into a screenshot or a shared link. Start without
`ORCHESTRA_TOKEN` on a non-loopback host and the server prints a loud warning
instead.

| Where the server runs | How the phone reaches it | Good for |
|---|---|---|
| **Your laptop, same Wi-Fi** | `http://<laptop-ip>:8000/?t=<token>` — find the IP with `ipconfig getifaddr en0` (macOS) or `hostname -I` (Linux) | The common case. Nothing leaves your network. |
| **A tunnel** (`cloudflared tunnel --url http://localhost:8000`, `ngrok http 8000`) | The public HTTPS URL the tunnel prints, plus `?t=<token>` | Reaching it off your home network. The token is doing real work here — treat the URL as a credential. |
| **A small VPS / Fly / Railway** | Its hostname, plus `?t=<token>` | Always-on. Put the provider keys in the host's secret store, not a committed `.env`. |

Two things to know before you rely on it: iOS Safari suspends background tabs,
so a long run can stall if you switch apps mid-council — the SSE stream replays
its full backlog on reconnect, so pulling to refresh recovers the run rather
than losing it. And sessions live in memory, so a server restart drops them;
download the markdown while the report is on screen.

Running Python *on* the phone (Termux on Android, a-Shell on iOS) technically
works but isn't worth it — the panel is network-bound anyway, so the phone may
as well just be the screen.

### Library

```python
from orchestra import Council, CouncilConfig, Registry

registry = Registry()
council = Council(registry, CouncilConfig(
    orchestrator="claude-fable-5",
    panel=["gpt-5", "gemini-2.5-pro", "grok-4", "deepseek-reasoner"],
    rounds=3,
))

report = await council.run("Pick a vector database for 50M embeddings.")
print(report["synthesis"]["headline"])
for claim in report["ledger"]["claims"]:
    print(claim["status"], claim["support"], claim["text"])
```

---

## The roster

Fifteen models across twelve providers. Five can chair.

| Model | Vendor | Lens | Chair |
|---|---|---|:---:|
| Claude Opus 5 | Anthropic | systems-architecture | ● |
| Claude Fable 5 | Anthropic | first-principles | ● |
| Claude Sonnet 5 | Anthropic | implementation-pragmatics | ● |
| GPT-5 | OpenAI | quantitative-rigor | ● |
| Gemini 2.5 Pro | Google | evidence-and-sources | ● |
| Grok 4 | xAI | contrarian-red-team | |
| DeepSeek R1 | DeepSeek | formal-verification | |
| Mistral Large | Mistral AI | regulatory-and-compliance | |
| Llama 4 Maverick | Meta (via Together) | open-source-ecosystem | |
| Qwen Max | Alibaba | apac-market-reality | |
| Kimi K2 | Moonshot AI | long-document-synthesis | |
| Command A | Cohere | enterprise-operations | |
| GLM-4.6 | Z.ai | cost-benefit-engineering | |
| Sonar Pro | Perplexity | external-evidence | |
| Nova Pro | Amazon | cost-and-operations | |

**Model IDs move fast.** Everything above lives in
[`orchestra/models.yaml`](orchestra/models.yaml) — lens, weight, endpoint, chair
eligibility, the lot. If a call 404s, edit the YAML, not the code. Adding a
sixteenth model is a YAML entry rather than a code change, as long as it speaks
either the Anthropic API or an OpenAI-compatible `/chat/completions`.

### Two backends, twelve providers

- **Anthropic** models go through the official SDK, with adaptive thinking
  (`display: "summarized"`, so the UI can show reasoning progress rather than a
  silent pause) and a per-model `effort` setting. Server-side refusal fallbacks
  are requested by default: Opus 5 and Fable 5 run elevated safety classifiers
  that occasionally decline benign security- or life-science-adjacent work, and
  `fallbacks: "default"` re-serves those on Anthropic's recommended substitute
  inside the same call. If the installed SDK can't type the parameter, the
  provider retries once without it instead of failing the run.
- **Everything else** goes through one OpenAI-compatible adapter. Google, Cohere
  and Bedrock all expose compatibility endpoints, so twelve vendors need exactly
  one HTTP client and the differences live in YAML.

---

## What you get back

Both the UI and `report.md` give you:

- **Headline + analysis** built on accepted claims (the chair may overrule the
  ledger, but must say so and give a reason).
- **Action plan** — sequenced, with a concrete first step, effort/impact, and an
  owner hint per item.
- **Dissent register** — contested claims with the holdouts and their arguments.
- **What would change this conclusion** — the falsification list.
- **Full claim ledger** — every claim with its support, participation, and the
  vote-by-vote record including stance changes between rounds.
- **Run stats** — calls made against the ceiling, failures, simulated calls,
  per-model token usage and latency, and the final reliability weights showing
  how far each model's influence drifted.

The JSON export additionally carries every proposal, the chair's steering notes,
per-round convergence, and the final reliability weights.

---

## Design decisions worth arguing with

**Convergence is not the goal.** The prompts explicitly tell panelists not to
converge for its own sake, and an honest 60/40 split is reported as a split. If
you want one confident answer regardless of the evidence, this is the wrong tool.

**The chair is not a vote counter.** It sees everything the panel produced and is
licensed to overrule the ledger with a stated reason. Pure aggregation produces
worse answers than a strong synthesiser with full visibility.

**Failure degrades, it doesn't abort.** A dead provider, a malformed JSON reply
or a safety refusal removes one voice; the council continues and the report names
which model dropped out and why. A model that answers in prose keeps its position
but contributes no claims — inventing claim boundaries on its behalf would
misattribute them. Phases run models concurrently, so every call also carries a
hard deadline: past its timeout plus a grace window a straggler becomes an error
result and the round proceeds without it, rather than one wedged connection
holding the whole panel.

**Cost is shown before you spend it.** A run costs at most
`(2 + rounds) × (1 + panelists)` calls — the chair spends framing, one steering
note per round, and synthesis; each panelist spends a proposal, a cross-exam
ballot, and one ballot per round. The UI and CLI both print that ceiling up
front, split by live vs simulated, because fifteen models at five rounds is 105
calls and that shouldn't be a surprise on a billing page. Start with 4–6
panelists and 2 rounds; add models for genuinely contested decisions, not for
everything.

**Sessions are memory-only.** Runs aren't persisted — the task box routinely
receives things people wouldn't want on disk. Export the markdown if you want a
record.

---

## Development

```bash
pip install -e ".[dev]"
pytest                        # 68 tests, no network, no keys required
ORCHESTRA_SIMULATE=1 orchestra-server
```

The suite covers the consensus math (weighting, abstention, thresholds,
weight conservation under the curve-graded reliability update), JSON recovery
from realistically messy model output, the full protocol end-to-end against the
simulator, call budgeting, the API token gate, and provider failure paths — missing keys, connection
errors, `json_object` rejection, hung providers hitting the deadline, Anthropic
refusals and old-SDK fallback degradation.

| Env var | Default | Purpose |
|---|---|---|
| `ORCHESTRA_SIMULATE` | unset | `1` forces the simulator for every model |
| `ORCHESTRA_CONCURRENCY` | `8` | max parallel model calls |
| `ORCHESTRA_MODELS` | bundled | path to an alternate `models.yaml` |
| `ORCHESTRA_TOKEN` | unset | shared token required on `/api/*`; set it whenever you bind past localhost |
| `PORT` / `HOST` | `8000` / `127.0.0.1` | server bind — `HOST=0.0.0.0` to reach it from another device |

### Layout

```
orchestra/
  models.yaml       the roster — edit this, not the code
  registry.py       loads specs, routes model → backend, live vs simulated
  providers/        anthropic (SDK) · openai_compat (12 vendors) · simulated
  protocol.py       every phase prompt; the protocol *is* the prompts
  consensus.py      the ledger: claims, weighted votes, convergence
  orchestrator.py   the five-phase engine
  server.py         FastAPI + SSE
  cli.py            terminal front-end
  static/           the UI (vanilla JS, no build step)
```
