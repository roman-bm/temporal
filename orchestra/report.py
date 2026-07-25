"""Markdown rendering of a finished council run."""

from __future__ import annotations

from typing import Any

_STATUS_MARK = {"accepted": "✓", "rejected": "✗", "contested": "~"}


def to_markdown(report: dict[str, Any]) -> str:
    cfg = report.get("config") or {}
    synth = report.get("synthesis") or {}
    ledger = report.get("ledger") or {}
    stats = report.get("stats") or {}
    lines: list[str] = []

    lines.append(f"# {synth.get('headline') or 'Council report'}")
    lines.append("")
    if report.get("any_simulated"):
        lines.append(
            "> **Simulated run.** One or more models had no API key and were "
            "replaced by the deterministic simulator. The content below is a "
            "protocol demonstration, not analysis."
        )
        lines.append("")

    lines.append(f"**Task:** {report.get('task', '').strip()}")
    lines.append("")
    lines.append(
        f"**Chair:** {cfg.get('orchestrator_name', cfg.get('orchestrator'))} · "
        f"**Panel:** {len(cfg.get('panel') or [])} models · "
        f"**Convergence:** {ledger.get('convergence', 0):.2f} · "
        f"**Confidence:** {synth.get('confidence', 0):.2f}"
    )
    lines.append("")

    if synth.get("analysis"):
        lines += ["## Analysis", "", synth["analysis"], ""]

    if synth.get("findings"):
        lines += ["## Findings", ""]
        lines += [f"{i}. {f}" for i, f in enumerate(synth["findings"], 1)]
        lines.append("")

    if synth.get("action_plan"):
        lines += ["## Action plan", ""]
        lines.append("| # | Action | First step | Effort | Impact | Owner |")
        lines.append("|---|--------|-----------|--------|--------|-------|")
        for i, item in enumerate(synth["action_plan"], 1):
            lines.append(
                f"| {i} | {_cell(item.get('action'))} | {_cell(item.get('first_step'))} "
                f"| {item.get('effort', '')} | {item.get('impact', '')} "
                f"| {_cell(item.get('owner_hint'))} |"
            )
        lines.append("")

    if synth.get("dissent"):
        lines += ["## Dissent register", "",
                  "_Positions the council could not settle. Read these before acting._", ""]
        lines += [f"- {d}" for d in synth["dissent"]]
        lines.append("")

    if synth.get("what_would_change_our_mind"):
        lines += ["## What would change this conclusion", ""]
        lines += [f"- {d}" for d in synth["what_would_change_our_mind"]]
        lines.append("")

    claims = ledger.get("claims") or []
    if claims:
        lines += ["## Claim ledger", ""]
        lines.append("| ID | Status | Support | Claim | Author |")
        lines.append("|----|--------|---------|-------|--------|")
        for c in sorted(claims, key=lambda c: (c["status"], -c["support"])):
            mark = _STATUS_MARK.get(c["status"], "?")
            lines.append(
                f"| {c['id']} | {mark} {c['status']} | {c['support']:+.2f} "
                f"| {_cell(c['text'])} | {c['author']} |"
            )
        lines.append("")

    rounds = ledger.get("rounds") or []
    if rounds:
        lines += ["## Negotiation rounds", ""]
        lines.append("| Round | Convergence | Accepted | Rejected | Contested | Claims that moved |")
        lines.append("|-------|-------------|----------|----------|-----------|-------------------|")
        for r in rounds:
            lines.append(
                f"| {r['round']} | {r['convergence']:.2f} | {r['accepted']} | "
                f"{r['rejected']} | {r['contested']} | {', '.join(r['moved']) or '—'} |"
            )
        lines.append("")

    if report.get("chair_notes"):
        lines += ["## Chair steering notes", ""]
        for i, note in enumerate(report["chair_notes"], 1):
            lines += [f"**Round {i + 1}.** {note}", ""]

    proposals = report.get("proposals") or []
    if proposals:
        lines += ["## Opening positions", ""]
        for p in proposals:
            tag = " _(simulated)_" if p.get("simulated") else ""
            lines.append(f"### {p['model_key']}{tag}")
            if p.get("error"):
                lines.append(f"- ⚠️ {p['error']}")
            if p.get("position"):
                lines.append(f"{p['position']}")
            lines.append("")

    lines += ["## Run stats", ""]
    lines.append(
        f"- Calls: {stats.get('calls', 0)} of a possible "
        f"{cfg.get('max_calls', stats.get('calls', 0))} "
        f"({stats.get('failures', 0)} failed, {stats.get('simulated_calls', 0)} simulated)"
    )
    if stats.get("repair_retries"):
        lines.append(
            f"- JSON repair retries: {stats['repair_retries']} "
            f"(a model replied in prose and was asked once more for the object)"
        )
    lines.append(
        f"- Tokens: {stats.get('input_tokens', 0):,} in / "
        f"{stats.get('output_tokens', 0):,} out"
    )
    lines.append(f"- Wall clock: {stats.get('wall_s', 0)}s")

    per_model = stats.get("per_model") or {}
    if per_model:
        lines += ["", "| Model | Calls | Failed | Tokens in | Tokens out | Latency |",
                  "|-------|-------|--------|-----------|------------|---------|"]
        for key, m in sorted(per_model.items(), key=lambda kv: -kv[1]["output_tokens"]):
            lines.append(
                f"| {key} | {m['calls']} | {m['failures']} | {m['input_tokens']:,} "
                f"| {m['output_tokens']:,} | {m['latency_s']}s |"
            )

    weights = (ledger.get("weights") or {})
    if weights:
        lines += ["", "**Final reliability weights** — how far each model's voting "
                  "influence drifted from its starting value across the rounds.", ""]
        lines.append("| Model | Weight |")
        lines.append("|-------|--------|")
        for key, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {key} | {w} |")
    lines.append("")
    lines.append("_Generated by the multi-model orchestration council._")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ").strip()
