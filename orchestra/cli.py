"""Terminal front-end: run a council without the web UI."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .doctor import preflight
from .orchestrator import Council, CouncilConfig
from .registry import Registry
from .report import to_markdown

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orchestra",
        description="Run a multi-model negotiation council on a task.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("models", help="list the registry and which models are live")
    ls.add_argument("--json", action="store_true")

    doc = sub.add_parser(
        "doctor",
        help="probe every configured model with one tiny call and report what's broken",
    )
    doc.add_argument("-p", "--panel", default="",
                     help="comma-separated keys to probe (default: all)")
    doc.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="run a council")
    run.add_argument("task", nargs="?", help="the task (omit to read stdin)")
    run.add_argument("-m", "--orchestrator", default="claude-opus-5",
                     help="the main model that chairs the council")
    run.add_argument("-p", "--panel", default="",
                     help="comma-separated model keys (default: every other model)")
    run.add_argument("-n", "--panel-size", type=int, default=8,
                     help="cap the auto-selected panel (default 8)")
    run.add_argument("-r", "--rounds", type=int, default=2,
                     help="negotiation rounds after cross-examination")
    run.add_argument("-t", "--target", type=float, default=0.78,
                     help="stop early at this convergence level")
    run.add_argument("--context", default="", help="extra context for the task")
    run.add_argument("--simulate", action="store_true",
                     help="force the offline simulator for every model")
    run.add_argument("-o", "--out", help="write the markdown report to this path")
    run.add_argument("--json-out", help="write the full JSON report to this path")
    run.add_argument("--quiet", action="store_true", help="only print the final report")
    return parser


def cmd_models(args: argparse.Namespace) -> int:
    registry = Registry()
    if args.json:
        print(json.dumps(registry.status(), indent=2))
        return 0
    live = 0
    print(f"{BOLD}{'KEY':<20} {'MODEL':<18} {'VENDOR':<16} {'LENS':<26} {'W':<5} STATUS{RESET}")
    for m in registry.status():
        state = {
            "direct": f"{GREEN}direct{RESET}",
            "openrouter": f"{CYAN}openrouter{RESET}",
        }.get(m["route"], f"{YELLOW}simulated{RESET} ({m['api_key_env']} unset)")
        live += 1 if m["live"] else 0
        chair = "*" if m["can_orchestrate"] else " "
        print(
            f"{chair}{m['key']:<19} {m['name']:<18} {m['vendor']:<16} "
            f"{m['lens']:<26} {m['weight']:<5} {state}"
        )
    routed = sum(1 for m in registry.status() if m["route"] == "openrouter")
    extra = f" ({routed} via OpenRouter)" if routed else ""
    print(f"\n{DIM}* = can chair the council. {live}/{len(registry.all())} models live{extra}.{RESET}")
    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    task = args.task or sys.stdin.read()
    if not task.strip():
        print("No task given.", file=sys.stderr)
        return 2

    registry = Registry(force_simulation=args.simulate)
    panel = (
        [k.strip() for k in args.panel.split(",") if k.strip()]
        if args.panel
        else [m.key for m in registry.all() if m.key != args.orchestrator][: args.panel_size]
    )

    try:
        config = CouncilConfig(
            orchestrator=args.orchestrator,
            panel=panel,
            rounds=args.rounds,
            convergence_target=args.target,
        )
        council = Council(registry, config)
    except (KeyError, ValueError) as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 2

    live = [k for k in [config.orchestrator, *council.panel_keys] if registry.is_live(k)]
    print(
        f"{DIM}chair {config.orchestrator} · {len(council.panel_keys)} panelists · "
        f"{args.rounds} rounds · up to {config.max_calls()} calls "
        f"({len(live)}/{len(council.panel_keys) + 1} models live){RESET}",
        file=sys.stderr,
    )

    printer = _make_printer(quiet=args.quiet)
    try:
        report = await council.run(task, args.context, emit=printer)
    finally:
        await registry.aclose()

    markdown = to_markdown(report)
    if args.out:
        Path(args.out).write_text(markdown)
        print(f"\n{DIM}markdown -> {args.out}{RESET}", file=sys.stderr)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"{DIM}json     -> {args.json_out}{RESET}", file=sys.stderr)
    if not args.out:
        print("\n" + markdown)
    return 0


def _make_printer(quiet: bool):
    async def emit(event: dict[str, Any]) -> None:
        if quiet:
            return
        kind = event.get("type")
        if kind == "phase":
            print(f"\n{BOLD}{CYAN}── {event['phase'].replace('_', ' ').upper()} ──{RESET}",
                  file=sys.stderr)
        elif kind == "model_done":
            if event.get("ok"):
                tag = f"{YELLOW}sim{RESET}" if event.get("simulated") else f"{GREEN}ok{RESET}"
                print(f"  {tag}  {event['model']:<20} {event['latency_s']}s "
                      f"{DIM}{event['output_tokens']} tok{RESET}", file=sys.stderr)
            else:
                print(f"  {RED}fail{RESET} {event['model']:<20} {event.get('error', '')[:90]}",
                      file=sys.stderr)
        elif kind == "round":
            d = event["data"]
            print(f"  {BOLD}round {d['round']}{RESET}: convergence {d['convergence']:.2f} "
                  f"({GREEN}{d['accepted']} accepted{RESET}, {d['contested']} contested, "
                  f"{d['rejected']} rejected)", file=sys.stderr)
        elif kind == "chair_note":
            print(f"  {DIM}chair: {event['text'][:160]}{RESET}", file=sys.stderr)
        elif kind in ("note", "warning"):
            print(f"  {YELLOW}{event['message']}{RESET}", file=sys.stderr)
        elif kind == "error":
            print(f"  {RED}{event['message']}{RESET}", file=sys.stderr)

    return emit


async def cmd_doctor(args: argparse.Namespace) -> int:
    registry = Registry()
    keys = [k.strip() for k in args.panel.split(",") if k.strip()] or None
    try:
        result = await preflight(registry, keys)
    finally:
        await registry.aclose()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if not result.broken else 1

    marks = {
        "ok": f"{GREEN}ok{RESET}",
        "no_key": f"{DIM}no key{RESET}",
        "unreachable": f"{RED}unreachable{RESET}",
        "blocked": f"{RED}blocked{RESET}",
        "stale_model_id": f"{RED}bad model id{RESET}",
        "no_json": f"{YELLOW}no json{RESET}",
    }
    for probe in result.probes:
        mark = marks.get(probe.status, probe.status)
        timing = f"{probe.latency_s}s" if probe.latency_s else ""
        print(f"  {mark:<22} {probe.key:<20} {timing:<8} {DIM}{probe.detail[:88]}{RESET}")

    print(
        f"\n{len(result.usable)} usable · {len(result.broken)} broken · "
        f"{len(result.unconfigured)} not configured"
    )
    blocked = [p for p in result.probes if p.status == "blocked"]
    if blocked:
        print(
            f"{YELLOW}{len(blocked)} model(s) blocked before the request left this "
            f"machine — a proxy, firewall or allowlist is refusing the endpoint, "
            f"not the provider. Nothing in models.yaml will fix that.{RESET}"
        )
    if any(p.status == "stale_model_id" for p in result.probes):
        print(
            f"{YELLOW}A 'bad model id' is usually a one-line fix in "
            f"orchestra/models.yaml — vendors rename models often.{RESET}"
        )
    if not result.can_run_a_council():
        print(f"{RED}Not enough working models for a live council "
              f"(need a chair plus at least one panelist).{RESET}")
        return 1
    return 1 if result.broken else 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "models":
        return cmd_models(args)
    if args.command == "doctor":
        return asyncio.run(cmd_doctor(args))
    return asyncio.run(cmd_run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
