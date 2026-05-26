"""
Interactive MySQL chatbot — entry point.

Run:
    python main.py
    python main.py --config path/to/config.yaml
"""

import argparse
import os
import sys

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from agent import HermesAgent
from db_tool import create_db_tool

console = Console()


# ─────────────────────────────────────────────────────────────
# Startup helpers
# ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def load_skills(config: dict) -> str:
    skills_file = config.get("claude", {}).get("skills_file", "")
    if not skills_file or not os.path.exists(skills_file):
        return ""
    with open(skills_file, "r") as fh:
        text = fh.read().strip()
    console.print(f"[dim]Loaded domain knowledge from [bold]{skills_file}[/bold] ({len(text):,} chars)[/dim]")
    return text


def print_welcome(model: str) -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]MySQL Chatbot[/bold cyan]  —  powered by [bold]{model}[/bold]\n"
            "[dim]Read-only MySQL access · Session memory enabled[/dim]\n\n"
            "[yellow]/schema[/yellow]   Show full database schema\n"
            "[yellow]/reset[/yellow]    Clear conversation history\n"
            "[yellow]/history[/yellow]  Print how many messages are in context\n"
            "[yellow]/quit[/yellow]     Exit",
            border_style="cyan",
        )
    )


# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("sk-ant-..."):
        console.print(
            "[red bold]Error:[/red bold] ANTHROPIC_API_KEY is not set.\n"
            "Create a [bold].env[/bold] file (copy [bold].env.example[/bold]) "
            "and add your API key."
        )
        sys.exit(1)

    config = load_config(config_path)
    db_tool = create_db_tool(config)
    skills = load_skills(config)
    agent = HermesAgent(config, db_tool, skills_text=skills)

    print_welcome(config["claude"]["model"])

    while True:
        try:
            user_input = console.input("\n[bold green]You >[/bold green] ").strip()

            if not user_input:
                continue

            # ── built-in slash commands ──────────────────────────────
            cmd = user_input.lower()

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Bye![/dim]")
                break

            if cmd == "/reset":
                agent.reset()
                console.print("[dim]Session cleared.[/dim]")
                continue

            if cmd == "/history":
                console.print(
                    f"[dim]{agent.history_length} messages currently in context.[/dim]"
                )
                continue

            if cmd == "/schema":
                schema = db_tool.get_schema()
                if "error" in schema:
                    console.print(f"[red]{schema['error']}[/red]")
                else:
                    for table, cols in schema["schema"].items():
                        console.print(Rule(f"[bold]{table}[/bold]"))
                        for col in cols:
                            console.print(
                                f"  [cyan]{col['Field']}[/cyan]  "
                                f"[dim]{col['Type']}[/dim]"
                                + (f"  [yellow]PK[/yellow]" if col.get("Key") == "PRI" else "")
                            )
                continue

            # ── normal message → agent ───────────────────────────────
            console.print()
            gen = agent.chat_stream(user_input)
            chunks: list[str] = []

            # Spinner shows while Claude is calling tools (no text yet).
            # Exits as soon as the first text token arrives.
            with console.status("[dim]Thinking…[/dim]", spinner="dots"):
                for chunk in gen:
                    chunks.append(chunk)
                    break  # first token received — stop spinner

            console.print(Rule(style="dim"))
            if chunks:
                print(chunks[0], end="", flush=True)
                for chunk in gen:
                    print(chunk, end="", flush=True)
                    chunks.append(chunk)
                print()  # final newline

        except KeyboardInterrupt:
            console.print("\n[dim](Ctrl-C) Type [bold]/quit[/bold] to exit.[/dim]")
        except EOFError:
            console.print("\n[dim]EOF — exiting.[/dim]")
            break
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red bold]Error:[/red bold] {exc}")


# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MySQL Chatbot powered by Claude")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in current directory)",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
