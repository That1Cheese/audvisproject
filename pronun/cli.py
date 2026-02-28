"""CLI entry point for pronunciation correction system."""

import argparse
import random
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from pronun.data.word_lists import (
    ALL_WORDS, BEGINNER_WORDS, INTERMEDIATE_WORDS, ADVANCED_WORDS, PHONEME_FOCUS,
)
from pronun.data.sentence_lists import (
    ALL_SENTENCES, BEGINNER_SENTENCES, INTERMEDIATE_SENTENCES,
    ADVANCED_SENTENCES, SENTENCE_FOCUS,
)


console = Console()

LEVEL_COLORS = {
    "excellent": "green",
    "good": "blue",
    "fair": "yellow",
    "needs_work": "red",
}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def show_score_table(result: dict):
    """Display per-phoneme scoring as a rich table (legacy word mode)."""
    table = Table(title=f"Pronunciation: '{result['word']}'")
    table.add_column("Phoneme", style="cyan")
    table.add_column("Audio", justify="right")
    table.add_column("Visual", justify="right")
    table.add_column("Combined", justify="right")
    table.add_column("Level", style="bold")
    table.add_column("Tip")

    for fb in result["feedback"]:
        level_color = LEVEL_COLORS.get(fb["level"], "white")

        audio = f"{fb['score']:.0f}"
        visual = "-"
        if result.get("visual_score_b") is not None:
            visual = f"{result['visual_score_b']:.0f}"

        table.add_row(
            fb["phoneme"],
            audio,
            visual,
            f"{fb['score']:.0f}",
            f"[{level_color}]{fb['level']}[/{level_color}]",
            fb.get("tip") or "",
        )

    console.print(table)
    console.print()

    panel_text = (
        f"Audio: {result['audio_score']:.1f}/100  "
        f"Visual A: {result.get('visual_score_a', 'N/A')}  "
        f"Visual B: {result.get('visual_score_b', 'N/A')}  "
        f"Combined: {result['combined_score']:.1f}/100\n\n"
        f"{result['overall_feedback']}"
    )
    console.print(Panel(panel_text, title="Score Summary"))


def show_sentence_result(result: dict):
    """Display three-level scoring for a sentence practice result."""
    sentence = result["sentence"]
    sentence_score = result["sentence_score"]

    # 1. Sentence overview panel
    score_color = _score_color(sentence_score)
    console.print(Panel(
        f"[bold]{sentence}[/bold]\n\n"
        f"Sentence Score: [{score_color}]{sentence_score:.1f}/100[/{score_color}]  "
        f"Audio: {result['audio_score']:.1f}  "
        f"Visual: {_fmt_visual(result)}",
        title="Sentence Overview",
    ))

    # 2. Word scores table
    word_table = Table(title="Word Scores")
    word_table.add_column("#", justify="right", style="dim")
    word_table.add_column("Word", style="cyan")
    word_table.add_column("Score", justify="right")
    word_table.add_column("Level", style="bold")

    for i, ws in enumerate(result["word_scores"], 1):
        level_color = LEVEL_COLORS.get(ws["level"], "white")
        word_table.add_row(
            str(i),
            ws["word"],
            f"{ws['score']:.1f}",
            f"[{level_color}]{ws['level']}[/{level_color}]",
        )

    console.print(word_table)
    console.print()

    # 3. Phoneme detail table grouped by word
    phoneme_table = Table(title="Phoneme Details")
    phoneme_table.add_column("Word", style="dim")
    phoneme_table.add_column("Phoneme", style="cyan")
    phoneme_table.add_column("Score", justify="right")
    phoneme_table.add_column("Level", style="bold")
    phoneme_table.add_column("Tip")

    for fb in result["feedback"]:
        level_color = LEVEL_COLORS.get(fb["level"], "white")
        # Find which word this phoneme belongs to
        word_label = _phoneme_to_word(fb, result)
        phoneme_table.add_row(
            word_label,
            fb["phoneme"],
            f"{fb['score']:.1f}",
            f"[{level_color}]{fb['level']}[/{level_color}]",
            fb.get("tip") or "",
        )

    console.print(phoneme_table)
    console.print()

    # 4. Feedback panel
    console.print(Panel(result["overall_feedback"], title="Feedback"))
    console.print()


def show_trend_line(tracker, sentence: str):
    """Show a one-line trend if there are 2+ attempts for the same sentence."""
    history = tracker.get_history(sentence)
    if len(history) < 2:
        return
    scores = [f"{a['combined_score']:.1f}" for a in history]
    delta = history[-1]["combined_score"] - history[-2]["combined_score"]
    sign = "+" if delta >= 0 else ""
    console.print(
        f"[dim]Attempt {len(history)}:[/dim] "
        + " -> ".join(scores)
        + f"  ({sign}{delta:.1f} from last)"
    )


def show_progress(session):
    """Display full session progress table."""
    progress = session.get_progress()
    history = progress["history"]
    summary = progress["summary"]

    if not history:
        console.print("[dim]No attempts recorded yet.[/dim]")
        return

    table = Table(title="Session Progress")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Sentence")
    table.add_column("Score", justify="right")
    table.add_column("Trend", justify="center")

    prev_scores: dict[str, float] = {}
    for i, attempt in enumerate(history, 1):
        sent = attempt["sentence"]
        score = attempt["combined_score"]

        if sent in prev_scores:
            delta = score - prev_scores[sent]
            if delta > 0.5:
                arrow = "[green]^[/green]"
            elif delta < -0.5:
                arrow = "[red]v[/red]"
            else:
                arrow = "[dim]->[/dim]"
        else:
            arrow = ""

        prev_scores[sent] = score
        short_sent = sent if len(sent) <= 40 else sent[:37] + "..."
        table.add_row(str(i), short_sent, f"{score:.1f}", arrow)

    console.print(table)
    console.print()

    console.print(Panel(
        f"Total attempts: {summary['total_attempts']}  "
        f"Avg score: {summary['avg_score']:.1f}  "
        f"Best score: {summary['best_score']:.1f}  "
        f"Improvement: {summary['improvement']:+.1f}",
        title="Session Summary",
    ))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 85:
        return "green"
    elif score >= 70:
        return "blue"
    elif score >= 50:
        return "yellow"
    return "red"


def _fmt_visual(result: dict) -> str:
    va = result.get("visual_score_a")
    vb = result.get("visual_score_b")
    parts = []
    if vb is not None:
        parts.append(f"B:{vb:.1f}")
    if va is not None:
        parts.append(f"A:{va:.1f}")
    return ", ".join(parts) if parts else "N/A"


def _phoneme_to_word(fb_entry: dict, result: dict) -> str:
    """Find which word a phoneme feedback entry belongs to."""
    idx = None
    for i, pf in enumerate(result["feedback"]):
        if pf is fb_entry:
            idx = i
            break
    if idx is None:
        return ""
    for ws in result.get("word_scores", []):
        if ws.get("phoneme_start") is not None:
            if ws["phoneme_start"] <= idx < ws["phoneme_end"]:
                return ws["word"]
    return ""


def _pick_sentence(level: str, index: int = None) -> str:
    """Pick a sentence from the given level, optionally by index."""
    sentences = {
        "beginner": BEGINNER_SENTENCES,
        "intermediate": INTERMEDIATE_SENTENCES,
        "advanced": ADVANCED_SENTENCES,
    }.get(level, BEGINNER_SENTENCES)

    if index is not None:
        if 0 <= index < len(sentences):
            return sentences[index]
        console.print(f"[yellow]Index {index} out of range, picking random.[/yellow]")
    return random.choice(sentences)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_practice(args):
    """Sentence-based interactive practice loop."""
    from pronun.workflow.session import Session

    level = getattr(args, "level", "beginner")
    index = getattr(args, "index", None)
    mode = getattr(args, "mode", "B")

    console.print(f"[bold]Pronunciation Practice[/bold] (Level: {level}, Mode: {mode})")
    console.print("[dim]Interactive session — press Enter for next, 'r' to retry, 'p' for progress, 'q' to quit[/dim]")
    console.print()

    with Session(use_camera=args.camera, mode=mode) as session:
        current_sentence = _pick_sentence(level, index)

        while True:
            console.print(f"[bold cyan]Say: \"{current_sentence}\"[/bold cyan]")
            console.print("Recording... (speak now, silence to stop)")

            result = session.practice_sentence(current_sentence)
            show_sentence_result(result)
            show_trend_line(session.tracker, current_sentence)

            # Interactive prompt
            try:
                choice = console.input(
                    "\n[dim]Enter=next, r=retry, p=progress, q=quit:[/dim] "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if choice == "q":
                break
            elif choice == "r":
                continue  # retry same sentence
            elif choice == "p":
                show_progress(session)
                # After showing progress, prompt again
                try:
                    choice2 = console.input(
                        "\n[dim]Enter=next, r=retry, q=quit:[/dim] "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                if choice2 == "q":
                    break
                elif choice2 == "r":
                    continue
                else:
                    current_sentence = _pick_sentence(level)
            else:
                current_sentence = _pick_sentence(level)

        # Show final summary on exit
        console.print()
        show_progress(session)
        console.print("[bold]Session complete.[/bold]")


def cmd_practice_word(args):
    """Legacy single-word practice mode."""
    from pronun.workflow.session import Session

    words = args.words if args.words else BEGINNER_WORDS[:3]
    mode = args.mode

    console.print(f"[bold]Word Practice[/bold] (Mode: {mode})")
    console.print(f"Words: {', '.join(words)}")
    console.print()

    with Session(use_camera=args.camera, mode=mode) as session:
        for word in words:
            console.print(f"[bold cyan]Say: '{word}'[/bold cyan]")
            console.print("Recording... (speak now, silence to stop)")

            result = session.practice_word(word)
            show_score_table(result)
            console.print()


def cmd_list(args):
    """List available practice sentences or words."""
    level = args.level

    if level == "focus":
        console.print("[bold]Phoneme Focus Sentences:[/bold]")
        for group, sents in SENTENCE_FOCUS.items():
            console.print(f"\n[bold]{group}[/bold]:")
            for i, s in enumerate(sents, 1):
                console.print(f"  {i}. {s}")
        return

    sentences = {
        "beginner": BEGINNER_SENTENCES,
        "intermediate": INTERMEDIATE_SENTENCES,
        "advanced": ADVANCED_SENTENCES,
    }.get(level)

    if sentences is None:
        sentences = ALL_SENTENCES

    console.print(f"[bold]{level.title()} Sentences:[/bold]")
    for i, s in enumerate(sentences, 1):
        console.print(f"  {i:2d}. {s}")


def cmd_list_words(args):
    """List available practice words (legacy)."""
    level = args.level

    if level == "focus":
        for group, group_words in PHONEME_FOCUS.items():
            console.print(f"[bold]{group}[/bold]: {', '.join(group_words)}")
        return

    words = {
        "beginner": BEGINNER_WORDS,
        "intermediate": INTERMEDIATE_WORDS,
        "advanced": ADVANCED_WORDS,
    }.get(level, ALL_WORDS)

    console.print(f"[bold]{level.title()} Words:[/bold]")
    for i, w in enumerate(words, 1):
        console.print(f"  {i:2d}. {w}")


def cmd_progress(args):
    """Show session progress (only meaningful inside interactive loop)."""
    console.print("[dim]Progress is shown during an interactive practice session.[/dim]")
    console.print("[dim]Run: pronun practice --level beginner[/dim]")


def cmd_compare(args):
    """Compare Mode A vs Mode B on the same word."""
    from pronun.workflow.session import Session
    from pronun.workflow.comparison import format_comparison

    word = args.word
    console.print(f"[bold]Comparing modes for '{word}'[/bold]")
    console.print("Recording... (speak now)")

    with Session(use_camera=True, mode="both") as session:
        result = session.practice_word(word)
        show_score_table(result)

        if result.get("visual_score_a") is not None and result.get("visual_score_b") is not None:
            console.print(f"\nMode A score: {result['visual_score_a']:.1f}")
            console.print(f"Mode B score: {result['visual_score_b']:.1f}")


def cmd_train(args):
    """Train HMM emission parameters from GRID corpus."""
    from pronun.training.train_emissions import train_from_grid

    def progress(speaker, clip_idx, clip_name):
        if clip_idx % 10 == 0:
            console.print(f"  [dim]{speaker}/{clip_name} (clip {clip_idx})[/dim]")

    console.print(f"[bold]Training HMM emissions[/bold]")
    console.print(f"Speakers: {', '.join(args.speakers)}")
    if args.max_clips:
        console.print(f"Max clips per speaker: {args.max_clips}")
    console.print()

    store, summary = train_from_grid(
        corpus_dir=args.grid_dir,
        speakers=args.speakers,
        max_clips_per_speaker=args.max_clips,
        output_path=args.output,
        progress_callback=progress,
    )

    console.print(f"\n[bold green]Training complete.[/bold green]")
    console.print(f"Trained {len(store)} visemes:")
    for vid, count in sorted(summary.items()):
        console.print(f"  Viseme {vid:2d}: {count:6d} observations")


def cmd_calibrate(args):
    """Calibrate reference baseline from GRID corpus."""
    from pronun.training.train_emissions import calibrate_baseline
    from pronun.visual.scoring.emission_store import EmissionStore
    from pronun.config import EMISSION_STORE_PATH

    emissions_path = args.emissions or str(EMISSION_STORE_PATH)

    try:
        emission_store = EmissionStore.from_file(emissions_path)
    except FileNotFoundError:
        console.print(f"[red]Emission store not found at {emissions_path}[/red]")
        console.print("Run 'train' first to generate emission parameters.")
        sys.exit(1)

    def progress(speaker, clip_idx, clip_name):
        if clip_idx % 10 == 0:
            console.print(f"  [dim]{speaker}/{clip_name} (clip {clip_idx})[/dim]")

    console.print(f"[bold]Calibrating reference baseline[/bold]")
    console.print(f"Speakers: {', '.join(args.speakers)}")
    console.print()

    kwargs = dict(
        corpus_dir=args.grid_dir,
        speakers=args.speakers,
        emission_store=emission_store,
        max_clips_per_speaker=args.max_clips,
        output_path=args.output,
        progress_callback=progress,
    )
    if args.n_aug is not None:
        kwargs["n_aug"] = args.n_aug

    baseline = calibrate_baseline(**kwargs)

    console.print(f"\n[bold green]Calibration complete.[/bold green]")
    console.print(f"Default reference: {baseline.default_reference:.4f}")
    console.print(f"Bandwidth: {baseline.bandwidth:.4f}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pronunciation Correction System",
        prog="pronun",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Practice command (sentence-based, interactive)
    p_practice = subparsers.add_parser("practice", help="Practice sentence pronunciation")
    p_practice.add_argument("--level", choices=["beginner", "intermediate", "advanced"],
                            default="beginner", help="Sentence difficulty level")
    p_practice.add_argument("--index", type=int, default=None,
                            help="Specific sentence index from the level list")
    p_practice.add_argument("--mode", choices=["A", "B", "both"], default="B",
                            help="Visual scoring mode")
    p_practice.add_argument("--no-camera", dest="camera", action="store_false",
                            help="Disable webcam (audio only)")
    p_practice.set_defaults(func=cmd_practice)

    # Practice-word command (legacy word mode)
    p_word = subparsers.add_parser("practice-word", help="Practice individual word pronunciation")
    p_word.add_argument("words", nargs="*", help="Words to practice")
    p_word.add_argument("--mode", choices=["A", "B", "both"], default="B",
                        help="Visual scoring mode")
    p_word.add_argument("--no-camera", dest="camera", action="store_false",
                        help="Disable webcam (audio only)")
    p_word.set_defaults(func=cmd_practice_word)

    # List command (sentences)
    p_list = subparsers.add_parser("list", help="List practice sentences")
    p_list.add_argument("level", nargs="?", default="all",
                        choices=["beginner", "intermediate", "advanced", "focus", "all"])
    p_list.set_defaults(func=cmd_list)

    # List-words command (legacy)
    p_listw = subparsers.add_parser("list-words", help="List practice words")
    p_listw.add_argument("level", nargs="?", default="all",
                         choices=["beginner", "intermediate", "advanced", "focus", "all"])
    p_listw.set_defaults(func=cmd_list_words)

    # Progress command
    p_progress = subparsers.add_parser("progress", help="Show session progress")
    p_progress.set_defaults(func=cmd_progress)

    # Compare command
    p_compare = subparsers.add_parser("compare", help="Compare Mode A vs Mode B")
    p_compare.add_argument("word", help="Word to compare")
    p_compare.set_defaults(func=cmd_compare)

    # Train command
    p_train = subparsers.add_parser("train", help="Train HMM emissions from GRID corpus")
    p_train.add_argument("--grid-dir", required=True, help="Path to grid_corpus directory")
    p_train.add_argument("--speakers", nargs="+", required=True,
                         help="Speaker IDs (e.g. s1 s2 s3)")
    p_train.add_argument("--max-clips", type=int, default=0,
                         help="Max clips per speaker (0=all)")
    p_train.add_argument("--output", default=None,
                         help="Output path for emission store")
    p_train.set_defaults(func=cmd_train)

    # Calibrate command
    p_cal = subparsers.add_parser("calibrate", help="Calibrate reference baseline from GRID corpus")
    p_cal.add_argument("--grid-dir", required=True, help="Path to grid_corpus directory")
    p_cal.add_argument("--speakers", nargs="+", required=True,
                       help="Speaker IDs (e.g. s1 s2)")
    p_cal.add_argument("--max-clips", type=int, default=0,
                       help="Max clips per speaker (0=all)")
    p_cal.add_argument("--emissions", default=None,
                       help="Path to trained emission store (default: ~/.cache/pronun/viseme_emissions.npz)")
    p_cal.add_argument("--output", default=None,
                       help="Output path for reference baseline")
    p_cal.add_argument("--n-aug", type=int, default=None,
                       help="Number of augmented copies per clip (default: AUG_COUNT=3, use 0 for no augmentation)")
    p_cal.set_defaults(func=cmd_calibrate)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
