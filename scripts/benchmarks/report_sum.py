#!/usr/bin/env python3
"""Generate a report-ready markdown summary for Pacman benchmark outputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def _rel(base: Path, target: Path) -> str:
    return target.relative_to(base).as_posix()


def _first_match(graphs_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(graphs_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _figure_row(base: Path, title: str, figure_path: Path | None) -> str:
    if figure_path is None:
        return f"| {title} | TODO | _Missing (add after run/baseline export)_ |"
    rel = _rel(base, figure_path)
    return f"| {title} | [{figure_path.name}]({rel}) | ![{title}]({rel}) |"


def _parse_mods(raw_mods: str) -> list[str]:
    return [m.strip() for m in raw_mods.split(",") if m.strip()]


def _video_link_for_mod(base: Path, videos: list[Path], mod_name: str) -> str:
    for video in videos:
        name = video.name
        if f"_{mod_name}_" in name or name.endswith(f"_{mod_name}.mp4"):
            rel = _rel(base, video)
            return f"[{video.name}]({rel})"
    return "TODO"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="Output directory of run_pacman.sh")
    parser.add_argument("--mods", default="", help="Comma-separated mod list")
    parser.add_argument("--with-ppo", type=int, default=0, help="1 if PPO section was enabled")
    parser.add_argument("--video-model", default="pixel", help="pixel|object|both")
    parser.add_argument("--quick", type=int, default=1, help="1 for quick mode, 0 for full mode")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    logs_dir = out_dir / "logs"
    metrics_dir = out_dir / "metrics"
    graphs_dir = out_dir / "graphs"
    videos_dir = out_dir / "videos"
    meta_dir = out_dir / "meta"
    cfg_dir = meta_dir / "config_snapshots"

    graphs = sorted(graphs_dir.glob("*.png"))
    videos = sorted(videos_dir.glob("*.mp4"))
    metrics = sorted(metrics_dir.glob("*"))
    logs = sorted(logs_dir.glob("*.log"))
    mods = _parse_mods(args.mods)

    required_figures: list[tuple[str, list[str]]] = [
        (
            "Figure 1a - Object-Centric PQN Episode Reward",
            ["*pqn*oc*reward*.png", "*pqn*object*reward*.png"],
        ),
        (
            "Figure 1b - Object-Centric PQN Training Loss",
            ["*pqn*oc*loss*.png", "*pqn*object*loss*.png"],
        ),
        (
            "Figure 2a - Pixel PQN Episode Reward",
            ["*pqn*pixel*reward*.png"],
        ),
        (
            "Figure 2b - Pixel PQN Training Loss",
            ["*pqn*pixel*loss*.png"],
        ),
    ]

    if args.with_ppo == 1:
        required_figures.extend(
            [
                (
                    "Figure 4a - Object-Centric PPO Episode Reward (Optional)",
                    ["*ppo*oc*reward*.png", "*ppo*object*reward*.png"],
                ),
                (
                    "Figure 4b - Object-Centric PPO Training Loss (Optional)",
                    ["*ppo*oc*loss*.png", "*ppo*object*loss*.png"],
                ),
            ]
        )

    figure_rows = []
    for title, patterns in required_figures:
        figure_rows.append(_figure_row(out_dir, title, _first_match(graphs_dir, patterns)))

    md_path = out_dir / "PACMAN_REPORT_DRAFT.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode_label = "quick" if args.quick == 1 else "full"

    lines: list[str] = []
    lines.append("# Pacman Report Draft (Auto-Generated)")
    lines.append("")
    lines.append(f"- Generated at: `{ts}`")
    lines.append(f"- Output folder: `{out_dir}`")
    lines.append(f"- Mode: `{mode_label}`")
    lines.append(f"- Video model: `{args.video_model}`")
    lines.append(f"- PPO included: `{bool(args.with_ppo)}`")
    lines.append("- PPO mode: `object_only`")
    lines.append("")
    lines.append("## 1. Output Package")
    lines.append("- This folder already contains logs, metrics, graphs, videos, configs, and this markdown file.")
    lines.append("- You can zip the full output folder and submit/share directly.")
    lines.append("")
    lines.append("### Artifact Index")
    lines.append(f"- Logs: `{len(logs)}` file(s) in `logs/`")
    lines.append(f"- Metrics: `{len(metrics)}` file(s) in `metrics/`")
    lines.append(f"- Graphs: `{len(graphs)}` file(s) in `graphs/`")
    lines.append(f"- Videos: `{len(videos)}` file(s) in `videos/`")
    run_summary = out_dir / "meta" / "run_summary.txt"
    hardware_info = out_dir / "meta" / "hardware_info.txt"
    if run_summary.exists():
        rel = _rel(out_dir, run_summary)
        lines.append(f"- Run summary: [{run_summary.name}]({rel})")
    if hardware_info.exists():
        rel = _rel(out_dir, hardware_info)
        lines.append(f"- Hardware info: [{hardware_info.name}]({rel})")
    lines.append("")
    lines.append("## 2. Coverage vs Report Checklist")
    lines.append("- PQN object/pixel training: covered by run script.")
    lines.append("- PQN base + per-mod videos: covered by run script.")
    if args.with_ppo == 1:
        lines.append("- Optional PPO section (object-centric): enabled in this run.")
    else:
        lines.append("- Optional PPO section (object-centric): not enabled in this run.")
    lines.append("- Baseline figure (external PPO/PQN/DQN source): **manual add required**.")
    lines.append("- Modification behavior descriptions: fill section 4 below after video review.")
    lines.append("")
    lines.append("## 3. Figures To Paste Into Report")
    lines.append("| Figure | Source (name/link) | Preview |")
    lines.append("|---|---|---|")
    lines.extend(figure_rows)
    lines.append("| Figure 3 - Baseline external graph | TODO: add source link/citation | TODO |")
    lines.append("")
    lines.append("### Extra Graph Files")
    if graphs:
        for g in graphs:
            rel = _rel(out_dir, g)
            lines.append(f"- [{g.name}]({rel})")
    else:
        lines.append("- No graph files found yet.")
    lines.append("")
    lines.append("## 4. Modifications (Fill Behavior Notes)")
    if not mods:
        lines.append("- No explicit mod list was provided to markdown generator.")
        lines.append("- Add your mod names and behavior notes manually.")
    else:
        for mod in mods:
            lines.append(f"### {mod}")
            lines.append("- Base env behavior: TODO")
            lines.append("- Mod env behavior: TODO")
            lines.append(f"- Video link: {_video_link_for_mod(out_dir, videos, mod)}")
            lines.append("")
    lines.append("## 5. Videos (Base + Mods)")
    if videos:
        for v in videos:
            rel = _rel(out_dir, v)
            lines.append(f"- [{v.name}]({rel})")
    else:
        lines.append("- No video files found yet.")
    lines.append("")
    lines.append("## 6. Appendix: Configuration Snapshots")
    if cfg_dir.exists():
        cfg_files = sorted(cfg_dir.rglob("*"))
        cfg_files = [p for p in cfg_files if p.is_file()]
        if cfg_files:
            for cfg in cfg_files:
                rel = _rel(out_dir, cfg)
                lines.append(f"- [{cfg.name}]({rel})")
        else:
            lines.append("- Config snapshot folder exists but no files were found.")
    else:
        lines.append("- Config snapshots not found. Re-run report script after latest updates.")
    lines.append("")
    lines.append("## 7. Notes")
    lines.append("- Replace all TODO items before final PDF export.")
    lines.append("- Keep figure numbering aligned with your report template.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated markdown draft: {md_path}")


if __name__ == "__main__":
    main()
