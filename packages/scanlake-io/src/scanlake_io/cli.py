"""`scanlake-data`: check Parquet files and profiles against the upload format.

    scanlake-data verify-parquet FILE --kind scans     # would the Store accept this file?
    scanlake-data coverage PROFILE.json                # can this profile fill every column?

`--format` takes a URL or a local format.json (default: the Gateway's /api/format).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scanlake_core.format import Source, coverage, parse_format, validate_columns
from scanlake_core.profile.load import load_hud_profile
from scanlake_core.schema_map import slot_layout
from scanlake_io.format_source import fetch_format
from scanlake_io.parquet import column_types

DEFAULT_GATEWAY = "https://gateway.scanlake.rocks"


def default_format_url() -> str:
    explicit = os.environ.get("SCANLAKE_FORMAT_URL")
    if explicit:
        return explicit
    return os.environ.get("SCANLAKE_GATEWAY_URL", DEFAULT_GATEWAY).rstrip("/") + "/api/format"


def _verify(args: argparse.Namespace) -> int:
    fmt = parse_format(fetch_format(args.format))
    spec = fmt.file(args.kind)
    if spec is None:
        print(f"format v{fmt.version} has no {args.kind!r} file; it is not checked")
        return 0
    types = column_types(Path(args.file).read_bytes())
    violations = validate_columns(types, spec)
    for line in violations:
        print(f"  {line}")
    print(f"{args.kind}: {'does not match' if violations else 'matches'} format v{fmt.version}")
    return 1 if violations else 0


def _coverage(args: argparse.Namespace) -> int:
    fmt = parse_format(fetch_format(args.format))
    cov = coverage(fmt, slot_layout(load_hud_profile(Path(args.profile))))
    for b in cov.bindings:
        detail = ", ".join(b.rois) if b.rois else ""
        flag = (
            ""
            if b.source is not Source.UNBOUND
            else ("  MISSING" if b.required else "  (left out)")
        )
        print(f"  {b.kind:13} {b.column:18} {b.source.value:8} {detail}{flag}")
    if cov.complete:
        print(f"complete: this profile can fill format v{fmt.version}")
        return 0
    print(f"incomplete: nothing fills {cov.describe_missing()}")
    return 1


def _manifest(args: argparse.Namespace) -> int:
    from scanlake_io.manifest import build_manifest, upload_commands

    source = Path(args.source)
    manifest = build_manifest(source, args.version, args.min_app)
    out = Path(args.out) if args.out else source / "manifest.json"
    out.write_text(manifest.to_json(), encoding="utf-8")
    print(f"wrote {out}: {len(manifest.files)} files, version {manifest.version}")
    print("Upload (files first, manifest last):")
    for command in upload_commands(manifest, source):
        print(" ", command)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scanlake-data",
        description="Check Parquet files and profiles against the upload format.",
    )
    parser.add_argument("--format", default=default_format_url(), help="format URL or file")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-parquet", help="check a Parquet file against the format")
    verify.add_argument("file")
    verify.add_argument("--kind", required=True, choices=["scans", "compositions", "confidences"])
    verify.set_defaults(run=_verify)
    cov = sub.add_parser("coverage", help="which source fills each format column")
    cov.add_argument("profile")
    cov.set_defaults(run=_coverage)
    man = sub.add_parser("manifest", help="build the desktop asset manifest of a folder")
    man.add_argument("source", help="folder with profiles/, anchors/, models/")
    man.add_argument("--version", required=True)
    man.add_argument("--min-app", default="0.1.0")
    man.add_argument("--out", help="default: <source>/manifest.json")
    man.set_defaults(run=_manifest)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
