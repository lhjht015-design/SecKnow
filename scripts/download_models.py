"""Import pinned offline model snapshots into ``models/``.

The application never downloads models at runtime. Run this script once while
connected, verify the generated SHA256SUMS, and then set SECKNOW_OFFLINE=true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

MODELS = {
    "bge-m3": {
        "repo": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    },
    "bge-reranker-v2-m3": {
        "repo": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    },
}
REVISION = re.compile(r"^[0-9a-f]{40}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=[*MODELS, "all"])
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face endpoint, e.g. https://hf-mirror.com",
    )
    args = parser.parse_args()
    selected = list(MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit("Install the semantic extra before importing models.") from error
    manifest_path = args.root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest.setdefault("models", [])
    for model_id in selected:
        spec = MODELS[model_id]
        revision = str(spec["revision"])
        if not REVISION.fullmatch(revision):
            raise SystemExit(f"Model {model_id} is not pinned to a 40-character revision.")
        target = args.root / model_id
        print(f"Downloading {spec['repo']} @ {revision} -> {target}")
        snapshot_download(
            repo_id=str(spec["repo"]),
            revision=revision,
            local_dir=str(target),
            ignore_patterns=["imgs/**", "*.DS_Store"],
        )
        files = _files_for_manifest(target)
        if not files:
            raise SystemExit(f"Model snapshot is empty: {target}")
        checksum_path = target / "SHA256SUMS"
        checksum_lines = [
            f"{_sha256(path)}  {path.relative_to(target).as_posix()}"
            for path in files
        ]
        _atomic_write(checksum_path, "\n".join(checksum_lines) + "\n")
        _atomic_write(target / "REVISION", revision + "\n")
        _update_manifest(
            manifest,
            model_id,
            revision,
            checksum_path.relative_to(args.root).as_posix(),
        )
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return 0


def _files_for_manifest(root: Path) -> list[Path]:
    excluded = {"SHA256SUMS", "REVISION"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name not in excluded)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _update_manifest(
    manifest: dict[str, object],
    model_id: str,
    revision: str,
    checksum: str,
) -> None:
    rows = manifest.setdefault("models", [])
    if not isinstance(rows, list):
        raise SystemExit("models/manifest.json has an invalid models list")
    for row in rows:
        if isinstance(row, dict) and row.get("id") == model_id:
            row["revision"] = revision
            row["sha256_manifest"] = checksum
            return
    rows.append({"id": model_id, "revision": revision, "sha256_manifest": checksum})


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
