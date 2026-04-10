#!/usr/bin/env python3
"""
Pre-download the all-MiniLM-L6-v2 embedding model for offline/airgapped use.

By default, fastembed downloads the model on first use and caches it at:
  Linux/macOS:  ~/.cache/huggingface/hub/
  Windows:      %USERPROFILE%\.cache\huggingface\hub\

This script downloads it explicitly so you can transfer the cache to
a machine without internet access.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --cache-dir /opt/models

Then on the offline machine:
    export HF_HOME=/opt/models
    # Now Memoire will find the model without network access.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Pre-download Memoire embedding model.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model ID to download",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify the model is cached, don't download",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        print(f"Using cache dir: {args.cache_dir}")

    # Determine where the cache actually is
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = hf_home / "hub"
    model_slug = args.model.replace("/", "--")
    expected = hub_dir / f"models--{model_slug}"

    if args.verify:
        if expected.exists():
            print(f"✓ Model is cached at: {expected}")
            sys.exit(0)
        else:
            print(f"✗ Model NOT found at: {expected}")
            print(f"  Run without --verify to download it.")
            sys.exit(1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Installing...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import snapshot_download

    print(f"Downloading {args.model}...")
    print(f"Cache directory: {hf_home}")
    print()

    path = snapshot_download(
        repo_id=args.model,
        cache_dir=str(hub_dir),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    size_bytes = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    size_mb = size_bytes / (1024 * 1024)

    print()
    print(f"✓ Downloaded to: {path}")
    print(f"  Total size: {size_mb:.1f} MB")
    print()
    print("To use on an offline machine:")
    print(f"  1. Copy {hf_home} to the offline machine")
    print(f"  2. Set:  export HF_HOME={hf_home}")
    print(f"  3. Run Memoire — it will use the cached model.")


if __name__ == "__main__":
    main()
