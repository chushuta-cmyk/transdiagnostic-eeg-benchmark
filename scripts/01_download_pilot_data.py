#!/usr/bin/env python3
"""
Step 01: Universal Multi-Threaded S3 Downloader for OpenNeuro Datasets
Compatible with OpenNeuro ds003505 (Cognitive PVT EEG), ds003478, and ds004902.
Features:
- Pure Python (Zero external aws-cli or boto3 dependency).
- Resumable & safe: Checks existing file sizes, never re-downloads intact files.
- Pilot mode: Allows downloading a single subject (e.g., sub-01) in seconds for local testing.
"""

import argparse
import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

S3_BASE_URL = "https://s3.amazonaws.com/openneuro.org"
SSL_CTX = ssl._create_unverified_context()


def list_s3_files(dataset_id: str, prefix_filter: str = "") -> list:
    """Lists files under an OpenNeuro S3 dataset bucket."""
    results = []
    continuation_token = None
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    query_prefix = f"{dataset_id}/{prefix_filter}".rstrip('/')

    print(f"📡 Querying OpenNeuro S3 bucket for {dataset_id}...")
    while True:
        url = f"{S3_BASE_URL}?list-type=2&prefix={query_prefix}"
        if continuation_token:
            url += f"&continuation-token={urllib.parse.quote(continuation_token)}"

        with urllib.request.urlopen(url, timeout=20, context=SSL_CTX) as resp:
            xml_data = resp.read()

        root = ET.fromstring(xml_data)
        for content in root.findall('s3:Contents', ns):
            key = content.find('s3:Key', ns).text
            size = int(content.find('s3:Size', ns).text)
            results.append((key, size))

        is_truncated = root.find('s3:IsTruncated', ns)
        if is_truncated is not None and is_truncated.text == 'true':
            continuation_token = root.find('s3:NextContinuationToken', ns).text
        else:
            break

    return results


def download_file(key: str, size: int, dataset_id: str, dest_dir: Path) -> str:
    rel_path = key[len(dataset_id) + 1:]
    target_path = dest_dir / rel_path

    if target_path.exists() and target_path.stat().st_size == size:
        return f"SKIP: {rel_path} (already complete)"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3_BASE_URL}/{key}"
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    with urllib.request.urlopen(url, timeout=30, context=SSL_CTX) as resp:
        with open(tmp_path, 'wb') as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)

    tmp_path.replace(target_path)
    return f"DONE: {rel_path} ({size / (1024*1024):.2f} MB)"


def main():
    parser = argparse.ArgumentParser(description="Download pilot data from OpenNeuro.")
    parser.add_argument("--dataset", type=str, default="ds003505", help="OpenNeuro dataset ID")
    parser.add_argument("--pilot-sub", type=str, default="sub-01", help="Subject to download for pilot (e.g. sub-01)")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "raw")
    args = parser.parse_args()

    files = list_s3_files(args.dataset, prefix_filter=args.pilot_sub)
    print(f"Found {len(files)} files for {args.pilot_sub} in {args.dataset}.")

    total_mb = sum(sz for _, sz in files) / (1024 * 1024)
    print(f"Total download size: {total_mb:.2f} MB")

    if args.dry_run:
        for k, sz in files[:10]:
            print(f"  - {k} ({sz / 1024:.1f} KB)")
        print("Dry run complete.")
        return

    dest_dataset = args.dest / args.dataset
    dest_dataset.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_file, k, sz, args.dataset, dest_dataset) for k, sz in files]
        for fut in as_completed(futures):
            print(f"  [+] {fut.result()}")

    print("✓ Pilot download finished successfully!")


if __name__ == '__main__':
    main()
