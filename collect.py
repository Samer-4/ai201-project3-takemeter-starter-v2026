#!/usr/bin/env python3
"""
Collect public fitness posts for TakeMeter. ← UNIT 5, MILESTONE 4

Reddit was the first choice, but its anonymous API answers 403 and its
robots.txt disallows everything, so this reads the public Lemmy API instead —
Reddit-style fitness communities with an open, documented, keyless API.

    python collect.py                      # fetch raw, then process
    python collect.py --process-only       # rebuild the processed CSV from raw

What it writes:

    data/raw/<community>.json      every post exactly as the API returned it
    data/posts_processed.csv       one row per usable post, deduplicated

Politeness: one request per host every `--delay` seconds. The default is 60,
which is the Crawl-delay both instances publish in their robots.txt. A full
run is about twenty requests, so it takes twenty minutes or so. Nothing here
logs in, and nothing touches a path robots.txt disallows.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass


# Each community is read from the instance that hosts it, so the listing is
# that community's own and not some other server's partial federated copy.
COMMUNITIES = [
    ("lemmy.world", "fitness@lemmy.world"),
    ("lemmy.ml", "fitness@lemmy.ml"),
    ("lemmy.world", "running@lemmy.world"),
]

USER_AGENT = "TakeMeter-coursework/0.1 (AI201 student classifier project; non-commercial)"
PAGE_SIZE = 50

# A post shorter than this many words is a title on a link or an image, and
# there's nothing in it for a person or a model to classify.
MIN_WORDS = 8

RAW_DIR = Path("data/raw")
PROCESSED = Path("data/posts_processed.csv")


def fetch_json(url, retries=3):
    """GET a URL and parse it. Backs off on 429 and 5xx; gives up on 4xx."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                wait = 60 * attempt
                print(f"  [{exc.code}] backing off {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as exc:
            print(f"  [network] {exc.reason}; retrying in 30s", flush=True)
            time.sleep(30)
    raise RuntimeError(f"Gave up on {url} after {retries} attempts")


def fetch_community(host, community, delay, max_pages=40):
    """Every post the community's listing returns, oldest page last."""
    posts, page, cursor = [], 1, None
    while page <= max_pages:
        params = {"community_name": community, "sort": "New",
                  "limit": PAGE_SIZE, "type_": "All"}
        # Lemmy 0.19 pages with an opaque cursor; older servers use page numbers.
        if cursor:
            params["page_cursor"] = cursor
        else:
            params["page"] = page
        url = f"https://{host}/api/v3/post/list?{urllib.parse.urlencode(params)}"
        data = fetch_json(url)
        batch = data.get("posts", [])
        print(f"  {community} page {page}: {len(batch)} posts", flush=True)
        if not batch:
            break
        posts.extend(batch)
        cursor = data.get("next_page")
        if len(batch) < PAGE_SIZE and not cursor:
            break
        page += 1
        time.sleep(delay)
    return posts


def clean_text(text):
    """Collapse runs of blank lines and trailing spaces. Words are untouched."""
    text = re.sub(r"[ \t]+\n", "\n", str(text or ""))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def post_text(post):
    """Title plus body — on Lemmy, as on Reddit, the title is often the question."""
    title = clean_text(post.get("name"))
    body = clean_text(post.get("body"))
    if body and title and not body.startswith(title):
        return f"{title}\n\n{body}"
    return body or title


def dedupe_key(text):
    """Match on the words, not on punctuation, case or spacing."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def process(raw_files):
    """Raw API dumps → one usable, deduplicated row per post. Returns (rows, dropped)."""
    rows, seen_ids, seen_text = [], set(), set()
    dropped = {"removed_or_deleted": 0, "nsfw": 0, "too_short": 0,
               "duplicate_id": 0, "duplicate_text": 0}

    for path in raw_files:
        dump = json.loads(Path(path).read_text(encoding="utf-8"))
        for item in dump["posts"]:
            post = item["post"]
            if post.get("removed") or post.get("deleted"):
                dropped["removed_or_deleted"] += 1
                continue
            if post.get("nsfw"):
                dropped["nsfw"] += 1
                continue
            # The same post can surface on two instances; ap_id is its one
            # canonical address across the whole federation.
            canonical = post.get("ap_id")
            if canonical in seen_ids:
                dropped["duplicate_id"] += 1
                continue
            text = post_text(post)
            if len(text.split()) < MIN_WORDS:
                dropped["too_short"] += 1
                continue
            key = dedupe_key(text)
            if key in seen_text:
                dropped["duplicate_text"] += 1
                continue
            seen_ids.add(canonical)
            seen_text.add(key)
            rows.append({
                "id": f"{dump['host']}:{post['id']}",
                "source": dump["community"],
                "url": canonical,
                "local_url": f"https://{dump['host']}/post/{post['id']}",
                "published": post.get("published"),
                "title": clean_text(post.get("name")),
                "has_link": bool(post.get("url")),
                "text": text,
            })
    return rows, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delay", type=float, default=60,
                        help="seconds between requests to one host (robots.txt asks for 60)")
    parser.add_argument("--process-only", action="store_true",
                        help="skip fetching; rebuild the processed CSV from data/raw")
    args = parser.parse_args()

    import pandas as pd

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = []
    last_hit = {}

    for host, community in COMMUNITIES:
        out = RAW_DIR / f"{community.replace('@', '_at_')}.json"
        raw_files.append(out)
        if args.process_only:
            if not out.exists():
                print(f"[FIX ME] {out} missing — run without --process-only first.")
                return 1
            continue
        since = time.time() - last_hit.get(host, 0)
        if since < args.delay:
            time.sleep(args.delay - since)
        print(f"Fetching {community} from {host}", flush=True)
        posts = fetch_community(host, community, args.delay)
        last_hit[host] = time.time()
        out.write_text(json.dumps({
            "host": host,
            "community": community,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": f"https://{host}/api/v3/post/list",
            "posts": posts,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {out} ({len(posts)} posts)", flush=True)

    rows, dropped = process(raw_files)
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED, index=False)

    print(f"\n{len(df)} usable posts → {PROCESSED}")
    print(df["source"].value_counts().to_string())
    print("\nDropped:")
    for reason, count in dropped.items():
        print(f"  {reason:<20} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
