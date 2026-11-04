"""Copies the local database (data/app.db) into the Turso database named by TURSO_DATABASE_URL and
TURSO_AUTH_TOKEN (from .env or the environment; the app itself uses them only in production, see
USE_TURSO in config.py). A one-off, for moving to Vercel: see "Deploying to Vercel" in the README.

It works on a copy, so data/app.db is left as it was. The copy is
- brought to the app's current schema (db.init);
- cleaned of NUL characters in paper text (libsql cuts strings at them);
- disowned: accounts aren't copied (their ids, passwords and sessions belong to each database), so
  the library goes up without an owner, and the first admin to sign in to the app on Turso gets it;
- pinned: papers added before the app recorded their PDF get its URL (for arXiv, with the version)
  and the SHA-256 of the file saved in data/papers. An arXiv version that came out before the paper
  was added is the one it was read from; any other is downloaded to check it's that file.
Then its rows go to Turso in one transaction, and the two databases' row counts are compared.

    uv run python scripts/migrate_to_turso.py [--replace]

--replace first deletes what the Turso database holds; without it, one that isn't empty is left
alone.
"""

import argparse
import asyncio
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db, papers  # noqa: E402
from app.turso import Connection  # noqa: E402

TABLES = ("folders", "papers", "conversations", "messages", "usage")  # parents before children
OWNED = ("folders", "papers", "conversations", "usage")  # the tables with a user_id


def init_schema(path=None, url=None, token=None):
    db.DB_PATH, db.TURSO_DATABASE_URL, db.TURSO_AUTH_TOKEN = path, url, token
    db.init()


def drop_nuls(copy: sqlite3.Connection):
    # In Python: SQLite's replace() can't see a NUL (char(0) is an empty string to it).
    rows = copy.execute("SELECT id, text FROM papers").fetchall()
    fixed = [(t.replace("\x00", ""), i) for i, t in rows if "\x00" in t]
    copy.executemany("UPDATE papers SET text = ? WHERE id = ?", fixed)
    copy.commit()
    print(f"Dropped NUL characters from the text of {len(fixed)} papers")


def disown(copy: sqlite3.Connection):
    owners = {r[0] for t in OWNED for r in copy.execute(f"SELECT DISTINCT user_id FROM {t}")} - {""}
    if len(owners) > 1:
        sys.exit(f"The local database holds the libraries of {len(owners)} accounts; this copies one person's.")
    for table in OWNED:
        copy.execute(f"UPDATE {table} SET user_id = ''")
    copy.commit()
    print("The library goes up without an owner: the first admin to sign in gets it")


async def pin(copy: sqlite3.Connection, pdf_dir: Path):
    """An arXiv paper's latest version is the one it was read from if it came out before the day the
    paper was added. Otherwise (and for a non-arXiv PDF) the saved file must match a fresh download."""
    rows = copy.execute("SELECT id, source_url, created_at FROM papers WHERE pdf_url IS NULL").fetchall()
    if not rows:
        return
    print(f"Pinning {len(rows)} papers to the PDF they were read from")
    arxiv_ids = [i for i, _, _ in rows if not i.startswith("url-")]
    latest = {}
    for n in range(0, len(arxiv_ids), 20):
        for m in await papers.arxiv_query({"id_list": ",".join(arxiv_ids[n:n + 20]), "max_results": 20}):
            latest[m["arxiv_id"]] = m
    for paper_id, source_url, created_at in rows:
        saved = pdf_dir / (paper_id.replace("/", "_") + ".pdf")
        if not saved.exists():
            print(f"  {paper_id}: no saved PDF, left unpinned")
            continue
        sha = hashlib.sha256(saved.read_bytes()).hexdigest()
        meta = latest.get(paper_id)
        if paper_id.startswith("url-"):
            url, version, known = source_url, None, False
        elif meta and meta["version"]:
            url, version = f"https://arxiv.org/pdf/{paper_id}v{meta['version']}", meta["version"]
            known = meta["updated"] < datetime.fromtimestamp(created_at, timezone.utc).date().isoformat()
        else:
            print(f"  {paper_id}: not found on arXiv, left unpinned")
            continue
        if not known:
            try:
                fresh = hashlib.sha256(await asyncio.wait_for(papers.download_pdf(url), 90)).hexdigest()
            except Exception as e:
                print(f"  {paper_id}: could not download {url} ({e or type(e).__name__}), left unpinned")
                continue
            if fresh != sha:
                print(f"  {paper_id}: {url} isn't the saved PDF, left unpinned")
                continue
        copy.execute("UPDATE papers SET pdf_url = ?, pdf_sha256 = ?, arxiv_version = ?, latest_version = ?"
                     " WHERE id = ?", (url, sha, version, version, paper_id))
        print(f"  {paper_id}: {url.rsplit('/', 1)[-1]}" + ("" if known else " (checked by download)"))
    copy.commit()


def counts(conn) -> dict:
    out = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}
    out["paper text (characters)"] = conn.execute("SELECT COALESCE(sum(length(text)), 0) FROM papers").fetchone()[0]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replace", action="store_true", help="delete what the Turso database holds first")
    args = parser.parse_args()
    url, token = os.environ.get("TURSO_DATABASE_URL"), os.environ.get("TURSO_AUTH_TOKEN")
    if not url:
        sys.exit("Set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN (in .env) to the Turso database first.")
    if not config.DB_PATH.exists():
        sys.exit(f"No local database at {config.DB_PATH}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "app.db"
        with sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True) as src, sqlite3.connect(path) as dst:
            src.backup(dst)  # a consistent copy, even while the app runs
        init_schema(path=path)
        copy = sqlite3.connect(path)
        drop_nuls(copy)
        disown(copy)
        asyncio.run(pin(copy, config.PAPERS_DIR))

        print(f"Connecting to {url}")
        init_schema(url=url, token=token)  # creates the tables if the database is new
        remote = Connection(url, token)
        held = counts(remote)
        if held["papers"] or held["conversations"]:
            if not args.replace:
                sys.exit(f"The Turso database isn't empty ({held['papers']} papers, {held['conversations']}"
                         " conversations). Run again with --replace to overwrite it.")
            print("Deleting what the Turso database holds (--replace)")
        for table in reversed(TABLES):
            remote.execute(f"DELETE FROM {table}")
        for table in TABLES:
            cols = [r[1] for r in copy.execute(f"PRAGMA table_info({table})")]
            rows = copy.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
            print(f"  {table}: {len(rows)} rows")
            remote.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", rows)
        remote.commit()

        local, uploaded = counts(copy), counts(Connection(url, token))
        for key in local:
            print(f"  {key}: local {local[key]}, Turso {uploaded[key]}")
        if local != uploaded:
            sys.exit("The counts differ: the copy is incomplete. Run again with --replace.")
        print("Done: the Turso database matches the local one.")


if __name__ == "__main__":
    main()
