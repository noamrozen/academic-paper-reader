"""Bring the database to the current schema: `uv run python -m app.migrate`.

The server does this by itself when it starts on an older database (db.init), so running it is
optional; running it before a deploy keeps the first requests after the deploy from waiting on it.
For Turso, set USE_TURSO=1 with TURSO_DATABASE_URL and TURSO_AUTH_TOKEN.
"""

import logging

from . import db

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    before = db.schema_version()
    db.init()
    print(f"Schema version {before} -> {db.schema_version()}")
