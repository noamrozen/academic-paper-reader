"""A Turso (hosted libSQL) database behind the parts of the sqlite3 interface db.py uses.

The libsql package follows sqlite3 except that its rows are plain tuples, its cursors can't be
iterated, it takes only positional parameters, and it cuts strings at a NUL character. Rows here
are like sqlite3.Row (by index or column name, and dict(row)), named parameters (:name) are turned
into positional ones, and NULs are dropped from the strings written.
"""

import re

import libsql

_NAMED = re.compile(r"(?<!:):([A-Za-z_]\w*)")


class Row(tuple):
    def __new__(cls, values, index: dict[str, int]):
        row = super().__new__(cls, values)
        row._index = index
        return row

    def __getitem__(self, key):
        return super().__getitem__(self._index[key] if isinstance(key, str) else key)

    def keys(self):
        return list(self._index)


class Cursor:
    def __init__(self, cursor):
        self._cursor = cursor
        names = [d[0] for d in cursor.description or ()]
        self._index = {n: i for i, n in enumerate(names)}

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    def fetchone(self):
        values = self._cursor.fetchone()
        return None if values is None else Row(values, self._index)

    def fetchall(self):
        return [Row(values, self._index) for values in self._cursor.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


def _clean(value):
    # libsql cuts a string at its first NUL character (PDF text extraction can leave some).
    return value.replace("\x00", "") if isinstance(value, str) else value


def _positional(sql: str, params):
    if isinstance(params, dict):
        sql, params = _NAMED.sub("?", sql), [params[name] for name in _NAMED.findall(sql)]
    return sql, tuple(_clean(p) for p in params)


class Connection:
    def __init__(self, url: str, auth_token: str | None):
        self._conn = libsql.connect(database=url, auth_token=auth_token or "")

    def execute(self, sql: str, params=()):
        return Cursor(self._conn.execute(*_positional(sql, params)))

    def executemany(self, sql: str, seq):
        rows = [_positional(sql, p)[1] for p in seq]
        return self._conn.executemany(_NAMED.sub("?", sql), rows) if rows else None

    def executescript(self, script: str):
        self._conn.executescript(script)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()
