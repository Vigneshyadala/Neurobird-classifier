"""
database.py  —  NeuroBird Personal Sighting Logbook
====================================================
SQLite schema and helper functions for saving and retrieving
personal bird sighting records with GPS data.

Tables:
  • sightings  — core identification records
  • lifelist   — one row per unique species ever identified
"""

import sqlite3
import datetime
import os
from pathlib import Path

# ── DB path ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "neurobird_sightings.db")


# ─────────────────────────────────────────────────────────────────────────────
#  Schema creation
# ─────────────────────────────────────────────────────────────────────────────

CREATE_SIGHTINGS = """
CREATE TABLE IF NOT EXISTS sightings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    species         TEXT    NOT NULL,
    scientific_name TEXT,
    confidence      REAL,
    source          TEXT    DEFAULT 'image',   -- 'image' | 'audio' | 'combined'
    lat             REAL,
    lng             REAL,
    location_name   TEXT,
    image_path      TEXT,
    notes           TEXT,
    location_badge  TEXT,
    date_identified TEXT    NOT NULL,          -- ISO date YYYY-MM-DD
    created_at      TEXT    NOT NULL           -- ISO datetime
);
"""

CREATE_LIFELIST = """
CREATE TABLE IF NOT EXISTS lifelist (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    species         TEXT    NOT NULL UNIQUE,
    scientific_name TEXT,
    first_seen_date TEXT,
    first_seen_lat  REAL,
    first_seen_lng  REAL,
    total_sightings INTEGER DEFAULT 1,
    created_at      TEXT    NOT NULL
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sightings_species ON sightings(species);",
    "CREATE INDEX IF NOT EXISTS idx_sightings_date    ON sightings(date_identified);",
    "CREATE INDEX IF NOT EXISTS idx_sightings_loc     ON sightings(lat, lng);",
]


def init_db():
    """Initialise database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(CREATE_SIGHTINGS)
    cur.execute(CREATE_LIFELIST)
    for idx in CREATE_INDEXES:
        cur.execute(idx)
    conn.commit()
    conn.close()
    print(f"  ✔  Database ready  →  {DB_PATH}")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # dict-like access
    return conn


# ─────────────────────────────────────────────────────────────────────────────
#  Save a sighting
# ─────────────────────────────────────────────────────────────────────────────

def save_sighting(species: str,
                  scientific_name: str = "",
                  confidence: float = 0.0,
                  source: str = "image",
                  lat: float | None = None,
                  lng: float | None = None,
                  location_name: str = "",
                  image_path: str = "",
                  notes: str = "",
                  location_badge: str = "",
                  date_identified: str | None = None) -> int:
    """
    Save a new sighting record and update the lifelist.
    Returns the new sighting id.
    """
    now   = datetime.datetime.utcnow().isoformat()
    today = date_identified or datetime.date.today().isoformat()

    conn = _get_conn()
    cur  = conn.cursor()

    cur.execute("""
        INSERT INTO sightings
            (species, scientific_name, confidence, source,
             lat, lng, location_name, image_path, notes,
             location_badge, date_identified, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (species, scientific_name, confidence, source,
          lat, lng, location_name, image_path, notes,
          location_badge, today, now))

    sighting_id = cur.lastrowid

    # Upsert lifelist
    cur.execute("SELECT id, total_sightings FROM lifelist WHERE species = ?", (species,))
    row = cur.fetchone()
    if row:
        cur.execute("""
            UPDATE lifelist SET total_sightings = total_sightings + 1
            WHERE species = ?
        """, (species,))
    else:
        cur.execute("""
            INSERT INTO lifelist
                (species, scientific_name, first_seen_date,
                 first_seen_lat, first_seen_lng, total_sightings, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
        """, (species, scientific_name, today, lat, lng, now))

    conn.commit()
    conn.close()
    return sighting_id


# ─────────────────────────────────────────────────────────────────────────────
#  Retrieve sightings
# ─────────────────────────────────────────────────────────────────────────────

def get_sightings(limit: int = 50,
                  species: str | None = None,
                  date_from: str | None = None,
                  date_to: str | None = None) -> list[dict]:
    """
    Retrieve sighting history with optional filters.
    Returns list of dicts ordered newest first.
    """
    conn   = _get_conn()
    cur    = conn.cursor()
    where  = []
    params = []

    if species:
        where.append("species LIKE ?")
        params.append(f"%{species}%")
    if date_from:
        where.append("date_identified >= ?")
        params.append(date_from)
    if date_to:
        where.append("date_identified <= ?")
        params.append(date_to)

    sql = "SELECT * FROM sightings"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY created_at DESC LIMIT {int(limit)}"

    cur.execute(sql, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_lifelist() -> list[dict]:
    """Return all unique species ever identified, sorted by first seen date."""
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM lifelist ORDER BY first_seen_date DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_sighting_stats() -> dict:
    """Return summary statistics for the logbook dashboard."""
    conn = _get_conn()
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM sightings")
    total_sightings = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM lifelist")
    unique_species = cur.fetchone()[0]

    cur.execute("""
        SELECT species, COUNT(*) as cnt
        FROM sightings
        GROUP BY species
        ORDER BY cnt DESC
        LIMIT 5
    """)
    top_species = [{"species": r[0], "count": r[1]} for r in cur.fetchall()]

    cur.execute("""
        SELECT date_identified, COUNT(*) as cnt
        FROM sightings
        GROUP BY date_identified
        ORDER BY date_identified DESC
        LIMIT 7
    """)
    recent_days = [{"date": r[0], "count": r[1]} for r in cur.fetchall()]

    cur.execute("""
        SELECT source, COUNT(*) as cnt
        FROM sightings
        GROUP BY source
    """)
    by_source = {r[0]: r[1] for r in cur.fetchall()}

    conn.close()
    return {
        "total_sightings": total_sightings,
        "unique_species" : unique_species,
        "top_species"    : top_species,
        "recent_days"    : recent_days,
        "by_source"      : by_source,
    }


def delete_sighting(sighting_id: int) -> bool:
    """Delete a sighting by id. Returns True if deleted."""
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("DELETE FROM sightings WHERE id = ?", (sighting_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    conn.close()
    return deleted
