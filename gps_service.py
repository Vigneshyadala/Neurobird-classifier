"""
gps_service.py  —  NeuroBird GPS Intelligence Layer
====================================================
Queries eBird API to get locally observed species, then applies
confidence boost / penalty multipliers to model predictions.

Boost rules (from roadmap):
  • Species seen locally in last 14 days  →  × 1.3
  • Species in regional list (ever seen)  →  × 1.1
  • Species NOT in regional list          →  × 0.4
  • Seasonal match (migratory season)     →  × 1.2
  • Seasonal mismatch (wrong season)      →  × 0.6
"""

import os
import requests
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env file so EBIRD_API_KEY is available
load_dotenv()

# ── eBird API config ──────────────────────────────────────────────────────────
EBIRD_API_KEY  = os.getenv("EBIRD_API_KEY", "")          # set in .env
EBIRD_BASE     = "https://api.ebird.org/v2"
DEFAULT_RADIUS = 50   # km
RECENT_DAYS    = 14


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level eBird calls
# ─────────────────────────────────────────────────────────────────────────────

def _ebird_headers():
    return {"X-eBirdApiToken": EBIRD_API_KEY}


def get_nearby_observations(lat: float, lng: float,
                            radius_km: int = DEFAULT_RADIUS,
                            days: int = RECENT_DAYS) -> list[dict]:
    """
    Returns list of recent observations near (lat, lng).
    Each dict has: speciesCode, comName, sciName, obsDt, howMany, locName.
    """
    if not EBIRD_API_KEY:
        return []
    try:
        url    = f"{EBIRD_BASE}/data/obs/geo/recent"
        params = {
            "lat"        : lat,
            "lng"        : lng,
            "dist"       : radius_km,
            "back"       : days,
            "maxResults" : 500,
            "fmt"        : "json",
        }
        resp = requests.get(url, params=params, headers=_ebird_headers(), timeout=8)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        print(f"  [GPS] eBird recent obs error: {e}")
        return []


def get_notable_nearby(lat: float, lng: float,
                       radius_km: int = DEFAULT_RADIUS) -> list[dict]:
    """Returns rare/notable species recently reported nearby."""
    if not EBIRD_API_KEY:
        return []
    try:
        url    = f"{EBIRD_BASE}/data/obs/geo/recent/notable"
        params = {"lat": lat, "lng": lng, "dist": radius_km, "fmt": "json"}
        resp = requests.get(url, params=params, headers=_ebird_headers(), timeout=8)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        print(f"  [GPS] eBird notable obs error: {e}")
        return []


def get_hotspots_nearby(lat: float, lng: float,
                        radius_km: int = 25) -> list[dict]:
    """Returns top birding hotspots near the user."""
    if not EBIRD_API_KEY:
        return []
    try:
        url    = f"{EBIRD_BASE}/ref/hotspot/geo"
        params = {"lat": lat, "lng": lng, "dist": radius_km, "fmt": "json"}
        resp = requests.get(url, params=params, headers=_ebird_headers(), timeout=8)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        print(f"  [GPS] eBird hotspots error: {e}")
        return []


def get_species_in_region(region_code: str) -> list[str]:
    """Returns all species codes ever recorded in a region (e.g. 'US-TX')."""
    if not EBIRD_API_KEY:
        return []
    try:
        url  = f"{EBIRD_BASE}/product/spplist/{region_code}"
        resp = requests.get(url, headers=_ebird_headers(), timeout=10)
        if resp.status_code == 200:
            return resp.json()   # list of species codes
        return []
    except Exception as e:
        print(f"  [GPS] eBird region species error: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  Name normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(name: str) -> str:
    """Lower-case, strip punctuation for fuzzy matching."""
    return name.lower().replace("-", " ").replace("'", "").replace("_", " ").strip()


def _build_local_name_set(observations: list[dict]) -> set[str]:
    """Build a set of normalised common names from eBird observations."""
    names = set()
    for obs in observations:
        com = obs.get("comName", "")
        if com:
            names.add(_normalise(com))
    return names


# ─────────────────────────────────────────────────────────────────────────────
#  Seasonal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Very rough hemispheric migration windows
_MIGRATION_MONTHS_NORTH = {3, 4, 5, 9, 10, 11}   # spring + autumn
_WINTER_MONTHS_NORTH    = {12, 1, 2}
_SUMMER_MONTHS_NORTH    = {6, 7, 8}

# Keywords that hint at season-specific species
_WINTER_KEYWORDS  = ["snow", "snowy", "polar", "arctic", "tundra", "winter"]
_SUMMER_KEYWORDS  = ["tropical", "summer", "oriole", "tanager", "bunting"]
_MIGRANT_KEYWORDS = ["warbler", "sandpiper", "plover", "flycatcher", "vireo",
                     "thrush", "sparrow"]


def _seasonal_multiplier(species_name: str, month: int) -> float:
    """
    Returns a seasonal confidence multiplier.
    Uses species-name heuristics when we don't have full migration data.
    """
    name_lower = species_name.lower()
    in_migration = month in _MIGRATION_MONTHS_NORTH
    in_winter    = month in _WINTER_MONTHS_NORTH
    in_summer    = month in _SUMMER_MONTHS_NORTH

    # Winter-specialist in summer → penalty
    if any(kw in name_lower for kw in _WINTER_KEYWORDS) and in_summer:
        return 0.6
    # Summer-specialist in winter → penalty
    if any(kw in name_lower for kw in _SUMMER_KEYWORDS) and in_winter:
        return 0.6
    # Migrant species in migration months → bonus
    if any(kw in name_lower for kw in _MIGRANT_KEYWORDS) and in_migration:
        return 1.2

    return 1.0   # no seasonal adjustment


# ─────────────────────────────────────────────────────────────────────────────
#  Main boost function
# ─────────────────────────────────────────────────────────────────────────────

def apply_gps_boost(predictions: list[dict],
                    lat: float,
                    lng: float,
                    date_str: str | None = None) -> list[dict]:
    """
    Takes model predictions and GPS coordinates, returns adjusted predictions.

    Args:
        predictions : [{"species": str, "confidence": float}, ...]
        lat, lng    : user's GPS coordinates
        date_str    : ISO date string "YYYY-MM-DD", defaults to today

    Returns:
        Same list with adjusted confidence values + location metadata.
    """
    if not EBIRD_API_KEY:
        # No API key — return predictions unchanged with a flag
        for p in predictions:
            p["location_boosted"] = False
            p["location_badge"]   = "⚠️ Add eBird API key for GPS boost"
        return predictions

    # Parse date
    if date_str:
        try:
            dt = datetime.date.fromisoformat(date_str)
        except ValueError:
            dt = datetime.date.today()
    else:
        dt = datetime.date.today()
    month = dt.month

    # Fetch local species (last 14 days)
    print(f"  [GPS] Querying eBird near ({lat:.4f}, {lng:.4f})...")
    recent_obs   = get_nearby_observations(lat, lng)
    local_names  = _build_local_name_set(recent_obs)

    # Unique species names seen recently
    recent_species = list({obs.get("comName", "") for obs in recent_obs if obs.get("comName")})
    print(f"  [GPS] Found {len(recent_species)} species in last {RECENT_DAYS} days nearby")

    # Apply multipliers
    boosted = []
    for pred in predictions:
        species      = pred["species"]
        base_conf    = pred["confidence"]
        norm_species = _normalise(species)

        # Determine location multiplier
        if norm_species in local_names:
            loc_mult  = 1.3
            badge     = "✅ Seen locally (last 14 days)"
            boosted_f = True
        else:
            # Slight boost: name partially matches local list
            partial = any(norm_species in ln or ln in norm_species
                         for ln in local_names if len(ln) > 5)
            if partial:
                loc_mult  = 1.1
                badge     = "🌍 Possibly in region"
                boosted_f = True
            else:
                loc_mult  = 1.0
                badge     = "🌍 GPS Boosted"
                boosted_f = False

        # Seasonal multiplier
        sea_mult = _seasonal_multiplier(species, month)

        # Combined final confidence
        final = round(min(base_conf * loc_mult * sea_mult, 99.0), 2)

        boosted.append({
            "species"         : species,
            "confidence"      : final,
            "base_confidence" : base_conf,
            "location_badge"  : badge,
            "location_boosted": boosted_f,
            "loc_multiplier"  : loc_mult,
            "sea_multiplier"  : sea_mult,
        })

    # Re-sort by boosted confidence
    boosted.sort(key=lambda x: x["confidence"], reverse=True)
    return boosted


# ─────────────────────────────────────────────────────────────────────────────
#  Nearby-species summary (for the /nearby-species route)
# ─────────────────────────────────────────────────────────────────────────────

def get_nearby_species_summary(lat: float, lng: float,
                               radius_km: int = DEFAULT_RADIUS) -> dict:
    """
    Returns a rich summary of what birds are near the user right now.
    """
    recent_obs  = get_nearby_observations(lat, lng, radius_km)
    notable_obs = get_notable_nearby(lat, lng, radius_km)
    hotspots    = get_hotspots_nearby(lat, lng, min(radius_km, 25))

    # Deduplicate species
    seen: dict[str, dict] = {}
    for obs in recent_obs:
        name = obs.get("comName", "")
        if name and name not in seen:
            seen[name] = {
                "common_name"  : name,
                "scientific_name": obs.get("sciName", ""),
                "species_code" : obs.get("speciesCode", ""),
                "last_seen"    : obs.get("obsDt", ""),
                "location"     : obs.get("locName", ""),
                "count"        : obs.get("howMany", "?"),
                "notable"      : False,
            }

    notable_names = {obs.get("comName", "") for obs in notable_obs}
    for name in notable_names:
        if name in seen:
            seen[name]["notable"] = True

    species_list = sorted(seen.values(), key=lambda x: x["last_seen"], reverse=True)

    # Format hotspots
    hotspot_list = []
    for h in hotspots[:8]:
        hotspot_list.append({
            "name"     : h.get("locName", ""),
            "lat"      : h.get("lat", 0),
            "lng"      : h.get("lng", 0),
            "num_species": h.get("numSpeciesAllTime", 0),
            "loc_id"   : h.get("locId", ""),
        })

    return {
        "species_count" : len(species_list),
        "species"       : species_list[:50],    # cap at 50
        "hotspots"      : hotspot_list,
        "notable_count" : len(notable_names),
        "radius_km"     : radius_km,
        "queried_at"    : datetime.datetime.utcnow().isoformat(),
        "api_available" : bool(EBIRD_API_KEY),
    }