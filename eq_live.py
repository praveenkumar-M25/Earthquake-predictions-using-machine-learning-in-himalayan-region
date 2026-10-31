import io, requests, pandas as pd
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def fetch_usgs(bbox, minmag, start, end=None):
    """bbox = (min_lat, max_lat, min_lon, max_lon). Returns the notebook's catalogue schema."""
    p = dict(format="csv", orderby="time-asc", minmagnitude=minmag, limit=20000, starttime=start,
             minlatitude=bbox[0], maxlatitude=bbox[1], minlongitude=bbox[2], maxlongitude=bbox[3])
    if end: p["endtime"] = end
    r = requests.get(USGS, params=p, timeout=60); r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text)).dropna(subset=["mag"])
    t = pd.to_datetime(d["time"], utc=True)
    return pd.DataFrame({"time": t, "year": t.dt.year, "month": t.dt.month, "day": t.dt.day, "hour": t.dt.hour,
                         "min": t.dt.minute, "second": t.dt.second + t.dt.microsecond / 1e6, "magnitude": d["mag"],
                         "lattitude": d["latitude"], "longitude": d["longitude"], "depth": d["depth"]})
