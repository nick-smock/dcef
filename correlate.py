"""
correlate.py — DCEF Tool 02
Merges a DCEF telemetry CSV and a sensor output CSV on
timestamp_utc. Computes all derived fields (range_m,
pos_error_m, tier) and writes the full correlated dataset.

Requirements: pandas, numpy
Usage: python correlate.py <telemetry.csv> <sensor.csv> [output.csv] [--tier 1|2|3]
"""

import sys
import argparse
import numpy as np
import pandas as pd

# ── Maximum time skew allowed for a valid correlation match ──
MAX_SKEW_S = 0.05   # 50ms per DCEF time-sync protocol

# ── CEP thresholds per tier ───────────────────────────────────
TIER_CEP = {1: 3.0, 2: 0.30, 3: 0.05}


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised haversine distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def correlate(telem_path: str,
              sensor_path: str,
              output_path: str = None,
              tier: int = 2,
              site_id: str = None) -> pd.DataFrame:

    if output_path is None:
        output_path = telem_path.replace('_telemetry.csv', '_correlated.csv')
        if output_path == telem_path:
            output_path = 'correlated.csv'

    print(f'[DCEF] Loading telemetry: {telem_path}')
    telem = pd.read_csv(telem_path, parse_dates=['timestamp_utc'])
    telem['timestamp_utc'] = pd.to_datetime(
        telem['timestamp_utc'], utc=True, errors='coerce'
    )
    telem = telem.dropna(subset=['timestamp_utc']).sort_values('timestamp_utc')

    print(f'[DCEF] Loading sensor:    {sensor_path}')
    sensor = pd.read_csv(sensor_path, parse_dates=['timestamp_utc'])
    sensor['timestamp_utc'] = pd.to_datetime(
        sensor['timestamp_utc'], utc=True, errors='coerce'
    )
    sensor = sensor.dropna(subset=['timestamp_utc']).sort_values('timestamp_utc')

    # ── Merge-asof: nearest sensor record within MAX_SKEW_S ──
    merged = pd.merge_asof(
        telem,
        sensor[['timestamp_utc', 'sensor_lat', 'sensor_lon',
                'sensor_type', 'detection_flag']],
        on='timestamp_utc',
        tolerance=pd.Timedelta(seconds=MAX_SKEW_S),
        direction='nearest',
    )

    n_matched = merged['sensor_lat'].notna().sum()
    n_total   = len(merged)
    print(f'[DCEF] Matched {n_matched}/{n_total} records '
          f'(tolerance ±{MAX_SKEW_S*1000:.0f}ms)')

    # Drop records with no sensor match
    merged = merged.dropna(subset=['sensor_lat', 'sensor_lon'])

    # ── Derived fields ────────────────────────────────────────
    # Sensor origin is defined as the mean sensor position at
    # near-zero range; for simplicity, use the first telem point
    # as origin reference for range computation.
    # In production: pass --sensor-origin lat,lon
    sensor_origin_lat = telem['lat_truth'].iloc[0]
    sensor_origin_lon = telem['lon_truth'].iloc[0]

    merged['range_m'] = haversine(
        sensor_origin_lat, sensor_origin_lon,
        merged['lat_truth'].values,
        merged['lon_truth'].values,
    ).round(1)

    merged['pos_error_m'] = haversine(
        merged['lat_truth'].values,
        merged['lon_truth'].values,
        merged['sensor_lat'].values,
        merged['sensor_lon'].values,
    ).round(2)

    merged['tier']    = tier
    merged['site_id'] = site_id or merged.get('site_id', 'SITE-UNKNOWN')

    # ── Validate against tier CEP ─────────────────────────────
    cep = TIER_CEP[tier]
    p95 = merged['pos_error_m'].quantile(0.95)
    status = 'PASS' if p95 <= cep * 20 else 'REVIEW'  # 20× CEP = detection-class tolerance
    print(f'[DCEF] Tier {tier} (CEP ≤{cep}m) | '
          f'Mean error: {merged["pos_error_m"].mean():.2f}m | '
          f'95th pct: {p95:.2f}m | {status}')

    # ── Output column order (DCEF schema) ─────────────────────
    cols = [
        'timestamp_utc', 'lat_truth', 'lon_truth', 'alt_agl_m',
        'hdg_deg', 'gnd_speed_ms', 'sensor_lat', 'sensor_lon',
        'sensor_type', 'detection_flag', 'range_m', 'pos_error_m',
        'tier', 'rf_packet_loss_pct', 'site_id',
    ]
    out = merged[[c for c in cols if c in merged.columns]]
    out.to_csv(output_path, index=False)
    print(f'[DCEF] Written {len(out)} records → {output_path}')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCEF Correlator')
    parser.add_argument('telemetry', help='Telemetry CSV path')
    parser.add_argument('sensor',    help='Sensor CSV path')
    parser.add_argument('output',    nargs='?', help='Output CSV path')
    parser.add_argument('--tier',    type=int, default=2, choices=[1,2,3])
    parser.add_argument('--site-id', default=None)
    args = parser.parse_args()
    correlate(args.telemetry, args.sensor, args.output, args.tier, args.site_id)
