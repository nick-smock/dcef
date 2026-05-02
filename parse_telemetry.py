"""
parse_telemetry.py — DCEF Tool 01
Reads a PX4 ULog binary file and extracts position, heading,
ground speed, and RF link quality fields. Resamples to 1 Hz
and writes a DCEF-schema-compliant telemetry CSV.

Requirements: pyulog, pandas, numpy
Usage: python parse_telemetry.py <flight.ulg> [output.csv]
"""

import sys
import numpy as np
import pandas as pd
from pyulog import ULog
from pyulog.px4 import PX4ULog

def parse_ulog(ulg_path: str, output_path: str = None) -> pd.DataFrame:
    if output_path is None:
        output_path = ulg_path.replace('.ulg', '_telemetry.csv')

    print(f'[DCEF] Parsing: {ulg_path}')
    log = ULog(ulg_path)
    px4 = PX4ULog(log)

    # ── Extract vehicle_global_position ───────────────────────
    gps = log.get_dataset('vehicle_global_position')
    gps_df = pd.DataFrame({
        'timestamp_us': gps.data['timestamp'],
        'lat_truth':    gps.data['lat'],
        'lon_truth':    gps.data['lon'],
        'alt_agl_m':    gps.data['alt'] - gps.data.get('terrain_alt',
                        np.zeros(len(gps.data['alt']))),
    })

    # ── Extract vehicle_attitude for heading ──────────────────
    att = log.get_dataset('vehicle_attitude')
    q = np.column_stack([
        att.data['q[0]'], att.data['q[1]'],
        att.data['q[2]'], att.data['q[3]'],
    ])
    # Yaw from quaternion
    yaw = np.degrees(np.arctan2(
        2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]),
        1 - 2*(q[:,2]**2 + q[:,3]**2)
    )) % 360
    att_df = pd.DataFrame({
        'timestamp_us': att.data['timestamp'],
        'hdg_deg':      yaw,
    })

    # ── Extract vehicle_local_position for ground speed ───────
    vel = log.get_dataset('vehicle_local_position')
    spd = np.sqrt(vel.data['vx']**2 + vel.data['vy']**2)
    vel_df = pd.DataFrame({
        'timestamp_us': vel.data['timestamp'],
        'gnd_speed_ms': spd,
    })

    # ── Extract telemetry_status for RF link quality ──────────
    try:
        link = log.get_dataset('telemetry_status')
        rf_df = pd.DataFrame({
            'timestamp_us':      link.data['timestamp'],
            'rf_packet_loss_pct': link.data['rxerrors'] /
                                  np.maximum(link.data['rxerrors'] +
                                             link.data['fixed'], 1) * 100,
        })
    except Exception:
        # Telemetry status not always present
        rf_df = pd.DataFrame({
            'timestamp_us':      gps_df['timestamp_us'],
            'rf_packet_loss_pct': 0.0,
        })

    # ── Merge all streams on microsecond timestamp ────────────
    base = gps_df.set_index('timestamp_us')
    for df, col in [(att_df, 'hdg_deg'),
                    (vel_df, 'gnd_speed_ms'),
                    (rf_df,  'rf_packet_loss_pct')]:
        base = base.join(
            df.set_index('timestamp_us')[col],
            how='outer'
        )

    base = base.sort_index().interpolate(method='index').dropna()

    # ── Convert timestamp, resample to 1 Hz ──────────────────
    # ULog timestamps are microseconds from boot; GPS gives UTC offset
    try:
        utc_offset_us = log.start_timestamp
    except Exception:
        utc_offset_us = 0

    base.index = pd.to_datetime(base.index + utc_offset_us, unit='us', utc=True)
    df_1hz = base.resample('1S').mean().dropna()
    df_1hz.index.name = 'timestamp_utc'
    df_1hz = df_1hz.reset_index()

    # ── Add schema fields ─────────────────────────────────────
    df_1hz['site_id'] = 'SITE-UNKNOWN'

    # Round to schema precision
    df_1hz['lat_truth']          = df_1hz['lat_truth'].round(8)
    df_1hz['lon_truth']          = df_1hz['lon_truth'].round(8)
    df_1hz['alt_agl_m']          = df_1hz['alt_agl_m'].round(2)
    df_1hz['hdg_deg']            = df_1hz['hdg_deg'].round(1)
    df_1hz['gnd_speed_ms']       = df_1hz['gnd_speed_ms'].round(2)
    df_1hz['rf_packet_loss_pct'] = df_1hz['rf_packet_loss_pct'].round(2)

    cols = ['timestamp_utc', 'lat_truth', 'lon_truth', 'alt_agl_m',
            'hdg_deg', 'gnd_speed_ms', 'rf_packet_loss_pct', 'site_id']
    out = df_1hz[cols]
    out.to_csv(output_path, index=False)
    print(f'[DCEF] Written {len(out)} records → {output_path}')
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python parse_telemetry.py <flight.ulg> [output.csv]')
        sys.exit(1)
    ulg   = sys.argv[1]
    out   = sys.argv[2] if len(sys.argv) > 2 else None
    parse_ulog(ulg, out)
