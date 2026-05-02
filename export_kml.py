"""
export_kml.py — DCEF Tool 06
Exports a DCEF correlated CSV as a KML file containing:
  - Ground truth track (orange LineString)
  - Sensor-reported track (blue LineString)
  - Detection event placemarks (green pushpins)
  - No-detect event placemarks (grey pins, optional)

Loads directly into Google Earth for site-specific overlay.

Requirements: pandas, simplekml
Usage: python export_kml.py <correlated.csv> [output.kml]
"""

import sys
import pandas as pd
import simplekml

SITE_ALT_M = 0   # AGL offset; set to site elevation MSL for accurate overlay


def export_kml(csv_path: str, output_path: str = None,
               include_no_detect: bool = False):
    if output_path is None:
        output_path = csv_path.replace('.csv', '.kml')

    df = pd.read_csv(csv_path)
    df['detection_flag'] = df['detection_flag'].astype(bool)
    site_id = df['site_id'].iloc[0] if 'site_id' in df.columns else 'DCEF'
    print(f'[DCEF] Exporting {len(df)} records for {site_id}')

    kml = simplekml.Kml(name=f'DCEF — {site_id}')

    # ── Folder: Ground Truth Track ────────────────────────────
    truth_folder = kml.newfolder(name='Ground Truth (Telemetry)')
    truth_line   = truth_folder.newlinestring(name='Truth Track')
    truth_line.coords = [
        (row['lon_truth'], row['lat_truth'],
         row.get('alt_agl_m', 0) + SITE_ALT_M)
        for _, row in df.iterrows()
    ]
    truth_line.altitudemode = simplekml.AltitudeMode.relativetoground
    truth_line.extrude      = 0
    truth_line.style.linestyle.width = 2.5
    truth_line.style.linestyle.color = simplekml.Color.rgb(242, 97, 21)  # DCEF orange

    # ── Folder: Sensor Track ──────────────────────────────────
    sensor_folder = kml.newfolder(name='Sensor Track (Radar)')
    sensor_line   = sensor_folder.newlinestring(name='Sensor Track')
    sensor_line.coords = [
        (row['sensor_lon'], row['sensor_lat'],
         row.get('alt_agl_m', 0) + SITE_ALT_M)
        for _, row in df.iterrows()
        if pd.notna(row.get('sensor_lat'))
    ]
    sensor_line.altitudemode = simplekml.AltitudeMode.relativetoground
    sensor_line.style.linestyle.width = 1.8
    sensor_line.style.linestyle.color = simplekml.Color.rgb(74, 158, 255)  # blue

    # ── Folder: Detection Events ──────────────────────────────
    det_folder = kml.newfolder(name='Detection Events')
    det_df = df[df['detection_flag']]
    for _, row in det_df.iterrows():
        pnt = det_folder.newpoint(
            name=f"DET @ {row.get('range_m', '—'):.0f}m",
            coords=[(row['lon_truth'], row['lat_truth'],
                     row.get('alt_agl_m', 0) + SITE_ALT_M)]
        )
        pnt.altitudemode = simplekml.AltitudeMode.relativetoground
        pnt.style.iconstyle.color = simplekml.Color.rgb(74, 222, 128)  # green
        pnt.style.iconstyle.scale = 0.6
        pnt.style.iconstyle.icon.href = (
            'http://maps.google.com/mapfiles/kml/pushpin/grn-pushpin.png'
        )
        err = row.get('pos_error_m', 0)
        pnt.description = (
            f"Timestamp: {row.get('timestamp_utc', '—')}\n"
            f"Range: {row.get('range_m', '—'):.1f} m\n"
            f"Pos error: {err:.2f} m\n"
            f"Heading: {row.get('hdg_deg', '—'):.1f}°\n"
            f"Speed: {row.get('gnd_speed_ms', '—'):.2f} m/s"
        )

    # ── Folder: No-Detect Events (optional) ───────────────────
    if include_no_detect:
        nodet_folder = kml.newfolder(name='No-Detect Events')
        nodet_df = df[~df['detection_flag']]
        for _, row in nodet_df.iterrows():
            pnt = nodet_folder.newpoint(
                name=f"MISS @ {row.get('range_m', '—'):.0f}m",
                coords=[(row['lon_truth'], row['lat_truth'],
                         row.get('alt_agl_m', 0) + SITE_ALT_M)]
            )
            pnt.altitudemode = simplekml.AltitudeMode.relativetoground
            pnt.style.iconstyle.color = simplekml.Color.rgb(100, 100, 100)
            pnt.style.iconstyle.scale = 0.4

    kml.save(output_path)
    n_det = det_df.shape[0]
    print(f'[DCEF] Saved → {output_path} | '
          f'{n_det} detections, {len(df)-n_det} no-detects')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python export_kml.py <correlated.csv> [output.kml] [--include-no-detect]')
        sys.exit(1)
    inc = '--include-no-detect' in sys.argv
    export_kml(sys.argv[1],
               sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None,
               include_no_detect=inc)
