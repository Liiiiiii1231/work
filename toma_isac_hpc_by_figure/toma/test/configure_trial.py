"""Configure the trial only after an explicit physical ground geometry is selected."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def configure(altitude, center_distance, distance_kind, radial_half_width=10.):
    if altitude <= 0 or center_distance <= 0:
        raise ValueError('Altitude and distance must be positive')
    if distance_kind == 'slant':
        if center_distance < altitude:
            raise ValueError('Slant distance cannot be smaller than altitude')
        horizontal = math.sqrt(center_distance**2 - altitude**2)
    else:
        horizontal = center_distance
    if radial_half_width <= 0 or horizontal <= radial_half_width:
        raise ValueError('Radial half width must be positive and smaller than the baseline horizontal radius')
    for name in ('simulation_config.json', 'tradeoff_config.json'):
        path = ROOT / name
        bundle = json.loads(path.read_text(encoding='utf-8'))
        bundle['protocol']['n_mc'] = 10
        bundle['protocol']['convergence']['n_mc'] = 10
        for model in bundle['models'].values():
            model['system'].update(M=8, N_c=4)
            # Preserve the existing sensing factor count; only array size changes.
            model['experiment']['n_mc'] = 10
            model['experiment'].pop('ground_disk',None)
            model['experiment']['ground_annulus'] = {
                'horizontal_radius_min_m': horizontal-radial_half_width,
                'horizontal_radius_max_m': horizontal+radial_half_width,
                'baseline_horizontal_radius_m': horizontal,
                'uav_altitude_m': altitude,
                'center_distance_m': center_distance,
                'center_distance_kind': distance_kind,
                'distribution': 'independent_uniform_area',
            }
        path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'M':8,'N_c':4,'N_tx':16,'N_rx':16,'horizontal_radius_m':[horizontal-radial_half_width,horizontal+radial_half_width],'ground_z_m':-altitude,'n_mc':10}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--altitude', type=float, required=True)
    ap.add_argument('--center-distance', type=float, required=True)
    ap.add_argument('--distance-kind', choices=['slant','horizontal'], required=True)
    args = ap.parse_args()
    configure(args.altitude, args.center_distance, args.distance_kind)
