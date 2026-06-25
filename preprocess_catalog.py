#!/usr/bin/env python3
"""FRB Catalog preprocessing script

Reads raw FITS catalog, filters by criteria, outputs clean data.
Analysis code reads preprocessed data, never touching the raw catalog directly.

Usage:
    python preprocess_catalog.py
    python preprocess_catalog.py --input data/chimefrbcat2.fits --output data/filtered
"""

import numpy as np
import argparse
from pathlib import Path
from astropy.io import fits


def load_raw_catalog(fname):
    """Load raw FITS catalog"""
    with fits.open(fname) as hdul:
        data = hdul[1].data
        col_names = data.columns.names
        print(f"Raw data: {len(data)} bursts")
        print(f"Column names: {col_names}")
        return data, col_names


def filter_minimum(data):
    """Minimum filter: only exclude repeaters and bursts without fluence

    Filter criteria:
    1. repeater_name is empty -> one-off FRB
    2. fluence is not NaN -> has valid flux measurement
    """
    n_raw = len(data)

    # exclude repeaters
    rep = np.array(data['repeater_name'])
    is_oneoff = np.array([str(r).strip() == '' for r in rep])
    n_after_rep = np.sum(is_oneoff)

    # exclude bursts without fluence
    fluence = np.array(data['fluence'], dtype=float)
    has_fluence = ~np.isnan(fluence)
    n_after_fluence = np.sum(is_oneoff & has_fluence)

    # combined filter
    mask = is_oneoff & has_fluence
    filtered = data[mask]

    print(f"\nFilter results:")
    print(f"  Raw:         {n_raw}")
    print(f"  Drop repeaters: {n_raw} -> {n_after_rep} (excluded {n_raw - n_after_rep})")
    print(f"  Drop no-fluence: {n_after_rep} -> {n_after_fluence} (excluded {n_after_rep - n_after_fluence})")
    print(f"  Final sample: {n_after_fluence} one-off FRBs")

    return filtered


def extract_columns(data):
    """Extract analysis columns, unify units

    Output columns:
    - name: FRB name
    - fluence: flux x duration [Jy ms]
    - width: pulse width [ms] (converted from seconds)
    - dm_obs: observed dispersion measure [pc cm^-3]
    - dm_exc_ne2001: DM - DM_MW(NE2001) [pc cm^-3]
    - dm_exc_ymw16: DM - DM_MW(YMW16) [pc cm^-3]
    """
    n = len(data)
    result = {}

    # name
    result['name'] = np.array(data['tns_name'])

    # fluence [Jy ms]
    result['fluence'] = np.array(data['fluence'], dtype=float)

    # width: CHIME bc_width is in seconds, convert to milliseconds
    width = np.array(data['bc_width'], dtype=float)
    med_w = np.nanmedian(width)
    if np.isfinite(med_w) and med_w < 1.0:
        width = width * 1000.0
    result['width'] = width

    # DM
    result['dm_obs'] = np.array(data['dm_fitb'], dtype=float)
    result['dm_exc_ne2001'] = np.array(data['dm_exc_ne2001'], dtype=float)
    result['dm_exc_ymw16'] = np.array(data['dm_exc_ymw16'], dtype=float)

    return result


def save_fits(result, fname):
    """Save as FITS format"""
    cols = []
    for key in result:
        if result[key].dtype.kind in ('U', 'S'):
            # string column: use max length to avoid truncation
            max_len = max(len(s) for s in result[key])
            col = fits.Column(name=key, format=f'{max_len}A', array=result[key])
        else:
            col = fits.Column(name=key, format='D', array=result[key])
        cols.append(col)

    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header['CATALOG'] = 'CHIME/FRB Catalog 2 (filtered)'
    hdu.header['NAXIS2'] = len(result['fluence'])
    hdu.header['WUNIT'] = 'ms'

    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(fname, overwrite=True)
    print(f"  FITS: {fname}")


def save_txt(result, fname):
    """Save as TXT format (space-delimited, first row is column names)"""
    keys = list(result.keys())
    header = ' '.join(keys)

    # separate string columns and numeric columns
    str_keys = [k for k in keys if result[k].dtype.kind in ('U', 'S', 'O')]
    num_keys = [k for k in keys if result[k].dtype.kind not in ('U', 'S', 'O')]

    # write with mixed format
    with open(fname, 'w') as f:
        f.write(f'# {header}\n')
        for i in range(len(result[keys[0]])):
            parts = []
            for k in keys:
                val = result[k][i]
                if k in str_keys:
                    parts.append(str(val))
                else:
                    parts.append(f'{float(val):.6f}')
            f.write(' '.join(parts) + '\n')
    print(f"  TXT:  {fname}")


def print_summary(result):
    """Print data summary"""
    print(f"\nData summary:")
    for key in result:
        vals = result[key]
        if vals.dtype.kind in ('U', 'S', 'O'):
            print(f"  {key:>15}: {len(vals)} entries")
        else:
            print(f"  {key:>15}: min={np.nanmin(vals):.4f}  max={np.nanmax(vals):.4f}  median={np.nanmedian(vals):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FRB Catalog preprocessing')
    parser.add_argument('-i', '--input', default='data/chimefrbcat2.fits',
                        help='Input FITS file path')
    parser.add_argument('-o', '--output', default='data/filtered',
                        help='Output file path (without extension)')
    args = parser.parse_args()

    print("=" * 50)
    print("FRB Catalog Preprocessing")
    print("=" * 50)

    # 1. load raw data
    print(f"\n[1/4] Load raw data: {args.input}")
    data, col_names = load_raw_catalog(args.input)

    # 2. filter
    print(f"\n[2/4] Minimum filter (exclude repeaters + no-fluence)")
    filtered = filter_minimum(data)

    # 3. extract columns and convert units
    print(f"\n[3/4] Extract columns and convert units")
    result = extract_columns(filtered)
    print_summary(result)

    # 4. save
    print(f"\n[4/4] Save preprocessed data")
    fits_out = args.output + '.fits'
    txt_out = args.output + '.txt'
    Path(fits_out).parent.mkdir(parents=True, exist_ok=True)
    save_fits(result, fits_out)
    save_txt(result, txt_out)

    print(f"\nDone! Analysis code uses preprocessed data:")
    print(f'  config.json: "catalog_path": "{fits_out}"')
