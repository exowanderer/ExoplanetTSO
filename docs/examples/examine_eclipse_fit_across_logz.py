import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ultranest.plot import cornerplot
from glob import glob
import joblib
import os


def combine_to_dataframe_1sigma(ultranests, idx=0, sort_column='aper'):

    percentiles = [32, 50, 68]

    df = ultranest2dataframe(
        ultranests,
        idx=idx,
        percentiles=percentiles,
        sort_column='aper'
    )

    df['median'] = df['perc1']
    df['lower'] = df['median'] - df['perc0']
    df['upper'] = df['perc2'] - df['median']

    del df['perc0']
    del df['perc1']
    del df['perc2']

    return df


def ultranest2dataframe(
        ultranests, idx=0, percentiles=None, sort_column='aper', nbins=100):

    if percentiles is None:
        nbins = int(nbins)
        assert (nbins > 0), (
            'if `percentiles` is None, '
            'then provide `nbins` as an positive integer'
        )

        percentiles = np.linspace(0.01, 99.99, nbins)

    ultranest_cdx = []
    for key, val in ultranests.items():
        aper_rad = key.split('rad_')[-1].replace('.joblb.save', '')
        static_rad, var_rad = aper_rad.split('_')
        static_rad = float(static_rad.replace('p', '.'))
        var_rad = float(var_rad.replace('p', '.'))
        results = val
        perctiles = np.percentile(
            results['weighted_samples']['points'][:, idx],
            percentiles
        )

        dict_ = {
            'aper': aper_rad,
            'static': static_rad,
            'var_rad': var_rad,
            'logz': val['logz'],
            'logzerr': val['logzerr']
        }

        for k, perc_ in enumerate(perctiles):
            dict_[f'perc{k}'] = perc_

        ultranest_cdx.append(dict_)

    df = pd.DataFrame(ultranest_cdx)

    return df.sort_values(by=sort_column).reset_index(drop=True)


def ultranest2hist_df(ultranests, idx=0, sort_column='aper', nbins=25):

    ultranest_hist = []
    for key, val in ultranests.items():
        aper_rad = key.split('rad_')[-1].replace('.joblb.save', '')
        static_rad, var_rad = aper_rad.split('_')
        static_rad = float(static_rad.replace('p', '.'))
        var_rad = float(var_rad.replace('p', '.'))
        results = val

        yhist, xhist = np.histogram(
            results['weighted_samples']['points'][:, idx],
            bins=nbins,
            range=[0, 500],
            density=True,
            weights=None  # TODO: check if we can use logz
        )

        dict_hist = dict(zip(xhist, yhist))

        dict_ = {
            'aper': aper_rad,
            'static': static_rad,
            'var_rad': var_rad,
            'logz': val['logz'],
            'logzerr': val['logzerr']
        } | dict_hist

        ultranest_hist.append(dict_)

    df = pd.DataFrame(ultranest_hist)

    return df.sort_values(by=sort_column).reset_index(drop=True)


def compute_weighted_average(df, value_col, weight_col='logz'):

    sum_logz = np.sum(df[weight_col])
    return np.sum(df[value_col] * df[weight_col]) / sum_logz


def compute_weighted_mid_low_high(df, weight_col='logz', verbose=False):

    sum_logz = np.sum(df[weight_col])

    med_logz = compute_weighted_average(df, 'median', weight_col='logz')
    upper_logz = compute_weighted_average(df, 'upper', weight_col='logz')
    lower_logz = compute_weighted_average(df, 'lower', weight_col='logz')

    if verbose:
        print(
            f'{med_logz:.3f} +{upper_logz:.3f} -{lower_logz:.3f}'
        )

    return med_logz, upper_logz, lower_logz


def plot100_cornerplots(ultranests, fpfs_idx=None):
    for _, results in ultranests.items():
        points = results['weighted_samples']['points']
        if fpfs_idx is not None and np.median(points[:, fpfs_idx]) < 1:
            points[:, fpfs_idx] = points[:, fpfs_idx]*1e6

        cornerplot(points)


if __name__ == '__main__':
    first_savedir = 'hatp26b_ultanest_savedir_take1'
    second_savedir = 'hatp26b_ultanest_savedir_take2'

    glob_string1 = os.path.join(
        first_savedir,
        'hatp26b_ultranest_krdata_2023*'
    )

    glob_string2 = os.path.join(
        second_savedir,
        'hatp26b_ultranest_krdata_2023*'
    )

    ultranests = {}
    for fname in glob(glob_string2):
        aper_rad = fname.split('rad_')[-1].replace('.joblb.save', '')
        ultranests[aper_rad] = joblib.load(fname)['ultranest']

        results = ultranests[aper_rad]
        if np.median(results['weighted_samples']['points'][:, 0]) < 1:
            fpfs_ = results['weighted_samples']['points'][:, 0]*1e6
            results['weighted_samples']['points'][:, 0] = fpfs_
            ultranests[aper_rad] = results

    verbose_plot = False
    if verbose_plot:
        plot100_cornerplots(ultranests, fpfs_idx=0)

    df_fpfs = combine_to_dataframe_1sigma(
        ultranests,
        idx=0,
        sort_column='aper'
    )
    df_ecenter = combine_to_dataframe_1sigma(
        ultranests,
        idx=1,
        sort_column='aper'
    )
    df_logf = combine_to_dataframe_1sigma(
        ultranests,
        idx=2,
        sort_column='aper'
    )

    med_fpfs, upper_fpfs, lower_fpfs = compute_weighted_mid_low_high(
        df_fpfs, weight_col='logz', verbose=True
    )
    med_ecenter, upper_ecenter, lower_ecenter = compute_weighted_mid_low_high(
        df_ecenter, weight_col='logz', verbose=True
    )
    med_logf, upper_logf, lower_logf = compute_weighted_mid_low_high(
        df_logf, weight_col='logz', verbose=True
    )

    df_fpfs.aper_rad = df_fpfs.static + df_fpfs.var_rad
    med_aper_rad = np.median(df_fpfs.aper_rad)
    """
    plt.errorbar(
        df_fpfs.aper_rad,
        df_fpfs.logz,
        df_fpfs.logzerr,
        fmt='o'
    )

    plt.errorbar(
        df_fpfs.aper_rad,
        df_fpfs.logz,
        df_fpfs.logzerr,
        fmt='o'
    )

    plt.errorbar(med_aper_rad, med_fpfs, lolim=lower_fpfs, uplim=upper_fpfs)

    plt.tight_layout()
    """
    df_hist_fpfs = ultranest2hist_df(
        ultranests,
        idx=0,
        sort_column='aper',
        nbins=25
    )

    xhist = np.arange(0, 500, 20)

    sum_logz = np.sum(df_hist_fpfs['logz'])
    yhist = {
        col_: np.sum(df_hist_fpfs[col_] * df_hist_fpfs['logz']) / sum_logz
        for col_ in xhist
    }

    min_yhist = np.min(list(yhist.values()))
    yhist = {key: val - min_yhist for key, val in yhist.items()}

    plt.bar(
        list(yhist.keys()),
        list(yhist.values()),
        width=np.median(np.diff(xhist))*0.9
    )
    plt.show()

    is_static_rad = df_fpfs.var_rad == 0

    fig = plt.figure()
    plt.plot(
        df_fpfs.aper_rad[is_static_rad],
        df_fpfs.logz[is_static_rad],
        'o',
        ms=15,
        label='Static Radii'
    )
    plt.plot(
        df_fpfs.aper_rad[~is_static_rad],
        df_fpfs.logz[~is_static_rad],
        'o',
        ms=15,
        label='Variable Radii'
    )
    plt.title('HAT-P-26b: Median Aperture Radii vs Log(z)', fontsize=20)
    plt.xlabel('Static + Variable Radius', fontsize=20)
    plt.ylabel('Log(z) [evidence]', fontsize=20)
    plt.legend(fontsize=20)
    fig.savefig('HAT-P-26b_Median_Aperture_Radii_vs_LogZ.png')
