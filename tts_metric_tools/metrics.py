"""Computes metric between one file and another.

Usage:
    python metrics.py \
        --metric ENUM \
        --ref_file FILE \
        --gen_file FILE \
        [--file_encoding ENUM]
"""

import argparse
from collections.abc import Callable
from enum import Enum
import functools

import numpy as np
from scipy.stats.stats import pearsonr

from tts_data_tools import file_io


def flatten_inputs(func):

    @functools.wraps(func)
    def wrapper(ref_data, gen_data):

        if not isinstance(ref_data, np.ndarray):
            ref_data = np.vstack(ref_data)
        if not isinstance(gen_data, np.ndarray):
            gen_data = np.vstack(gen_data)

        return func(ref_data, gen_data)

    return wrapper


def both_voiced_mask(ref_data, gen_data):
    ref_data_voiced = np.not_equal(ref_data, 0.)
    gen_data_voiced = np.not_equal(gen_data, 0.)

    return np.logical_and(ref_data_voiced, gen_data_voiced)


@flatten_inputs
def RMSE(ref_data, gen_data):
    square_diff = (ref_data - gen_data) ** 2
    mse = np.mean(square_diff)
    rmse = np.sqrt(mse)
    return rmse


@flatten_inputs
def MSE(ref_data, gen_data):
    square_diff = (ref_data - gen_data) ** 2
    mse = np.mean(square_diff)
    return mse


@flatten_inputs
def corr(ref_data, gen_data):
    corr_coef, _ = pearsonr(ref_data, gen_data)
    return corr_coef


@flatten_inputs
def f0_corr(ref_data, gen_data):
    voiced_mask = both_voiced_mask(ref_data, gen_data)

    ref_data_voiced = ref_data[voiced_mask]
    gen_data_voiced = gen_data[voiced_mask]

    return corr(ref_data_voiced, gen_data_voiced)


@flatten_inputs
def lf0_corr(ref_data, gen_data):
    voiced_mask = both_voiced_mask(ref_data, gen_data)

    ref_data_voiced = ref_data[voiced_mask]
    gen_data_voiced = gen_data[voiced_mask]

    return corr(np.exp(ref_data_voiced), np.exp(gen_data_voiced))


#
# Setup Enum and helper functions to allow running metrics from command line
#


SupportedMetricsEnum = Enum("SupportedMetricsEnum", ("RMSE", "MSE", "CORR", "F0_CORR"))


def infer_metric(file_ext):
    """Converts file_ext to a list of SupportedMetricsEnum."""
    if file_ext in ['f0']:
        metrics = ['RMSE', 'F0_CORR']
    elif file_ext in ['lf0']:
        metrics = ['RMSE', 'LF0_CORR']
        pass
    if file_ext in ['dur']:
        metrics = ['RMSE', 'CORR']

    elif file_ext in ['sp', 'mgc']:
        raise NotImplementedError
    elif file_ext in ['ap', 'bap']:
        raise NotImplementedError

    return [SupportedMetricsEnum[metric.upper()] for metric in metrics]


def supported_metrics_enum_type(astring):
    try:
        return SupportedMetricsEnum[astring.upper()]
    except KeyError:
        msg = ', '.join([t.name.lower() for t in SupportedMetricsEnum])
        msg = 'CustomEnumType: use one of {%s}'%msg
        raise argparse.ArgumentTypeError(msg)


def add_arguments(parser):
    parser.add_argument("--metric", action="store", dest="metric", type=supported_metrics_enum_type, required=True,
                        help="The metric to compute.")
    file_io.add_arguments(parser)


def compute(metric_fn_or_name, ref_data, gen_data):
    if isinstance(metric_fn_or_name, Callable):
        metric_fn = metric_fn_or_name
        return metric_fn(ref_data, gen_data)
    elif isinstance(metric_fn_or_name, str):
        metric_name = SupportedMetricsEnum[metric_fn_or_name.upper()]
    else:
        metric_name = metric_fn_or_name

    if metric_name == SupportedMetricsEnum.RMSE:
        return RMSE(ref_data, gen_data)

    elif metric_name == SupportedMetricsEnum.MSE:
        return MSE(ref_data, gen_data)

    elif metric_name == SupportedMetricsEnum.CORR:
        return corr(ref_data, gen_data)

    elif metric_name == SupportedMetricsEnum.F0_CORR:
        return f0_corr(ref_data, gen_data)

    elif metric_name == SupportedMetricsEnum.LF0_CORR:
        return lf0_corr(ref_data, gen_data)

    else:
        raise NotImplementedError("Metric '{}' is not implemented".format(metric_name))


def main():
    parser = argparse.ArgumentParser(description="Script to compute metric between two files.")

    # Arguments necessary for running this file directly on a pair of files.
    parser.add_argument("--file_encoding", action="store", dest="file_encoding", type=file_io.file_encoding_enum_type,
                        help="The encoding to load the file with.")
    parser.add_argument(
        "--ref_file", action="store", dest="ref_file", type=str, required=True, help="Ground truth file.")
    parser.add_argument(
        "--gen_file", action="store", dest="gen_file", type=str, required=True, help="Generated file.")

    add_arguments(parser)
    args = parser.parse_args()

    ref_data = file_io.load(args.ref_file, args.file_encoding, args.feat_dim)
    gen_data = file_io.load(args.gen_file, args.file_encoding, args.feat_dim)
    print(compute(args.metric, [ref_data], [gen_data]))


if __name__ == "__main__":
    main()

