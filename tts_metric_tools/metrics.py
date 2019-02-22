"""Computes metric between one file and another.

Usage:
    python metrics.py \
        --metric ENUM
        --ref_file FILE \
        --gen_file FILE
"""

import argparse
from collections.abc import Callable
from enum import Enum

import numpy as np

from tts_data_tools import file_io


SupportedMetricsEnum = Enum("SupportedMetricsEnum", ("RMSE", "MSE"))

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


def RMSE(ref_data, gen_data):
    pass


def MSE(ref_data, gen_data):
    pass


def compute_metric(ref_data, gen_data, metric):
    if isinstance(metric, Callable):
        return metric(ref_data, gen_data)

    if metric == SupportedMetricsEnum.RMSE:
        return RMSE(ref_data, gen_data)

    elif metric == SupportedMetricsEnum.MSE:
        return MSE(ref_data, gen_data)

    else:
        raise NotImplementedError("Metric '{}' is not implemented".format(metric))


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
    compute_metric(ref_data, gen_data, args.metric)


if __name__ == "__main__":
    main()

