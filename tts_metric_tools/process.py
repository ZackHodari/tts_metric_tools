"""Runs the feature extraction on the waveforms and binarises the label files.
Usage:
    python process.py \
        [--lab_dir DIR] [--state_level] \
        [--wav_dir DIR] \
        [--id_list FILE] \
        --out_dir DIR
"""

import argparse
import functools
import os

from tts_data_tools import file_io
from tts_data_tools import utils

from . import metrics


def add_arguments(parser):
    parser.add_argument("--ref_dir", action="store", dest="lab_dir", type=str, default=None,
                        help="Directory of the ground truth files.")
    parser.add_argument("--gen_dir", action="store", dest="wav_dir", type=str, default=None,
                        help="Directory of the generated files.")
    parser.add_argument("--id_list", action="store", dest="id_list", type=str, default=None,
                        help="List of file ids to compute the metric for (must be contained in ref_dir and gen_dir).")
    parser.add_argument("--out_file", action="store", dest="out_file", type=str, default=None,
                        help="File to save the output to.")
    parser.add_argument("--feat_ext", action="store", dest="feat_ext", type=str, default=None,
                        help="File extension of the features being loaded.")
    parser.add_argument("--file_is_txt", action="store_true", dest="file_is_txt", default=False,
                        help="Whether the files being loaded are in .txt files (not .npy files).")
    metrics.add_arguments(parser)


def compute_metric(metric, ref_dir, gen_dir, id_list, feat_ext, out_file=None, is_npy=True, feat_dim=None):
    """Processes label and wave files in id_list, saves the numerical labels and vocoder features to .npy binary files.
    Args:
        metric (str OR metrics.SupportedMetricsEnum): The metric to compute.
        ref_dir (str): Directory containing the ground truth files.
        gen_dir (str): Directory containing the generated files.
        id_list (str): List of file basenames to process.
        feat_ext (str): Name of the feature being compared, also the file extension of individual files.
        out_file (str): File to save the output to.
        is_npy (bool): If True, uses `file_io.load_bin`, otherwise uses `file_io.load_txt` to load each file.
        """
    file_ids = utils.get_file_ids(ref_dir, id_list)
    _file_ids = utils.get_file_ids(gen_dir, id_list)

    if len(file_ids) != len(_file_ids) or sorted(file_ids) != sorted(_file_ids):
        raise ValueError("Please provide id_list, or ensure that wav_dir and lab_dir contain the same files.")

    if is_npy:
        load_fn = functools.partial(file_io.load_bin, feat_dim=feat_dim)
    else:
        load_fn = file_io.load_txt

    # Load the reference and generated data
    ref_data = file_io.load_dir(load_fn, ref_dir, file_ids, feat_ext)
    gen_data = file_io.load_dir(load_fn, gen_dir, file_ids, feat_ext)

    metric_value = metrics.compute_metric(ref_data, gen_data, metric)

    print("{metric} for {id_list} is {value}\n"
          "\tref_dir = {ref_dir}\n"
          "\tgen_dir = {gen_dir}".format(metric=metric, id_list=id_list, value=metric_value,
                                         ref_dir=ref_dir, gen_dir=gen_dir))

    if out_file is not None:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        file_io.save_txt(metric_value, out_file)


def main():
    parser = argparse.ArgumentParser(
        description="Script to calculate metrics for directories of files.")
    add_arguments(parser)
    args = parser.parse_args()

    compute_metric(args.metric, args.ref_dir, args.gen_dir, args.id_list, args.feat_ext,
                   out_file=args.out_file, is_npy=not args.file_is_txt, feat_dim=args.feat_dim)


if __name__ == "__main__":
    main()

