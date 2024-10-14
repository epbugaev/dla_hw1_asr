import argparse
import os

from src.metrics.utils import calc_cer, calc_wer


def process_files(predictions_folder, results_folder):
    prediction_files = sorted(os.listdir(predictions_folder))
    result_files = sorted(os.listdir(results_folder))

    if len(prediction_files) != len(result_files):
        raise ValueError(
            "The number of files in 'predictions' and 'results' folders does not match."
        )

    total_cer = 0
    total_wer = 0

    for pred_file, res_file in zip(prediction_files, result_files):
        pred_file_path = os.path.join(predictions_folder, pred_file)
        res_file_path = os.path.join(results_folder, res_file)

        with open(pred_file_path, "r") as pred_f:
            prediction_content = pred_f.read()

        with open(res_file_path, "r") as res_f:
            result_content = res_f.read()

        total_wer += calc_wer(prediction_content, result_content)
        total_cer += calc_cer(prediction_content, result_content)

    print("CER:", total_cer / len(prediction_files))
    print("WER:", total_wer / len(prediction_files))


def main():
    parser = argparse.ArgumentParser(description="Process prediction and result files.")
    parser.add_argument(
        "predictions_folder",
        type=str,
        help="Path to the folder containing prediction files.",
    )
    parser.add_argument(
        "results_folder", type=str, help="Path to the folder containing result files."
    )

    args = parser.parse_args()

    process_files(args.predictions_folder, args.results_folder)


if __name__ == "__main__":
    main()
