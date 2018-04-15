import argparse
import logging
import os
from pathlib import Path

from evaluator.EvaluationSummary import EvaluationSummary

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize clustering results.')
    parser.add_argument('iterations', metavar='I', type=int, nargs='+',
                        help='iterations to summarize')
    args = parser.parse_args()

    # Current directory
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Useful directories
    temp_dir = os.path.join(dir_path, "..", "temp")
    out_dir = os.path.join(temp_dir, 'visualization', 'summary')

    # Make sure directories exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    summary = EvaluationSummary()

    print(args.iterations)

    # Load all evaluations
    for i in args.iterations:
        logging.info("Loading evaluation statistics from iteration %d" % i)
        temp_visualization_dir = os.path.join(temp_dir, 'visualization', '{:02d}'.format(i))
        evaluation_file = os.path.join(temp_visualization_dir, 'evaluation.csv')

        summary.add(evaluation_file)

    # Save the summary evaluation
    summary.save(out_dir)
