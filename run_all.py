import argparse
import os
from os import listdir
from os import path

from multiprocessing import Pool
import datetime

DATA_HOME = "./all_datasets"

def _HERE(*args):
  """Helper function for getting the current directory of this script."""
  h = os.path.dirname(os.path.realpath(__file__))
  return os.path.abspath(os.path.join(h, *args))

def run_baseline(args):
    dname, code_dir, exp_dir = args
    output_dir = path.join(exp_dir, dname)
    print("Running {} over {}, start from {}".format(code_dir, dname,
                                                     datetime.datetime.now()))

    os.makedirs(output_dir)
    copy_code_dir = path.join(output_dir, "code")
    os.system("cp -r {} {}".format(code_dir, copy_code_dir))
    args = {
            "CODE_DIR": copy_code_dir,
            "DATASET_DIR": path.join(DATA_HOME, dname),
            "PREDICTION_DIR": path.join(output_dir, 'prediction'),
            "SCORE_DIR": path.join(output_dir, 'score'),
            "TIME_BUDGET": 10800, # 3 hours
            "STDOUT": path.join(output_dir, 'stdout'),
            "STDERR": path.join(output_dir, 'stderr')
            }
    os.system(("python run_local_test_detailed.py --code_dir={CODE_DIR} "
               "--dataset_dir={DATASET_DIR} --prediction_dir={PREDICTION_DIR} "
               "--score_dir={SCORE_DIR} --time_budget={TIME_BUDGET} "
               "1>{STDOUT} 2>{STDERR}").format(**args))

    print("{} finished, end at {}".format(dname, datetime.datetime.now()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_dir', type=str)
    parser.add_argument('--exp_dir', type=str)
    args = parser.parse_args()
    code_dir = args.code_dir
    exp_dir = args.exp_dir

    os.system("rm -rf {}".format(exp_dir))
    datasets = listdir('./all_datasets')
    print(datasets)

    with Pool(5) as p:
        p.map(run_baseline, [(dname, code_dir, exp_dir) for dname in datasets])
