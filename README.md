AutoNLP/AutoDL starting kit
======================================

THE ORIGINAL VERSION IS FROM https://github.com/zhengying-liu/autodl_starting_kit_stable,
MODIFIED BY WENHAO LI

## Contents
AutoDL_ingestion_program/: The code and libraries used on Codalab to run your submmission.

AutoDL_scoring_program/: The code and libraries used on Codalab to score your submmission.

AutoDL_sample_code_submission/: An example of code submission you can use as template.

AutoDL_sample_data/: Some sample data to test your code before you submit it.

run_local_test.py: A python script to simulate the runtime in codalab

## Local development and testing
To make your own submission to AutoNLP/AutoDL challenge, you need to modify the
file `model.py` in `AutoDL_sample_code_submission/`, which implements the logic
of your algorithm. You can then test it on your local computer using Docker,
in the exact same environment as on the CodaLab challenge platform. Advanced
users can also run local test without Docker, if they install all the required
packages,

If you are new to docker, install docker from https://docs.docker.com/get-started/.
Then, at the shell, run:
```
cd path/to/autonlp_starting_kit/
docker run -it -v "$(pwd):/app/codalab" wahaha909/autonlp:gpu
```
The option `-v "$(pwd):/app/codalab"` mounts current directory
(`autonlp_starting_kit/`) as `/app/codalab`. If you want to mount other
directories on your disk, please replace `$(pwd)` by your own directory.

The Docker image
```
wahaha909/autonlp:gpu
```
has Nvidia GPU supports. see the
[site](https://hub.docker.com/r/wahaha909/autonlp)
to check installed packages in the docker image.

Make sure you use enough RAM (**at least 4GB**).

You will then be able to run the `ingestion program` (to produce predictions)
and the `scoring program` (to evaluate your predictions) on toy sample data.
In the AutoNLP/AutoDL challenge, these two programs will run in parallel to give
real-time feedback (with learning curves). So we provide a Python script to
simulate this behavior:
```
python run_local_test.py
```
Then you can view the real-time feedback with a learning curve by opening the
HTML page in `AutoDL_scoring_output/`.

The full usage is
```
python run_local_test.py -dataset_dir=./AutoDL_sample_data/DEMO -code_dir=./AutoDL_sample_code_submission
```
You can change the argument `dataset_dir` to other datasets (e.g. the five
practice datasets we provide). On the other hand,
you can also modify the directory containing your other sample code
(`model.py`).

## Download practice datasets
We provide 5 practice datasets for participants. They can use these datasets to:
1. Do local test for their own algorithm;
2. Enable meta-learning.

You may refer to [codalab site](https://autodl.lri.fr/competitions/35#learn_the_details-get_data) for practice datasets.

Unzip the zip file and you'll get 5 datasets.

## Prepare a ZIP file for submission on CodaLab
Zip the contents of `AutoDL_sample_code_submission`(or any folder containing
your `model.py` file) without the directory structure:
```
cd AutoDL_sample_code_submission/
zip -r mysubmission.zip *
```
then use the "Upload a Submission" button to make a submission to the
competition page on CodaLab platform.

Tip: to look at what's in your submission zip file without unzipping it, you
can do
```
unzip -l mysubmission.zip
```

## Report bugs and create issues

If you run into bugs or issues when using this starting kit, please create
issues on the
[*Issues* page](https://github.com/mortal123/autonlp_starting_kit/issues)
of this repo. Two templates will be given when you click the **New issue**
button.

## Contact us
If you have any questions, please contact us via:
<autonlp2019@4paradigm.com>
