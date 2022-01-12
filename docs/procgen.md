# Running an experiment using OpenAI's procgen

## Installation

To isolate the setup, using conda is recommended.

1. Clone this fork of TeachMyAgent locally
2. Clone Nikita's fork of the procgen repo which supports parametric environment generation: https://github.com/meln1k/procgen
3. Create a new conda environment with python 3.7: `conda create --name teachMyAgent-procgen python=3.7`
4. Activate the environment: `conda activate teachMyAgent-procgen`
5. Go into the procgen directory and install the dependencies into the environment: `conda env update --name teachMyAgent-procgen --file environment.yml`
6. Install tensorflow 1.15, e.g. `conda install tensorflow=1.1`
7. Leave the procgen directory and go into the TeachMyAgent dir.
8. Install TeachMyAgent dependencies: `pip install -e .`

Done! Now we're ready to launch an experiment.

## Launching an experiment

At the moment only the coinrun environment is supported.

To launch the experiment in the coinrun environment, you need to provide the environment `--env coinrun` flag and optionally the seeds which will be used by the random generator: `--level-seeds 243234 123152 423413 324123 31234234 234321`. Every seed corresponds to a different section of the environment, so it is possible to change the number of section, but keep in mind that the original coinrun environment was running on 1-6 sections.

Example:

`python run.py --exp_name procgen-test --env coinrun --student ppo --nb_env_steps 4 --teacher Random --use_pretrained_samples --level-seeds 243234 123152 423413 324123 31234234 234321`