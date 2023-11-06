TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL
==================================
* [Paper]
* [Website]
* [Documentation]
----
`TeachMyAgent` is a testbed platform for **Automatic Curriculum Learning** methods. We leverage Box2D procedurally generated environments to assess the performance of teacher algorithms in continuous task spaces.
Our repository provides:

* **Two parametric Box2D environments**: Stumps Tracks and Parkour
* **Multiple embodiments** with different locomotion skills (e.g. bipedal walker, spider, climbing chimpanzee, fish)
* **Two Deep RL students**: SAC and PPO
* **Several ACL algorithms**: ADR, [ALP-GMM](https://github.com/flowersteam/teachDeepRL), Covar-GMM, SPDL, GoalGAN, Setter-Solver, RIAC
* **Two benchmark experiments** using elements above: Skill-specific comparison and global performance assessment
* **Three notebooks for systematic analysis** of results using statistical tests along with visualization tools (plots, videos...) allowing to reproduce our figures

See our [documentation] for an exhaustive list.

![global_schema](TeachMyAgent/graphics/readme_graphics/global_schema.png)

Using this, we performed a benchmark of the previously mentioned ACL methods which can be seen in our [paper]. We also provide additional visualization on our [website].

## Installation

1- Get the repository
```
git clone https://github.com/flowersteam/TeachMyAgent
cd TeachMyAgent/
```
2- Install it, using Conda for example (use Python >= 3.6)
```
conda create --name teachMyAgent python=3.6
conda activate teachMyAgent
pip install -e .
```

**Note: For Windows users, add `-f https://download.pytorch.org/whl/torch_stable.html` to the `pip install -e .` command.**

## Import baseline results from our paper

In order to benchmark methods against the ones we evaluated in our [paper](https://arxiv.org/abs/2103.09815) you must download our results:

1. Go to the `notebooks` folder
2. Make the `download_baselines.sh` script executable: `chmod +x download_baselines.sh`
3. Download results: `./download_baselines.sh`
> **_WARNING:_**  This will download a zip weighting approximayely 4.5GB. Then, our script will extract the zip file in `TeachMyAgent/data`. Once extracted, results will weight approximately 15GB. 

## Usage

See our [documentation] for details on how to use our platform to benchmark ACL methods.

## Development

See [CONTRIBUTING.md] for details.

## Citing

If you use `TeachMyAgent` in your work, please cite the accompanying [paper]:

```bibtex
@inproceedings{romac2021teachmyagent,
  author    = {Cl{\'{e}}ment Romac and
               R{\'{e}}my Portelas and
               Katja Hofmann and
               Pierre{-}Yves Oudeyer},
  title     = {TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep
               {RL}},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning,
               {ICML} 2021, 18-24 July 2021, Virtual Event},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {9052--9063},
  publisher = {{PMLR}},
  year      = {2021}
}
```

[paper]: https://arxiv.org/abs/2103.09815
[website]: http://developmentalsystems.org/TeachMyAgent/
[documentation]: http://developmentalsystems.org/TeachMyAgent/doc/

[Paper]: https://arxiv.org/abs/2103.09815
[Website]: http://developmentalsystems.org/TeachMyAgent/
[Documentation]: http://developmentalsystems.org/TeachMyAgent/doc/

[CONTRIBUTING.md]: CONTRIBUTING.md
