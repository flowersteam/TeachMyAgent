from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor in [6, 7] , \
    "This repo is designed to work with Python 3.6 or 3.7." \
    + "Please install it before proceeding."

setup(
    name='TeachMyAgent',
    py_modules=['TeachMyAgent'],
    version="1.0",
    install_requires=[
        'cloudpickle==1.2.0',
        'gym[atari,box2d,classic_control]>=0.10.8',
        'ipython',
        'joblib',
        'matplotlib',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'sklearn',
        'imageio',
        'seaborn==0.8.1',
        'dm-sonnet<2',
        'tensorflow-probability==0.7.0',
        'torch==1.4.0',
        'setuptools',
        'setuptools_scm',
        'pep517',
        'treelib',
        'gizeh',
        'tqdm',
        'emcee',
        'notebook',
        'huggingface_hub',
    ],
    description="TeachMyAgent: A benchmark to study and compare ACL algorithms for DeepRL in continuous procedural environments.",
    author="Clément Romac",
)

# ensure there is some tensorflow build with version above 1.4
import pkg_resources
import re
tf_pkg = None
for tf_pkg_name in ['tensorflow', 'tensorflow-gpu', 'tf-nightly', 'tf-nightly-gpu']:
    try:
        tf_pkg = pkg_resources.get_distribution(tf_pkg_name)
    except pkg_resources.DistributionNotFound:
        pass
assert tf_pkg is not None, 'TensorFlow needed, of version above 1.4'
from distutils.version import LooseVersion
tf_version = LooseVersion(re.sub(r'-?rc\d+$', '', tf_pkg.version))
assert tf_version >= LooseVersion('1.4.0') and tf_version <= LooseVersion('1.15.5'), \
    'TensorFlow version between 1.4 and 1.15.5 required'
