#!/bin/bash
mkdir -p ../TeachMyAgent/data
cd ../TeachMyAgent/data
wget https://flowers.inria.fr/teachmyagent/baselines_results.zip --no-check-certificate
unzip baselines_results.zip