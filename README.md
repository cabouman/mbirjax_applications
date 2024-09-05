# Repository for MBIRJAX demonstrations
## Overview
This repository contains scripts that demonstrate the usage of [MBIRJAX](https://github.com/cabouman/mbirjax) in selected CT applications.
## Quick start guide
1. *Install the conda environment and package for MBIRJAX.* Instructions available [here](https://github.com/cabouman/mbirjax).
2. *Clone this repository:*
   ```
   git clone git@github.com:cabouman/mbirjax_applications.git
   ```
3. Run demo scripts for the application of your choice. Availble applications include:
   * Cone-beam CT reconstruction with NorthStar Instrument (NSI) system:
     ```
     pip install mbirjax
     cd mbirjax_applications/nsi
     python demo_nsi.py
     ```
   * NERSC:
     ```
     cd mbirjax_applications/nersc
     source install_requirements.sh 
     python demo_nersc.py
     ```
On machines utilizing Nividia GPUs, it may be necessary to install the packages in install_requirements.sh before installing MBIRJAX into your current environment
