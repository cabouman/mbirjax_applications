# Repository for MBIRJAX demonstrations
## Overview
This repository contains scripts that demonstrate the usage of [MBIRJAX](https://github.com/cabouman/mbirjax) in selected CT applications.
## Quick start guide
1. *Install the conda environment and package for MBIRJAX.* Instructions available [here](https://github.com/cabouman/mbirjax).
2. *Clone this repository:*
   ```
   git clone git@github.com:cabouman/mbirjax_applications.git
   ```
3. Run demo scripts for the application of your choice. Available applications include:
   * Cone-beam CT reconstruction with NorthStar Instrument (NSI) system:
     ```
     pip install mbirjax
     cd mbirjax_applications/nsi
     python demo_nsi.py
     ```
     
   * Parallel-beam CT reconstruction with data provided by Wiebke Koepp, Tanny Andrea Chavez Esparza, Alexander Hexemer, and Dula Parkinson, Advanced Light Source, LBNL:
     ```
     pip install mbirjax
     cd mbirjax_applications/nersc
     python demo_nsi.py
     ```
