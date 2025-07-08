# Repository for MBIRJAX demonstrations
## Overview
This repository contains scripts that demonstrate the usage of [MBIRJAX](https://github.com/cabouman/mbirjax) in selected CT applications.
## Quick start guide
1. *Install the conda environment and package for MBIRJAX.* Instructions available [here](https://github.com/cabouman/mbirjax).
2. *Clone this repository:*
   ```
   git clone git@github.com:cabouman/mbirjax_applications.git
   ```
3. Run demo scripts for the application of your choice. Note that if you do not have a GPU with cuda, 
then replace `pip install mbirjax[cuda12]` with `pip install mbirjax`

Available applications include:
   * Cone-beam CT reconstruction using NorthStar Instrument (NSI) data:
     ```
     pip install mbirjax[cuda12]
     cd mbirjax_applications/nsi
     python demo_fdk_mbir_compare.py
     ```
   * View Selection using VCL as published in ICCP/PAMI 2025:
     ```
     pip install mbirjax[cuda12]
     cd mbirjax_applications/vcls
     python demo_vcls.py
     ```
   * Parallel-beam CT reconstruction using NERSC/Advanced Light Source (ALS) data:
     ```
     pip install mbirjax[cuda12]
     cd mbirjax_applications/nersc
     python demo_nersc_public.py
     ```
