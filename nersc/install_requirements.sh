#!/bin/bash
# This script installs the dependencies for the nersc demo
# It is assumed that mbirjax has already been installed
conda install conda-forge::tomopy  # was getting an error w/o conda-forge, GA
conda install conda-forge::dxchange
