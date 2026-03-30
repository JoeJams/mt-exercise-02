#! /bin/bash

# virtualenv must be installed on your system, install with e.g.
# pip install virtualenv

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

mkdir -p $base/venvs

# python3 needs to be installed on your system
# removed '3' for my system as it doesn't work otherwise

python -m virtualenv -p python $base/venvs/torch3

echo "To activate your environment:"
echo "source $base/venvs/torch3/Scripts/activate"
