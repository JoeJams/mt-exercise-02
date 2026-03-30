#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models/drop

num_threads=4
device=""

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/wow/splits \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 1 --tied \
        --save $models/drop/model_d1.0.pt --save_perp_log
)

echo "time taken:"
echo "$SECONDS seconds"
