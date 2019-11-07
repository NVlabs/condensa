#!/usr/bin/env bash

if [[ $# -eq 0 ]]; then
  echo "Usage: compress_alexnet.sh [scheme] [density] [#iterations]"
  exit 1
fi

SCHEME=${1}
DENSITY=${2}
STEPS=${3}

PREFIX=alexnet_${SCHEME}_${DENSITY//[\.]/_}

python compress.py\
       --arch alexnet --dataset cifar10\
       --lr 0.01 --lr_end 1e-4\
       --weight_decay 0\
       --momentum 0.95\
       --mb_iterations_per_l 3000\
       --mb_iterations_first_l 30000\
       --mu_init 1e-3 --mu_multiplier 1.1\
       --l_batch_size 128\
       --model trained/alexnet.pth\
       --scheme ${SCHEME}\
       --density ${DENSITY}\
       --out compressed/${PREFIX}.pth\
       --csv results/${PREFIX}.csv\
       -v --steps ${STEPS}

