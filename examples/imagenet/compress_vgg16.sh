#!/bin/bash

if [[ $# -eq 0 ]]; then
  echo "Usage: vgg16.sh [scheme] [density] [#iterations]"
  exit 1
fi

SCHEME=${1}
DENSITY=${2}
STEPS=${3}
PREFIX=vgg16_${SCHEME}_node${HOSTNAME}_den${DENSITY//[\.]/_}

python compress.py --arch vgg16_bn\
       --lr 0.1 --lr_end 1e-5\
       --weight_decay 0\
       --momentum 0.9\
       --sgd_iterations_per_l 5005\
       --mu_init 1e-3 --mu_multiplier 1.1\
       --l_batch_size 256\
       --val_batch_size 256\
       --sgd_iterations_first_l 30030\
       --scheme ${SCHEME}\
       --density ${DENSITY}\
       --out compressed/${PREFIX}.pth\
       --csv results/${PREFIX}.csv\
       -v --steps ${STEPS}\
       /tmp/imagenet
