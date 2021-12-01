#!/bin/bash
function output {
  eval ${cmd}
  RESULT=$?
  if [ $RESULT -eq 0 ]; then
    echo -e "\e[1;32m ${cmd} [Success] \e[0m"
  else
    echo -e "\e[1;31m ${cmd} [Failure] \e[0m"
    exit 1
  fi
}

rm -r ./LaMa_test_images

curl -L $(yadisk-direct https://disk.yandex.ru/d/xKQJZeVRk5vLlQ) -o LaMa_test_images.zip
unzip LaMa_test_images.zip

cmd="python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output"
output
