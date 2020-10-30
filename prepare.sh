#!/bin/bash

arr=(25000 4019 5849)

for value in ${arr[@]}
do
  echo $value
  python -u mbrin_process.py --task $value 2>&1 1>results/prepare/$value.preout

done

