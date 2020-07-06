#!/bin/sh
module load pytorch/1.0.1
CUDA_VISIBLE_DEVICES=0 python eval.py --method MME --dataset office_home --source Art --target Clipart --net alexnet 
