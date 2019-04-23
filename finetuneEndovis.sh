#!/usr/bin/env bash
python mainEndovis.py --save-dir=/media/salman/DATA/save_finetuneEndovisOnRecon --batchSize 4 --lr 0.005 --epochs 200 --saveTest True |& tee -a /media/salman/DATA/save_finetuneEndovisOnRecon/log_Endovis
