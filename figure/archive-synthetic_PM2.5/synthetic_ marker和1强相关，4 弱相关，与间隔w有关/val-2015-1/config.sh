#python ./data/PM2.5-multiclass/data_processing.py 2015
#

rm -rf "./figure/val-2015/"
mkdir "./figure/val-2015/"
mkdir "./figure/val-2015/training/"
mkdir "./figure/val-2015/validation/"

device=0
data=data/synthetic_/
# data=data/PM2.5-multiclass/
# data=data/data_so/fold1/
# data=data/PM2.5-multiclass/
batch=16
n_head=4
n_layers=1
d_model=256
d_rnn=64
d_inner=256
d_k=256
d_v=256

dropout=0
lr=1e-3
smooth=0
epoch=200
log=log.txt

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch -log $log
