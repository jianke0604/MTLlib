mkdir -p ./save
mkdir -p ./trainlogs


export PYTHONPATH=$PYTHONPATH:/mnt/nas-new/home/qinxiaohan/MTLlib
export CUDA_VISIBLE_DEVICES=0

method=pivrg
seed=0
bound=2
mintemp=10

python -u trainer.py \
 --method=$method \
 --seed=$seed \
 --bound=$bound \
 --mintemp=$mintemp \
