mkdir -p ./save
mkdir -p ./trainlogs

export PYTHONPATH=$PYTHONPATH:YOUR_PYTHON_PATH
export CUDA_VISIBLE_DEVICES=0

method=pivrg
seed=0
bound=2
mintemp=10


python -u trainer.py \
 --method=$method \
 --seed=$seed  \
 --bound=$bound \
 --mintemp=$mintemp \
 --wandb_logger_name "XXX" \
 --wandb_project=XXX \
 --wandb_entity=XXX
