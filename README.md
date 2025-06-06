# [CVPR’25] PIVRG & ConsMTL

The official implementation of our CVPR 2025 papers:  
- [Revisiting Fairness in Multitask Learning: A Performance-Driven Approach for Variance Reduction](https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_Revisiting_Fairness_in_Multitask_Learning_A_Performance-Driven_Approach_for_Variance_CVPR_2025_paper.pdf)  
- [Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters](https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.pdf)  

The former highlights the strong positive correlation between cross-task performance variance and average benchmark performance in MTL methods.
The latter leverages the optimization of both shared and task-specific parameters to consistently alleviate gradient conflicts.

This repository also includes implementations of most prior MTL methods.

## Setup Environment
First, use miniconda to create a new environment:
```bash
conda create -n mtl python=3.9.7
conda activate mtl
```
Regarding the torch version in the environment, no stringent constraints are imposed; minor adjustments may be made based on the locally installed CUDA version.

Then, install the repository :
```bash
git clone https://github.com/jianke0604/MTLlib.git
cd MTLlib
pip install -r requirements.txt
```


## Download Datasets
The performance is evaluated under 3 scenarios:
 - Image-level Classification. The [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains 40 tasks.
 - Regression. The QM9 dataset contains 11 tasks, which can be downloaded automatically from Pytorch Geometric.
 - Dense Prediction. The [NYU-v2](https://github.com/lorenmt/mtan) dataset contains 3 tasks and the [Cityscapes](https://github.com/lorenmt/mtan) dataset contains 2 tasks.

## Run Experiments
To run the experiments, use the following command:
```bash
cd experiments/EXP_NAME
sh run.sh
```
For example, the `run.sh` script in `experiments/nyuv2` contains the following command:
```bash
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
```

## Hyperparameters
For **PIVRG**, the hyperparameter `bound` and `mintemp` have been set in `run.sh`.

For **ConsMTL**, the hyperparameter `lambda_` is set as follows:

| Benchmark         | $\lambda$ |
|------------------|-----------|
| NYUv2            | 1        |
| CityScapes and CelebA     | 50        |
| QM9              | 1e-3       |

Please note that for the **CityScapes** benchmark, we found that the task-specific parameters of the **depth estimation** task are highly sensitive to the additional gradients introduced by the method. Therefore, in our experiments, we only apply the additional gradients to the task-specific parameters of the **semantic segmentation** task. In the implementation, we modified the clipping magnitude in **line 1512** of `weight_methods.py` from `[0.1, 0.1]` to `[0.05, 0]`. For QM9, we add an additional hyperparameter `--lr_patience=8`.

## Important Note
There are no dominant solutions on the Pareto front, and without prior prioritization, it is not possible to directly compare two methods on the Pareto front. The two widely-used metrics ($\Delta m\%$ and MR) are merely proxies and cannot fully capture the performance of MTL methods. As commonly recognized, $\Delta m\%$ exhibits significant bias toward tasks with smaller baseline values. 

We emphasize that quantitative performance represents only one aspect of MTL method evaluation. Greater attention should be paid to the methodological innovations and technical contributions that advance the field's development.

## Acknowledgements
This codebase is built upon [FairGrad](https://github.com/OptMN-Lab/fairgrad). We sincerely thank the authors for their efforts and contributions.

## Citation
If you find this repository helpful, please consider citing our papers:
```bibtex
@inproceedings{qin2025towards,
  title={Towards Consistent Multi-Task Learning: Unlocking the Potential of Task-Specific Parameters},
  author={Qin, Xiaohan and Wang, Xiaoxing and Yan, Junchi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={10067--10076},
  year={2025}
},

@inproceedings{qin2025revisiting,
  title={Revisiting Fairness in Multitask Learning: A Performance-Driven Approach for Variance Reduction},
  author={Qin, Xiaohan and Wang, Xiaoxing and Yan, Junchi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20492--20501},
  year={2025}
}
```

We also recommend that you pay attention to the following important early works:
```bibtex
@inproceedings{ban2024fair,
  title={Fair resource allocation in multi-task learning},
  author={Ban, Hao and Ji, Kaiyi},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={2715--2731},
  year={2024}
},

@inproceedings{navon2022multi,
  title={Multi-Task Learning as a Bargaining Game},
  author={Navon, Aviv and Shamsian, Aviv and Achituve, Idan and Maron, Haggai and Kawaguchi, Kenji and Chechik, Gal and Fetaya, Ethan},
  booktitle={International Conference on Machine Learning},
  pages={16428--16446},
  year={2022},
  organization={PMLR}
},

@article{liu2023famo,
  title={Famo: Fast adaptive multitask optimization},
  author={Liu, Bo and Feng, Yihao and Stone, Peter and Liu, Qiang},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={57226--57243},
  year={2023}
}
```
