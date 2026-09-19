# CONTRASTIVE SELF SUPERVISED LEARNING ON CROHN'S DISEASE DETECTION

![image](https://github.com/Jing-XING/Contrastive-SSL-on-CD-Detection/blob/master/paper/images/method.png)
### Abstract
Crohn’s disease is a type of inflammatory bowel
illness that is typically identified via computer-aided diagnosis
(CAD), which employs images from wireless capsule endoscopy
(WCE). While deep learning has recently made significant advancements in Crohn’s disease detection, its performance is still
constrained by limited labeled data. We suggest using contrastive
self-supervised learning methods to address these difficulties
which was barely used in detection of Crohn’s disease. Besides,
we discovered that, unlike supervised learning, it is difficult to
monitor contrastive self-supervised pretraining process in real
time. So we propose a method for evaluating the model during
contrastive pretraining (EDCP) based on the Euclidean distance
of the sample representation, so that the model can be monitored
during pretraining. Our comprehensive experiment results show
that with contrastive self-supervised learning, better results in
Crohn’s disease detection can be obtained. EDCP has also been
shown to reflect the model’s training progress. Furthermore, we
discovered some intriguing issues with using contrastive selfsupervised learning for small dataset tasks in our experiments
that merit further investigation.


## Usage
### Baseline and finetuning 

The code for baseline and finetuning are in ./baseline&finetuning&EDCP/main.py
```
python main.py --mode=baseline_fixed --SSL_method=BYOL --resnet_version=resnet34 --dataset_dir=/path/to/folds --pretrained_backbone=False
```

Run these commands from `baseline&finetuning&EDCP`. Modes use `baseline` or `finetune` followed by `_fixed` or `_unfixed`. `--load_path` is the input pretrained checkpoint; `--ckpt_path` is the output checkpoint directory. `--SSL_method` selects `BYOL`, `MoCo` or `BarlowTwins`. Fine-tuning requires a compatible checkpoint through `--load_path`.


### Evaluation during contrastive pretraining

![image](https://github.com/Jing-XING/Contrastive-SSL-on-CD-Detection/blob/master/paper/images/edcp.png)
The evaluation tool can be run by:

```
python evaluate_pretrain.py --models_tobe_evaluated=/path/to/checkpoints --dataset_dir=/path/to/folds
```

to check the EDCP protocal's performance.

### Pretrain

![image](https://github.com/Jing-XING/Contrastive-SSL-on-CD-Detection/blob/master/paper/images/frameworks.png)


Pretraining sources are under `contrastive_self_supervised_learning`. Their entry points differ: BYOL uses `byol_main.py` with `config/train_config.yaml`; Barlow Twins uses `main.py`; MoCo uses `main_moco.py`. Review each implementation's dataset paths, encoder, checkpoint settings and GPU/distributed setup before running. Defaults differ across these implementations.



## PAPER
[Contrastive Self-Supervised Learning on Crohn’s
Disease Detection](https://ieeexplore.ieee.org/document/9995504).


## Organized local copy

The baseline and EDCP command help can be read without installing PyTorch:

```bash
python "baseline&finetuning&EDCP/main.py" --help
python "baseline&finetuning&EDCP/evaluate_pretrain.py" --help
```

Importing these two entry points no longer starts training. `--pretrained_backbone False` is parsed as false; the historical boolean default remains true. Dataset location defaults to `CROHN_DATA_ROOT` when set, otherwise `./data/CrohnIPI/imgs_split_by_folds`, and `--dataset_dir` overrides it. The expected layout is `fold1`…`fold5` and `notfold1`…`notfold5`, with ImageFolder class subdirectories.

See [organization and runtime notes](ORGANIZATION.md) and [verification scope](CHECKS.md). The original research abstract above is retained; this organization pass does not independently reproduce its reported results.
