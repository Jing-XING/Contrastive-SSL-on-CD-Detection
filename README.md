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
python main.py --mode='baseline_fixed' --SSl_method='BYOL' --resnet_version='resnet34' --ckpy_path=' '
```

mode can be started with 'baseline' or 'finetuning', the following 'fixed' or 'unfixed' determines to fix the parameters of backbone(resnet) or not. 'ckpt_path' is the path of the pretrained models. SSL_mode('BYOL','MoCo' or 'BarlowTwins' is the contrastive self-supervised methods used for pretraining.


### Evaluation during contrastive pretraining

![image](https://github.com/Jing-XING/Contrastive-SSL-on-CD-Detection/blob/master/paper/images/edcp.png)
The evaluation tool can be run by:

```
python evaluate_pretrain.py --models_tobe_evaluated=' '
```

to check the EDCP protocal's performance.

### Pretrain

![image](https://github.com/Jing-XING/Contrastive-SSL-on-CD-Detection/blob/master/paper/images/frameworks.png)


The pretraining codes for BYOL Barlow Twins and MoCo are in ./contrastive_self_supervised_learning. Each pretraining can be run by
```
python main.py
```
The default encoder is Resnet34 and the pretrained model will be saved in the corresponding ckpt folder.



## PAPER
[Contrastive Self-Supervised Learning on Crohn’s
Disease Detection](https://ieeexplore.ieee.org/document/9995504).
