'''
this file is for evaluating the encoder with different epoch of  pretraining
the evaluation will be :
1. few shot(default 1 epoch) linear evaluation
2. knn similar distance
'''

import torch
import argparse
from torch import optim
import time
import os
import numpy as np
from utils import train_linear_eval,Model,set_writers_linear_eval_epoch,set_dataiter,knn_eval,Resnet_without_fc,knn_dataiter
from torch.utils.tensorboard import SummaryWriter
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
parser = argparse.ArgumentParser()
parser.add_argument('--models_tobe_evaluated',default='/shareData3/lab-xing.jing/project/nantes/ssl_crohn/saved_model/resnet18_epoch600_1viewPerTime_acc1_pretrained_true/',type=str,help='the folder of pretrained models to be evaluated')
parser.add_argument('--mode', default='finetune_unfixed', type=str, help='mode of the train(baseline_fixed/baseline_unfixed/finetune_fixed/finetune_unfixed')
parser.add_argument('--cuda_num',default='11',type=str,help='number of used gpu')
parser.add_argument('--pretrained_backbone',default=True,type=bool,help='whether use pretrained backbone')
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Initial learning rate.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training.")
parser.add_argument("--num_epochs", default=1, type=int, help="Number of epochs to train for.")
# parser.add_argument("--checkpoint_epochs", default=10, type=int, help="Number of epochs between checkpoints/summaries.")
parser.add_argument("--dataset_dir", default="/shareData3/lab-xing.jing/dataset/CrohnIPI/imgs_split_by_folds/", type=str, help="Directory where dataset is stored.")
parser.add_argument('--val_proportion',default=0.2,type=float,help='proportion of training set for validation')
parser.add_argument("--num_workers", default=4, type=int, help="Number of data loading workers")
parser.add_argument('--load_path', default="/shareData3/lab-xing.jing/project/nantes/ssl_crohn/reference_code/moco/ckpt/bs96_epoch300/checkpoint_0300.pth.tar", type=str, help='path of pretrained backbone args.model')
parser.add_argument('--log_path', default='./log_evlauate_pretrained_epoch', type=str, help='path of logments')
parser.add_argument('--ckpt_path', default='./ckpt_evaluate_pretrained_epoch', type=str, help='path of model saving')
parser.add_argument("--image_size", default=256, type=int, help="Image size")
parser.add_argument("--resnet_version", default="resnet18", type=str, help="ResNet version.")
parser.add_argument("--Kfold_num", default=5, type=int, help="number of k-fold cross validation")
parser.add_argument("--time_stamp", default=now, type=str, help="time stamp for log and model saving")
parser.add_argument("--test_interval", default=10, type=int, help="number of k-fold cross validation")
parser.add_argument('--fold_num', default='1', type=str, help='fold number for training on crohnipi dataset')
parser.add_argument('--SSL_method',default='BYOL',type=str,help='the methods of pretrain')
parser.add_argument('--linearval_epochs',default=1,type=int,help='how many epochs for linear evaluation training')
args = parser.parse_args()
for arg in vars(args):
    print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))  # str, arg_type
loss = torch.nn.CrossEntropyLoss()

#set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


knn_val_writer,train_writer,val_writer,test_writer=set_writers_linear_eval_epoch(args)
epoch_list=np.arange(20,601,20)
for e in epoch_list:
    model=f'resnet18_{e}.pth.tar'
    model_path=os.path.join(args.models_tobe_evaluated,model)
    args.load_path=model_path
    pretrained_epoch=int(model.split('_')[-1].split('.')[0])
    #knn evaluation
    knn_time_start=time.time()

    net_knn = Resnet_without_fc(args)
    knn_1N_iter, knn_1ABN_iter, knn_2N_iter, knn_2ABN_iter = knn_dataiter(args,lenth=100)
    knn_dis=knn_eval(net_knn=net_knn, device=device, knn_1N_iter=knn_1N_iter,
             knn_1ABN_iter=knn_1ABN_iter, knn_2N_iter=knn_2N_iter,
             knn_2ABN_iter=knn_2ABN_iter, knn_val_writer=knn_val_writer, args=args,
             pretrained_epoch=pretrained_epoch)
    print(f'time for knn-evaluaton on pretrained epoch{pretrained_epoch}:',time.time()-knn_time_start)

    test_acc_avg=0
    train_acc_avg=0
    val_acc_avg=0
    log_path = os.path.join(args.log_path, 'fold' + args.fold_num, args.mode)
    for fold in range(1,6):
        args.fold_num=str(fold)
        #set dataset and net
        net = Model(nb_classes=2, args=args)
        net_test = Model(nb_classes=2, args=args)
        train_iter, val_iter,test_iter=set_dataiter(args)
        ckpt_path = os.path.join(args.ckpt_path,'fold'+args.fold_num, args.mode,args.time_stamp)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        test_acc,val_acc,train_acc=train_linear_eval(net=net,net_test=net_test, train_iter=train_iter,val_iter=val_iter,test_iter=test_iter,
                                                   criterion=loss, optimizer=optimizer, num_epochs=args.num_epochs,device=device,ckpt_path=ckpt_path,args=args)
        test_acc_avg+=test_acc
        val_acc_avg+=val_acc
        train_acc_avg+=train_acc
    train_writer.add_scalar('accuracy', train_acc_avg/5, pretrained_epoch)
    val_writer.add_scalar('accuracy', val_acc_avg/5, pretrained_epoch)
    test_writer.add_scalar('accuracy', test_acc_avg/5, pretrained_epoch)
    knn_val_writer.add_scalar('knn_distance', knn_dis,pretrained_epoch)


