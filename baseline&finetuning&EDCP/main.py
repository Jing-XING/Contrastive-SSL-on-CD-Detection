import torch
import argparse
from torch import optim
import time
import os
from numpy import *
from utils import train,Model,set_writers,set_dataiter
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='finetune_unfixed', type=str, help='mode of the train(baseline_fixed/baseline_unfixed/finetune_fixed/finetune_unfixed')
parser.add_argument('--cuda_num',default='10',type=str,help='number of used gpu')
parser.add_argument('--pretrained_backbone',default=True,type=bool,help='whether use pretrained backbone')
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Initial learning rate.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training.")
parser.add_argument("--num_epochs", default=50, type=int, help="Number of epochs to train for.")
# parser.add_argument("--checkpoint_epochs", default=10, type=int, help="Number of epochs between checkpoints/summaries.")
parser.add_argument("--dataset_dir", default="/shareData3/lab-xing.jing/dataset/CrohnIPI/imgs_split_by_folds/", type=str, help="Directory where dataset is stored.")
parser.add_argument('--val_proportion',default=0.2,type=float,help='proportion of training set for validation')
parser.add_argument("--num_workers", default=4, type=int, help="Number of data loading workers")
parser.add_argument('--load_path', default="", type=str, help='path of pretrained backbone args.model')
parser.add_argument('--log_path', default='./log_finetune_barlowtwins', type=str, help='path of logments')
parser.add_argument('--ckpt_path', default='./ckpt_finetune_barlowtwins', type=str, help='path of model saving')
parser.add_argument("--image_size", default=256, type=int, help="Image size")
parser.add_argument("--resnet_version", default="resnet34", type=str, help="ResNet version.")
parser.add_argument("--Kfold_num", default=5, type=int, help="number of k-fold cross validation")
parser.add_argument("--time_stamp", default=now, type=str, help="time stamp for log and model saving")
parser.add_argument("--test_interval", default=50, type=int, help="number of k-fold cross validation")
parser.add_argument('--fold_num', default='1', type=str, help='fold number for training on crohnipi dataset')
parser.add_argument('--SSL_method',default='BarlowTwins',type=str,help='the methods of pretrain')
parser.add_argument('--rounds',default=1,type=int,help='the number of runds of baseline running')
args = parser.parse_args()
loss = torch.nn.CrossEntropyLoss()
#set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_acc_round=[]
log_path=args.log_path
ckpt_path=args.ckpt_path
for round in range(args.rounds):
    print(f'round{round}')
    args.log_path=os.path.join(log_path,f'round{round}')
    args.ckpt_path=os.path.join(ckpt_path,f'round{round}')
    test_acc_fold = []
    for fold in range(1,6):

        args.fold_num=str(fold)
        for arg in vars(args):
            print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))   # str, arg_type

        train_writer,val_writer,test_writer=set_writers(args)
        args_dic = vars(args)
        for key in args_dic:
            train_writer.add_text(key,str(args_dic[key]))

        #set dataset and net
        net = Model(nb_classes=2, args=args)
        net_test = Model(nb_classes=2, args=args)
        train_iter, val_iter,test_iter=set_dataiter(args)
        ckpt_path = os.path.join(args.ckpt_path,'fold'+args.fold_num, args.mode,args.time_stamp)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

        test_acc=train(net=net,net_test=net_test, train_iter=train_iter,val_iter=val_iter,test_iter=test_iter, criterion=loss, optimizer=optimizer, num_epochs=args.num_epochs,device=device,
              train_writer=train_writer,val_writer=val_writer,test_writer=test_writer,ckpt_path=ckpt_path,args=args)
        test_acc_fold.append(test_acc)
    print(f'test acc for round{round}:',mean(test_acc_fold))
    test_acc_round.append(mean(test_acc_fold))
print(f'five rounds mean test acc{mean(test_acc_round)}')
