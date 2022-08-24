# -*- coding:utf-8 -*-
import torch
from torch.utils.data import random_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models.resnet import ResNet, BasicBlock,Bottleneck
from torchvision.models import resnet34,resnet18,resnet50
import time
import os
import random
import torch.nn.functional as F
from numpy import *
from torch.utils.tensorboard import SummaryWriter
def set_dataiter(args):
    # set data transform

    train_augs = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((180, 180)),
        transforms.Resize(args.image_size),
        transforms.ToTensor()
    ])
    test_augs = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor()
    ])
    test_dir = os.path.join(args.dataset_dir, f"fold{args.fold_num}")
    train_dir = os.path.join(args.dataset_dir, f"notfold{args.fold_num}")
    not_test_set = datasets.ImageFolder(train_dir, transform=test_augs)
    test_set = datasets.ImageFolder(test_dir, transform=test_augs)
    val_size = int(len(not_test_set) * args.val_proportion)
    #get 200 abnormal and 200 normal images from test_set

    train_size = len(not_test_set) - val_size
    train_set, val_set = random_split(
        dataset=not_test_set,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(0)
    )
    train_set.transforms = train_augs

    print('length of train, val, test set', len(train_set), len(val_set), len(test_set))

    train_iter = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    val_iter = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    return train_iter,val_iter,test_iter
def knn_dataiter(args,lenth):
    #use normal and abnormal imgs in fold1
    # use test transform
    test_augs = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor()
    ])
    test_dir = os.path.join(args.dataset_dir, "fold1")
    test_set = datasets.ImageFolder(test_dir, transform=test_augs)
    # get 200 abnormal and 200 normal images from test_set
    test_subABN = torch.utils.data.Subset(test_set, range(0, lenth*2))
    test_subN = torch.utils.data.Subset(test_set, range(len(test_set) - lenth*2, len(test_set)))
    #make sure the subset is right,ABN--0, N--1
    assert sum(test_subABN.dataset.targets[test_subABN.indices.start:test_subABN.indices.stop])==0
    assert sum(test_subN.dataset.targets[test_subN.indices.start:test_subN.indices.stop])==lenth*2
    knn_1N_set, knn_2N_set = torch.utils.data.random_split(test_subN,
                                                           [lenth, lenth],
                                                         )
    knn_1ABN_set, knn_2ABN_set = torch.utils.data.random_split(test_subABN,
                                                           [lenth, lenth],
                                                         )

    knn_1N_iter = DataLoader(knn_1N_set, batch_size=1, shuffle=False)
    knn_1ABN_iter = DataLoader(knn_1ABN_set, batch_size=1, shuffle=False)
    knn_2N_iter = DataLoader(knn_2N_set, batch_size=1, shuffle=False)
    knn_2ABN_iter = DataLoader(knn_2ABN_set, batch_size=1, shuffle=False)
    return knn_1N_iter, knn_1ABN_iter, knn_2N_iter, knn_2ABN_iter

def set_writers(args):
    '''set writers for logment'''
    log_path = os.path.join(args.log_path, 'fold'+args.fold_num, args.mode)
    train_log_path = os.path.join(log_path,args.time_stamp, 'train')
    val_log_path = os.path.join(log_path,args.time_stamp, 'val')
    test_log_path = os.path.join(log_path,args.time_stamp,'test')
    if not os.path.exists(val_log_path):
        os.makedirs(val_log_path)
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
    if not os.path.exists(test_log_path):
        os.makedirs(test_log_path)
    train_writer = SummaryWriter(train_log_path)
    val_writer = SummaryWriter(val_log_path)
    test_writer = SummaryWriter(test_log_path)

    return train_writer,val_writer,test_writer
def set_writers_linear_eval_epoch(args):
    '''set writers for logment'''
    log_path = args.log_path
    knn_log_path = os.path.join(log_path,args.time_stamp,'knn_dis')
    train_log_path = os.path.join(log_path,args.time_stamp, 'train')
    val_log_path = os.path.join(log_path,args.time_stamp, 'val')
    test_log_path = os.path.join(log_path,args.time_stamp,'test')
    if not os.path.exists(val_log_path):
        os.makedirs(val_log_path)
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
    if not os.path.exists(test_log_path):
        os.makedirs(test_log_path)
    if not os.path.exists(knn_log_path):
        os.makedirs(knn_log_path)
    train_writer = SummaryWriter(train_log_path)
    val_writer = SummaryWriter(val_log_path)
    test_writer = SummaryWriter(test_log_path)
    knn_val_writer=SummaryWriter(knn_log_path)

    return knn_val_writer,train_writer,val_writer,test_writer
def load_dic(net,pretrained_model_path,SSL_method):
    baseline_dict = net.state_dict()
    if torch.cuda.is_available():
        pretrained_dict = torch.load(pretrained_model_path)
    else:
        pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))


    if SSL_method=='MoCo':
        i=2# the first two elements of pretrained MoCo model is about the queue
        keys_pretrained=list(pretrained_dict['state_dict'])
        for k, v in baseline_dict.items():
            param=pretrained_dict['state_dict'][keys_pretrained[i]]
            if v.size() == param.size():
                baseline_dict[k] = param
            i = i + 1
        net.load_state_dict(baseline_dict)
    elif SSL_method=='BYOL':
        i=0
        keys_pretrained = list(pretrained_dict['model'].keys())
        for k, v in baseline_dict.items():
            if v.size() == pretrained_dict['model'][keys_pretrained[i]].size():
                baseline_dict[k] = pretrained_dict['model'][keys_pretrained[i]]
            i = i + 1
        net.load_state_dict(baseline_dict)
    elif SSL_method=='BarlowTwins':
        i = 0
        keys_pretrained = list(pretrained_dict.keys())
        for k, v in pretrained_dict.items():
            baseline_dict[k]=v
        net.load_state_dict(baseline_dict)

    return net
class Resnet_without_fc(nn.Module):

    def __init__(self, args):
        super(Resnet_without_fc, self).__init__()
        resnet_version=args.resnet_version
        if resnet_version == 'resnet34':
            model = ResNet(BasicBlock, [3, 4, 6, 3], 1000)
            base_resnet34 = resnet34(pretrained=True)
            model.load_state_dict(base_resnet34.state_dict())
        elif resnet_version == 'resnet18':
            model = ResNet(BasicBlock, [2, 2, 2, 2], 1000)
            base_resnet18 = resnet18(pretrained=True)
            model.load_state_dict(base_resnet18.state_dict())
        elif resnet_version == 'resnet50':
            model = ResNet(Bottleneck, [3, 4, 6, 3], 1000)
            base_resnet50 = resnet50(pretrained=True)
            model.load_state_dict(base_resnet50.state_dict())
        else:
            raise ValueError('wrong encoder name')
        print('loading pretrained args.model:', args.load_path)
        model = load_dic(model, args.load_path, args.SSL_method)
        print('finish load')
        self.encoder=torch.nn.Sequential( *( list(model.children())[:-1] ) )

    def forward(self, x):
        x = self.encoder(x)
        return x
def Model(nb_classes,args):
    # Pretrained Resnet 34
    mode=args.mode
    pretrained=args.pretrained_backbone
    resnet_version=args.resnet_version
    print(f'using {resnet_version}')

    if resnet_version=='resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], 1000)
        base_resnet34 = resnet34(pretrained=pretrained)
        model.load_state_dict(base_resnet34.state_dict())
    elif resnet_version=='resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], 1000)
        base_resnet18 = resnet18(pretrained=pretrained)
        model.load_state_dict(base_resnet18.state_dict())
    elif resnet_version=='resnet50':
        model=ResNet(Bottleneck,[3,4,6,3],1000)
        base_resnet50 = resnet50(pretrained=pretrained)
        model.load_state_dict(base_resnet50.state_dict())
    else:
        raise ValueError('wrong encoder name')
    if resnet_version!='resnet50':
        model.fc = nn.Linear(512, nb_classes)
    elif resnet_version=='resnet50':
        model.fc=nn.Linear(2048,nb_classes)

    if mode.startswith('finetune'):
        print('loading pretrained args.model:', args.load_path)
        model=load_dic(model, args.load_path,args.SSL_method)
        print('finish load')
    if not mode.endswith('unfixed'):
        for name, value in model.named_parameters():
            # print(name)
            # print(value.requires_grad)
            # print(value)
            if not name.startswith('fc'):
                # confient
                value.requires_grad = False
        print('backbone freezed')
    return model

def train(net,net_test, train_iter,val_iter, test_iter,
          criterion,optimizer, num_epochs,device,train_writer,val_writer,
          test_writer,ckpt_path,args):
    net = net.to(device)
    net_test.to(device)
    best_val_accuracy = 0.0
    count_write_images = 0
    best_test_accuracy=0
    for epoch in range(1,num_epochs+1):
        start = time.time()
        net.train()
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            if count_write_images==0:
                train_writer.add_images('train_transform',X)
                count_write_images+=1
            optimizer.zero_grad()
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        # validation per epoch
        if epoch % 1== 0:
            with torch.no_grad():
                net.eval()
                val_acc_sum, val_loss_sum, n2, batch_count2 = 0.0, 0.0, 0, 0
                for X, y in val_iter:
                    X, y = X.to(device), y.to(device)
                    if count_write_images == 1:
                        val_writer.add_images('val_transform', X)
                        count_write_images += 1
                    y_hat = net(X)
                    loss = criterion(y_hat, y)
                    val_loss_sum += loss.item()
                    val_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                    n2 += y.shape[0]
                    batch_count2 += 1
            if val_acc_sum / n2>=best_val_accuracy:
                torch.save(net.state_dict(), os.path.join(ckpt_path,'current_best.pth'))
                best_val_accuracy = val_acc_sum / n2
            val_writer.add_scalar('loss', val_loss_sum / batch_count2, epoch)
            val_writer.add_scalar('accuracy', val_acc_sum / n2, epoch)
        train_writer.add_scalar('loss', train_loss_sum / batch_count, epoch)
        train_writer.add_scalar('accuracy', train_acc_sum / n, epoch)

        print('fold: %s, epoch %d, train loss %.4f, train acc %.3f, val loss %.4f, val_acc %.3f, best val acc %.3f, time %.1f sec'
              % (args.fold_num, epoch, train_loss_sum / batch_count, train_acc_sum / n, val_loss_sum / batch_count2,
                 val_acc_sum / n2,
                 best_val_accuracy, time.time() - start))
        #test per 20 epoch
        if epoch % args.test_interval == 0:
            print(f'test for training {epoch}epoch')
            test_path=os.path.join(ckpt_path,'current_best.pth')
            # val_path = os.path.join(ckpt_path, 'current_best.pth')
            # test_path=os.path.join(ckpt_path,f'epoch{str(epoch)}.pth')
            # os.system(f'cp {val_path} {test_path}')
            with torch.no_grad():
                net_test.load_state_dict(torch.load(test_path))
                net_test.eval()
                test_acc_sum, test_loss_sum, n3, batch_count3 = 0.0, 0.0, 0, 0
                for X, y in test_iter:
                    X, y = X.to(device), y.to(device)

                    y_hat = net_test(X)
                    loss = criterion(y_hat, y)
                    test_loss_sum += loss.item()
                    test_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                    # test_acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                    n3 += y.shape[0]
                    batch_count3 += 1

            test_writer.add_scalar('loss', test_loss_sum / batch_count3, epoch)
            test_writer.add_scalar('accuracy', test_acc_sum / n3, epoch)
            print('test loss,acc:',test_loss_sum / batch_count3,test_acc_sum / n3)
    return test_acc_sum / n3
def train_linear_eval(net,net_test, train_iter,val_iter, test_iter,
          criterion,optimizer, num_epochs,device
          ,ckpt_path,args):
    net = net.to(device)
    net_test.to(device)
    best_val_accuracy = 0.0


    for epoch in range(1,args.linearval_epochs+1):
        start = time.time()
        net.train()
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        # validation per epoch
        if epoch % 1== 0:
            with torch.no_grad():
                net.eval()
                val_acc_sum, val_loss_sum, n2, batch_count2 = 0.0, 0.0, 0, 0
                for X, y in val_iter:
                    X, y = X.to(device), y.to(device)

                    y_hat = net(X)
                    loss = criterion(y_hat, y)
                    val_loss_sum += loss.item()
                    val_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                    n2 += y.shape[0]
                    batch_count2 += 1
            if val_acc_sum / n2>=best_val_accuracy:
                torch.save(net.state_dict(), os.path.join(ckpt_path,'current.pth'))
                best_val_accuracy = val_acc_sum / n2
        print('fold: %s, epoch %d, train loss %.4f, train acc %.3f, val loss %.4f, val_acc %.3f, best val acc %.3f, time %.1f sec'
              % (args.fold_num, epoch, train_loss_sum / batch_count, train_acc_sum / n, val_loss_sum / batch_count2,
                 val_acc_sum / n2,
                 best_val_accuracy, time.time() - start))
        #test per 20 epoch
        if epoch % args.linearval_epochs == 0:
            print(f'test for training {epoch}epoch')
            test_path=os.path.join(ckpt_path,'current.pth')
            # val_path = os.path.join(ckpt_path, 'current_best.pth')
            # test_path=os.path.join(ckpt_path,f'epoch{str(epoch)}.pth')
            # os.system(f'cp {val_path} {test_path}')
            with torch.no_grad():
                net_test.load_state_dict(torch.load(test_path))
                net_test.eval()
                test_acc_sum, test_loss_sum, n3, batch_count3 = 0.0, 0.0, 0, 0
                for X, y in test_iter:
                    X, y = X.to(device), y.to(device)

                    y_hat = net_test(X)
                    loss = criterion(y_hat, y)
                    test_loss_sum += loss.item()
                    test_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                    # test_acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                    n3 += y.shape[0]
                    batch_count3 += 1
            print('test loss,acc:',test_loss_sum / batch_count3,test_acc_sum / n3)
            return test_acc_sum / n3,best_val_accuracy,train_acc_sum / n
def knn_eval(net_knn,device,knn_1N_iter,knn_1ABN_iter,
             knn_2N_iter,knn_2ABN_iter,knn_val_writer,args,pretrained_epoch):

    net_knn.to(device)
    with torch.no_grad():
        net_knn.eval()
        knn_1N_features=[]
        knn_1ABN_features=[]
        knn_2N_features=[]
        knn_2ABN_features=[]

        for X, y in knn_1N_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net_knn(X)
            knn_1N_features.append(y_hat)
        for X, y in knn_1ABN_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net_knn(X)
            knn_1ABN_features.append(y_hat)
        for X, y in knn_2N_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net_knn(X)
            knn_2N_features.append(y_hat)
        for X, y in knn_2ABN_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net_knn(X)
            knn_2ABN_features.append(y_hat)
    knn_dis=0.5*(cal_knn_dis(knn_1N_features,knn_2N_features)+cal_knn_dis(knn_1ABN_features,knn_2ABN_features))-\
            0.25*(cal_knn_dis(knn_1N_features,knn_1ABN_features)+cal_knn_dis(knn_1N_features,knn_2ABN_features)+\
                  cal_knn_dis(knn_2N_features,knn_1ABN_features)+cal_knn_dis(knn_2N_features,knn_2ABN_features))
    return knn_dis
def knn_dis(preds, targets):
    preds_norm = F.normalize(preds.squeeze(), dim=0)
    targets_norm = F.normalize(targets.squeeze(), dim=0)
    loss = 2 - 2 * (preds_norm * targets_norm).sum()
    return loss
def cal_knn_dis(m,n):
    dis_list=[]
    for i in range(len(m)):
        temp_list = []
        for j in range(len(n)):
            temp_list.append(knn_dis(m[i],n[j]))
        dis_list.append(min(temp_list))
    return torch.mean(torch.stack(dis_list))

