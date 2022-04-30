import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import resnet34
import time
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def Model(nb_classes, pretrained=False):
    # Pretrained Resnet 34
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000)
    if(pretrained):
        base_resnet34 = resnet34(pretrained=True)
        model.load_state_dict(base_resnet34.state_dict())
    model.fc = nn.Linear(512, nb_classes)

    return model
pretrain_epoch_list=np.linspace(10, 300, 30, endpoint=True)
pretrained_models_path = '../reference_code/BYOL-PyTorch-yaox12-/ckpt/byol_crohnIPI/275_first_saved'

epochs_num=700
foldnum=1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# test_dir = "/shareData3/lab-xing.jing/project/nantes/imgs_split_by_labels_2/fold1/"
# train_dir = "/shareData3/lab-xing.jing/project/nantes/imgs_split_by_labels_2/notfold1/"
test_dir = "/opt/data/home-jing/SSL_for_CD/dataset/val/fold1"
train_dir = "/opt/data/home-jing/SSL_for_CD/dataset/val/notfold1"
# test_dir = r"D:\Nantes_class\internship\experiment\SSL_for_CD\dataset\DataCrohnIPI\val\imgs_split_by_labels_2/fold1"
# train_dir = r"D:\Nantes_class\internship\experiment\SSL_for_CD\dataset\DataCrohnIPI\val\imgs_split_by_labels_2/notfold1"
batch_size = 64
log_path = './log/finetune'
lr = 0.001
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))



train_augs = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_augs = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


train_set = datasets.ImageFolder(train_dir, transform=train_augs)
test_set = datasets.ImageFolder(test_dir, transform=test_augs)
print('len train set/test_set',len(train_set),len(test_set))


train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_set, batch_size=batch_size)


def train(net, train_iter, test_iter, criterion, optimizer, num_epochs,pretrain_epoch_num):
    net = net.to(device)
    print("training on", device)
    best_accuracy=0.0
    for epoch in range(num_epochs):
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
        if epoch % 2 == 0:
            with torch.no_grad():
                net.eval()
                test_acc_sum, test_loss_sum, n2, batch_count2 = 0.0, 0.0, 0, 0
                for X, y in test_iter:
                    X, y = X.to(device), y.to(device)
                    y_hat = net(X)
                    loss = criterion(y_hat, y)
                    test_loss_sum+=loss.item()
                    test_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                    # test_acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                    n2 += y.shape[0]
                    batch_count2 += 1
            if best_accuracy < (test_acc_sum / n2):
                best_accuracy=(test_acc_sum / n2)
                torch.save(net.state_dict(), ckpt_path)

            writer.add_scalar(f'loss_pretrainepoch{pretrain_epoch_num}/val_loss', test_loss_sum / batch_count2, epoch)
            writer.add_scalar(f'accuracya_pretrinepoch{pretrain_epoch_num}/val_acc', test_acc_sum / n2, epoch)

        writer.add_scalar(f'loss_pretrinepoch{pretrain_epoch_num}/train_loss', train_loss_sum / batch_count, epoch)
        writer.add_scalar(f'accuracy_pretrinepoch{pretrain_epoch_num}/train_acc', train_acc_sum / n, epoch)

        print('epoch %d, train loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f, best acc %.3f, time %.1f sec'
              % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, test_loss_sum / batch_count2, test_acc_sum / n2, best_accuracy, time.time() - start))
    return  best_accuracy


baseline_net = Model(nb_classes=2, pretrained=True)


loss = torch.nn.CrossEntropyLoss()
for pretrain_epoch_num in pretrain_epoch_list:
    pretrain_epoch_num=str(int(pretrain_epoch_num))
    print('pretrained epoch:',pretrain_epoch_num)
    ckpt_path = os.path.join('ckpt','finetune', now,pretrain_epoch_num)
    log_path = os.path.join(log_path, now,pretrain_epoch_num)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ckpt_path = os.path.join(ckpt_path,'fold' + str(foldnum) + '.pth')

    writer = SummaryWriter(log_path)

    pretrained_model_path = pretrained_models_path+f"/resnet34_{pretrain_epoch_num}.pth.tar"
    if torch.cuda.is_available():
        pretrained_dict = torch.load(pretrained_model_path)
    else:
        pretrained_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))
    keys_pretrained = list(pretrained_dict['model'].keys())
    baseline_dict=baseline_net.state_dict()
    i = 0
    for k, v in baseline_dict.items():
        if v.size() == pretrained_dict['model'][keys_pretrained[i]].size():
            baseline_dict[k] = pretrained_dict['model'][keys_pretrained[i]]
        i = i + 1

    baseline_net.load_state_dict(baseline_dict)

    output_params = list(map(id, baseline_net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, baseline_net.parameters())
    optimizer = optim.SGD([{'params': feature_params},
                           {'params': baseline_net.fc.parameters(), 'lr': lr * 10}],
                          lr=lr, momentum=0.9)

    best_acc = train(baseline_net, train_iter, test_iter, loss, optimizer, num_epochs=epochs_num, pretrain_epoch_num=pretrain_epoch_num)
    writer.add_scalar('best_acc', best_acc, pretrain_epoch_num)
    print('best_acc', best_acc)


