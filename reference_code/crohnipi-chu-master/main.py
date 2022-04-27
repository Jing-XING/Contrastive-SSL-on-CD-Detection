import torch
import numpy as np
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import torch.nn.functional as F
from data_loader import get_vid_loader
from torch.autograd import Variable
import json
import argparse

##  debug tool
DEBUG=False
def debug(s):
    if DEBUG:
        print("DEBUG : " + s)

def str2bool(v):
    return v.lower() in ('true', '1')


## create the RESNET model
# if pretrained == True, the weigths are downloaded 
# head is cuted to nb_classes
def Model(nb_classes, pretrained=False):
    # Pretrained Resnet 34
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000)
    if(pretrained):
        base_resnet34 = resnet34(pretrained=True)
        model.load_state_dict(base_resnet34.state_dict())
    model.fc = nn.Linear(512, nb_classes)

    return model


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('--batch_size', type=int, default=16,
                     help='Number of image per batch')
    arg.add_argument('--vid_path', type=str, default='../Data/full_video/E170C0F8-BE7D-4412-9435-DC893594496E',
                     help='Path of the folder containing the images')
    arg.add_argument('--weights_path', type=str, default='./ResNet34_Adam_3annotators_Crohn2020_0_model_best.pth.tar',
                     help='Path of the weights to load the model')
    arg.add_argument('--output_json', type=str, default='output.json',
                     help='Name of the output json file')
    arg.add_argument('--use_gpu', type=str2bool, default=True,
                     help='Whether to use GPU for computation')
    arg.add_argument('--advencement', type=str, default='advencement_tmp.txt', help='Name of the advencement file')
    arg.add_argument('--debug', type=str2bool, default=False,
                     help='print debug info')
    args = vars(arg.parse_args())

    global DEBUG
    DEBUG =args['debug']
    return args['batch_size'], args['vid_path'], args['output_json'], args['weights_path'], args['use_gpu'], args['advencement']


## record the advencement of the process
def advencement(f,i, total):
    with open(f, 'w') as out:
            out.write(str(i) + '/' + str(total) + '\n')

def main(batch_size, vid_path, output_json, weights_path, use_gpu,adv_file):
    # init advencement
    advencement(adv_file, 0,100)
    #build model
    model = Model(nb_classes=2)
    #load weigths 
    if not use_gpu:
        ckpt = torch.load(weights_path, map_location=lambda storage, loc: storage)
    else:
        ckpt = torch.load(weights_path)
        model.cuda()
    model.load_state_dict(ckpt['model_state'])
    # eval mode, no grad 
    model.eval()

    # init dataloader
    testloaders = get_vid_loader(data_dir=vid_path,
                                 batch_size=batch_size)
    all_pred = []
    size_vid = len(testloaders)
    # re-init advencement knowing the complete size
    advencement(adv_file, 0,size_vid)
    # batch loop
    for i, (x, y, path) in enumerate(testloaders):
        debug("batch " + str(i) + " ... \n " )
        with torch.no_grad():
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            # compute prediction
            pred = F.softmax(model(x), dim=1)
            # get ouputs
            score = pred.cpu().detach().numpy()[:, 1]
            predicted = torch.max(pred, 1)[1].cpu().detach().numpy()
            # save results in all_pred
            for i, el in enumerate(score):
                batch_pred = {'im_path': path[i], 'score': str(el), 'prediction': str(predicted[i])}
                debug(str(i) + " = " + str(batch_pred))
                all_pred.append(batch_pred)
        # update advencement
        advencement(adv_file, i,size_vid)
    # write all results
    with open(output_json, 'w') as f:
        json.dump(all_pred, f)

    advencement(adv_file, size_vid,size_vid)


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
