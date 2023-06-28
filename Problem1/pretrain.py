from  torchvision.models import resnet18
from DataProcess.ImageNet import build_dataloader
from tqdm import tqdm
import torch
from  Models.moco import MoCo
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from time import time

class Config:
    root = "./data/ImageNet/ILSVRC2012_img_val"
    batch_size = 4
    device = "cuda:0"
    moco_dim = 128
    moco_k = 65536
    moco_m = 0.999
    moco_T = 0.07

    lr = 0.03
    momentum = 0.9
    weight_decay = 1e-4
    total_epoch = 200


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    losses = []
    accs = []
    for images in tqdm(train_loader):
        images[0] = images[0].to(device)
        images[1] = images[1].to(device)
        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == target).float().mean()
        losses.append(loss.item())
        accs.append(acc.item())
    print("loss:",sum(losses)/len(losses))
    print("acc:",sum(accs)/len(accs))
    if(epoch%10==0):
        torch.save(model.encoder_q.state_dict(),"./save/"+str(epoch)+"_moco.pth")
    return sum(losses)/len(losses),sum(accs)/len(accs)



def main():
    # 加载ImageNet数据集
    config = Config()
    train_dataloader = build_dataloader(config)

    # 加载模型
    model = MoCo(
        resnet18,
        config.moco_dim,
        config.moco_k,
        config.moco_m,
        config.moco_T,
    ).to(config.device)
    criterion = nn.CrossEntropyLoss().to(config.device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        config.lr,
        momentum = config.momentum,
        weight_decay = config.weight_decay,
    )

    writer = SummaryWriter('logs')

    for epoch in range(config.total_epoch):
        print("Epoch:",epoch+1)
        # train for one epoch
        loss,acc = train(train_dataloader, model, criterion, optimizer, epoch, config.device)
        writer.add_scalar("pretrainLoss",loss,epoch)
        writer.add_scalar("pretrainacc", acc, epoch)
    writer.close()

if __name__=="__main__":
    main()