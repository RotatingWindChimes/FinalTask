from Models.Resnet import Resnet_full
from DataProcess.Cifar import CIFAR100DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch

lr = 0.003
batch_size = 256
num_workers = 4
shuffle = True
epochs = 100
device = "cuda:0"
writer = SummaryWriter(log_dir="logs")


def main():
    # 加载数据集

    train_dataloader = CIFAR100DataLoader(split='train', batch_size=batch_size, num_workers=num_workers,
                                          shuffle=shuffle, size='224', normalize='standard')
    test_dataloader = CIFAR100DataLoader(split='test', batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                         size='224', normalize='standard')

    criterion1 = nn.CrossEntropyLoss().to(device)


    pretrain = Resnet_full("save/190_moco.pth").to(device)



    optimizer1 = optim.Adam(pretrain.parameters(), lr=lr)



    for epoch in range(epochs):
        pretrain_running_loss, pretrain_running_accuracy = train(pretrain, train_dataloader, criterion1, optimizer1,device)
        print(f"Epoch : {epoch + 1} - pretrain_acc: {pretrain_running_accuracy:.4f} - pretrain_loss : {pretrain_running_loss:.4f}\n")

        pretrain_test_loss, pretrain_test_accuracy = evaluation(pretrain, test_dataloader, criterion1, device)
        print(f"test pretrain_acc: {pretrain_test_accuracy:.4f} - test pretrain_loss : {pretrain_test_loss:.4f}\n")

        writer.add_scalars(main_tag='fullloss', tag_scalar_dict={'pretrain_train_loss': pretrain_running_loss,
                                                             'pretrain_test_loss': pretrain_test_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag='fullacc', tag_scalar_dict={'pretrain_train_acc': pretrain_running_accuracy,
                                                            'pretrain_test_acc': pretrain_test_accuracy},
                           global_step=epoch)
        torch.save(pretrain.state_dict(), "save/FullResnet" + str(epoch) + "_model_params.pkl")



def train(model, dataloader, criterion, optimizer, device):
    running_loss = 0.0
    running_accuracy = 0.0

    for data, target in tqdm(dataloader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == target).float().mean()
        running_accuracy += acc / len(dataloader)
        running_loss += loss.item() / len(dataloader)

    return running_loss, running_accuracy


def evaluation(model, dataloader, criterion,device):
    with torch.no_grad():
        test_accuracy = 0.0
        test_loss = 0.0
        for data, target in tqdm(dataloader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            acc = (output.argmax(dim=1) == target).float().mean()
            test_accuracy += acc / len(dataloader)
            test_loss += loss.item() / len(dataloader)

    return test_loss, test_accuracy


if __name__=="__main__":
    main()