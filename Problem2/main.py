import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from model import VisionTransformer
from dataloader import  CIFAR100DataLoader
from train import train,evaluation
device = torch.device("cuda")
from torch.utils.tensorboard import SummaryWriter
from ResNet import create_model



lr = 0.003
batch_size = 256
num_workers = 2
shuffle = True
patch_size = 4
image_sz = 32
max_len = 100 # All sequences must be less than 1000 including class token
embed_dim = 512
classes = 10
layers = 12
channels = 3
resnet_features_channels = 64
heads = 16
epochs = 100
writer = SummaryWriter(log_dir="logs")

def main():
    model=VisionTransformer(patch_size = 4,max_len = 100,layers=9,embed_dim = 512,classes = 100).to("cuda:0")
    ResNet_model = create_model().to(device)

    train_dataloader = CIFAR100DataLoader(split='train', batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, size='32', normalize='standard')
    test_dataloader = CIFAR100DataLoader(split='test', batch_size=batch_size, num_workers=num_workers, shuffle=False, size='32', normalize='standard')

    criterion = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer2 = optim.Adam(ResNet_model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=epochs)


    for epoch in range(epochs):

        running_loss, running_accuracy = train(model, train_dataloader, criterion, optimizer, None,device,"VIT")
        print(f"Epoch : {epoch+1} VIT - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n")
        ResNet_running_loss, ResNet_running_accuracy = train(ResNet_model, train_dataloader, criterion2, optimizer2,
                                                             None, device, "ResNet")
        print(
            f"Epoch : {epoch + 1} - ResNet_acc: {ResNet_running_accuracy:.4f} - ResNet_loss : {ResNet_running_loss:.4f}\n")

        test_loss, test_accuracy = evaluation(model, test_dataloader, criterion,device,"ResNet")
        print(f"test acc: {test_accuracy:.4f} VIT - test loss : {test_loss:.4f}\n")
        ResNet_test_loss, ResNet_test_accuracy = evaluation(ResNet_model, test_dataloader, criterion2, device, "ResNet")
        print(f"test ResNet_acc: {ResNet_test_accuracy:.4f} - test ResNet_loss : {ResNet_test_loss:.4f}\n")
        writer.add_scalars(main_tag='loss', tag_scalar_dict={'ViT_train_loss': running_loss,
                                                             'ViT_test_loss': test_loss,
                                                             'ResNet_train_loss': ResNet_running_loss,
                                                             'ResNet_test_loss': ResNet_test_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag='acc', tag_scalar_dict={'ViT_train_acc': running_accuracy,
                                                            'ViT_test_acc': test_accuracy,
                                                            'ResNet_train_acc': ResNet_running_accuracy,
                                                            'ResNet_test_acc': ResNet_test_accuracy},
                           global_step=epoch)
        torch.save(model.state_dict(), "save/" + str(epoch) + "_model_params.pkl")


if __name__=="__main__":
    main()