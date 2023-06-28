# 测试三个训练结果的精度
from DataProcess.Cifar import CIFAR100DataLoader
from Models.Resnet import Resnet_pretrain,Resnet_normal,Resnet_full
import torch
from tqdm import tqdm

device = "cuda:0"


def evaluation(model, dataloader,device):
    with torch.no_grad():
        test_accuracy = 0.0
        for data, target in tqdm(dataloader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            acc = (output.argmax(dim=1) == target).float().mean()
            test_accuracy += acc / len(dataloader)

    return test_accuracy


def test():
    test_dataloader = CIFAR100DataLoader(split='test', batch_size=256, num_workers=4, shuffle=False,
                                         size='224', normalize='standard')
    pretrain = Resnet_pretrain("save/190_moco.pth").to(device)
    pretrain.load_state_dict(torch.load("save/pretrainResnet70_model_params.pkl"))
    full = Resnet_full("save/190_moco.pth").to(device)
    full.load_state_dict(torch.load("save/FullResnet12_model_params.pkl"))
    normal = Resnet_normal().to(device)
    normal.load_state_dict(torch.load("save/normalResnet56_model_params.pkl"))
    acc1 = evaluation(pretrain, test_dataloader,device)
    print("pretrain acc:",acc1)
    acc2 = evaluation(full, test_dataloader, device)
    print("Fulltrain acc:", acc2)
    acc3 = evaluation(normal, test_dataloader, device)
    print("normal acc:", acc3)




if __name__=="__main__":
    test()
