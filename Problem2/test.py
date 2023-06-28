import torch
from model import  VisionTransformer
from dataloader import CIFAR100DataLoader
from train import evaluation

device = torch.device("cuda")
model=VisionTransformer(patch_size = 4,max_len = 100,layers=9,embed_dim = 512,classes = 100).to(device)
model.load_state_dict(torch.load("save/92_model_params.pkl"))
test_dataloader = CIFAR100DataLoader(split='test', batch_size=128, num_workers=2, shuffle=False, size='32', normalize='standard')
criterion = torch.nn.CrossEntropyLoss().to(device)
test_loss, test_accuracy = evaluation(model, test_dataloader, criterion,device,"ResNet")
print(f"test acc: {test_accuracy:.4f} - test loss : {test_loss:.4f}\n")