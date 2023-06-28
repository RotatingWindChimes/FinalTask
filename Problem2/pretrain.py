import torch
from torch import nn
from torch import optim
from transformers.models.auto.modeling_auto import AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup
from dataloader import CIFAR100DataLoader
from train import train,evaluation
from torch.utils.tensorboard import SummaryWriter

batch_size=128
epochs = 10
writer = SummaryWriter(log_dir="logs")

def main():
    # 加载数据集
    train_dataloader = CIFAR100DataLoader(split='train', batch_size=batch_size, num_workers=4, shuffle=True, size='224',
                                          normalize='standard')
    test_dataloader = CIFAR100DataLoader(split='test', batch_size=batch_size, num_workers=4, shuffle=False, size='224',
                                         normalize='standard')
    device = torch.device("cuda")
    model = AutoModelForImageClassification.from_pretrained(
        "vit-base-patch16-224-in21k",
        num_labels=100,
        ignore_mismatched_sizes=True,
        image_size=224,
    ).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=5000,
        num_warmup_steps=500,
    )
    criterion = nn.CrossEntropyLoss().to(device)



    for epoch in range(epochs):
        running_loss, running_accuracy = train(model, train_dataloader, criterion, optimizer, scheduler,device,"pretrain")
        print(f"Epoch : {epoch + 1} - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n")


        test_loss, test_accuracy = evaluation(model, test_dataloader, criterion,device,"pretrain")
        print(f"test acc: {test_accuracy:.4f} - test loss : {test_loss:.4f}\n")

        writer.add_scalars(main_tag='loss',
                           tag_scalar_dict={'train_loss': running_loss,
                                            'test_loss': test_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag='acc',
                           tag_scalar_dict={'train_acc': running_accuracy,
                                            'test_acc': test_accuracy},
                           global_step=epoch)



if __name__=="__main__":
    main()