from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import clip
import argparse
import PIL
import timm
import time
from dataset.RSNA_dataset import RSNADataset
from models.model import ImageNetModel, CLIPModel, ViTModel, CheXpertModel

import medmnist
from medmnist import INFO, Evaluator
from transformers import AutoImageProcessor, ViTForImageClassification
import torchvision.models as models
from sklearn.metrics import f1_score, roc_auc_score
import sys
import random
from timm.models import vit_base_patch16_224
from torchvision import datasets, transforms
import torchxrayvision as xrv

class GeoModel(nn.Module):
    def __init__(self):
        super(GeoModel, self).__init__()
        self.backbone  = getattr(models, "resnet18")(pretrained=True)
        self.backbone.conv1 = torch.nn.Conv2d(3, 64, 3,stride=1, padding=1, bias=False)
        self.backbone.maxpool = torch.nn.Identity()
    def forward(self, x):
        x = self.backbone(x)


        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', default='vit_b_16', type=str, help='pretrained model')
    parser.add_argument('--task', default='binary-class', type=str, help='available things {multi-label, multi-class, binary-class}')
    parser.add_argument('--csv', default='/home/santosh.sanjeev/model-soups/my_soups/metadata/RSNA_final_df.csv', type=str, help='Data directory')
    parser.add_argument('--data_dir', default='/home/santosh.sanjeev/rsna_18/train/', type=str, help='csv file containing the stats')
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--initialisation', default='imagenet', type=str, help='weight initialisation')
    parser.add_argument('--dataset', default='pneumoniamnist', type=str, help='which dataset')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--lp_ft', default='LP', type=str, help='which type of finetuning')
    
    parser.add_argument('--device', default='cuda', type=str, help='which device')
    parser.add_argument('--norm', default=0.5, type=float, help='which norm')

    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--output_model_path', default='/home/santosh.sanjeev/model-soups/my_soups/checkpoints/full_finetuning/pneumoniamnist/initial_full_finetuning_model.pth', type=str, help='model path')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set seed for PyTorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    if args.dataset[-5:] == 'mnist':
        data_flag = args.dataset
        download = True
        info = INFO[data_flag]
        task = info['task']            
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        print(data_flag, info, task, n_channels, n_classes)
    else:
        n_classes = args.n_classes
    NUM_EPOCHS = args.n_epochs
    BATCH_SIZE = args.batch_size
    lr = args.lr
    load_model = args.model
    print(args.use_pretrained)
    
    if args.norm!=0.5:
        print('USING IMAGENET NORM')
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        print('NOT USING IMAGENET NORM')
        # mean = [0.5,]
        # std = [0.5,]
        mean, std = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        normalize = transforms.Normalize(mean=mean, std=std)

    # preprocessing
    data_transform = transforms.Compose([
        # transforms.RandomRotation(degrees=15, fill=(0,0,0)),  # Random rotation with 15 degrees
        # transforms.RandomVerticalFlip(p=0.3),  # Random vertical flip with 50% probability
        # transforms.RandomHorizontalFlip(p=0.3),  # Random horizontal flip with 50% probability
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        print('CIFARRRRRRRRRRRRRRRRR')
        # # Define transformations to be applied to the images
        # transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        #     transforms.RandomCrop(32, padding=4),  # Randomly crop the image
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
        # ])

        # Load CIFAR-10 dataset
        train_dataset = datasets.CIFAR100(root='./data', train=True, transform=data_transform, download=True)
        test_dataset = datasets.CIFAR100(root='./data', train=False, transform=data_transform, download=True)

        # Split the training dataset into training and validation sets
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    elif args.dataset == 'RSNA':
        train_dataset = RSNADataset(csv_file = args.csv, data_folder = args.data_dir, split='train', transform=data_transform)
        val_dataset = RSNADataset(csv_file = args.csv, data_folder = args.data_dir, split='val', transform=data_transform)
        test_dataset = RSNADataset(csv_file = args.csv, data_folder = args.data_dir, split='test', transform=data_transform)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=True)
        test_dataset = DataClass(split='test', transform=data_transform, download=download,  as_rgb=True)
        val_dataset = DataClass(split='val', transform=data_transform, download=download,  as_rgb=True)

        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(args.dataset)
    print(len(train_dataset))
    print("===================")
    print(len(val_dataset))
    print("===================")
    print(len(test_dataset))
    # exit()
    if args.initialisation == 'clip':
        base_model, preprocess = clip.load(load_model, 'cuda', jit=False)
        full_model = CLIPModel(base_model, num_classes=n_classes)
    elif args.initialisation == 'imagenet' or args.initialisation == 'chexpert':
        if args.use_pretrained:
            print('USING PRETRAINED WEIGHTS')
            if args.model == 'resnet18':
                # full_model =  getattr(models, load_model)(pretrained=True)
                full_model =  GeoModel()
                
                ret_msg = full_model.load_state_dict(torch.load("/home/santosh.sanjeev/Downloads/vicreg-cifar100-2wdjrj1u-ep=999.ckpt", 
                )["state_dict"],strict=False)
                # print(ret_msg)
                # time.sleep(5)
                # print(full_model)
                full_model.backbone.fc = nn.Linear(full_model.backbone.fc.in_features, args.n_classes)
            elif args.model == 'densenet121-res224-chex':
                full_model = CheXpertModel(xrv.models.DenseNet(weights="densenet121-res224-chex"),2) # CheXpert (Stanford)
                # full_model.classifier = nn.Linear(full_model.classifier.in_features, args.n_classes)

            elif args.model=='vit_b_16':
                print('in thisssssssssss')
                #image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
                # model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
                # print(model)
                # model_name = 'vit_base_patch16_224'  # Example model, you can use other available ViT models
                full_model = vit_base_patch16_224(pretrained=True)
                full_model.head = nn.Linear(full_model.head.in_features, args.n_classes)

                # print(full_model)
                # model =  getattr(models, load_model)(pretrained=True)
        else:
            model =  getattr(models, load_model)(pretrained=False)
        # model = torch.nn.Sequential(*list(model.children())[:-1])
        # print('modellllll after', model)
        # full_model = ViTModel(model, in_channels = 768, num_classes=n_classes)
        # print('finallll modellll', full_model)
    # full_model.head = nn.Linear(full_model.head.in_features, args.n_classes)
    print(full_model)
    if args.lp_ft=='LP':
        print('ONLY LINEAR PROBING')
        for name,param in full_model.backbone.named_parameters():
            if not name.startswith('fc'):
                # print('LPPPPPPPPPPPPPP')
                param.requires_grad = False
    else:
        print('FULL FINETUNING')
    
    full_model.to(device)
    criterion = nn.CrossEntropyLoss()    
    optimizer = optim.AdamW(full_model.parameters(), lr=args.lr,weight_decay=1e-6)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    
    
    best_auc = 0

    for epoch in range(NUM_EPOCHS):
        train_correct = 0
        train_total = 0
        val_correct = 0
        val_total = 0
        train_predictions = []
        train_targets = []
        val_predictions = []
        val_targets = []
        train_loss = 0
        val_loss = 0

        train_f1_predictions = []
        val_f1_predictions = []

        full_model.train()
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # if args.model == 'vit':
                # inputs = image_processor(inputs)
            
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = full_model.forward(inputs)
            m = nn.Softmax(dim=1)
            predicted = m(outputs)

            if args.task == 'multi-label':
                targets = targets.squeeze().float()
                loss = criterion(outputs.squeeze(), targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs.squeeze(), targets)
            # print(targets, outputs)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

            _, t = torch.max(outputs.data, 1)

            # print(predicted)
            train_total += targets.size(0)
            train_correct += (t == targets).sum().item()
            if args.task == 'binary-class':
                predicted = predicted[:,-1]

            # outputs, predicted, t = outputs.to("cpu"), predicted.to("cpu"), t.to("cpu")


            train_predictions.extend(predicted.detach().cpu().numpy())
            train_f1_predictions.extend(t.detach().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())
            # breakpoint()
        train_accuracy = 100 * train_correct / train_total

        if args.task != 'binary-class':
            train_f1 = f1_score(train_targets, train_f1_predictions, average = 'macro')
            train_auc = roc_auc_score(train_targets, train_predictions, multi_class = 'ovr')
        else:
            train_f1 = f1_score(train_targets, train_f1_predictions)
            train_auc = roc_auc_score(train_targets, train_predictions)

        train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Training Accuracy: {train_accuracy:.2f}%, F1-Score: {train_f1:.4f}, AUC: {train_auc:.4f}, Loss: {train_loss:.4f}")

        full_model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                # forward pass
                outputs = full_model(inputs)
                m = nn.Softmax(dim=1)
                predicted = m(outputs)

                if args.task == 'multi-label':
                    targets = targets.squeeze().float()
                    # print(targets.shape, outputs.squeeze().shape)
                    loss = criterion(outputs.squeeze(), targets)
                else:
                    targets = targets.squeeze().long()
                    loss = criterion(outputs.squeeze(), targets)


                _, t = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (t == targets).sum().item()

                val_loss += loss.item() * inputs.size(0)
                if args.task == 'binary-class':
                    predicted = predicted[:,-1]
                # outputs, predicted, t = outputs.to("cpu"), predicted.to("cpu"), t.to("cpu")

                val_predictions.extend(predicted.detach().cpu().numpy())
                val_targets.extend(targets.detach().cpu().numpy())
                val_f1_predictions.extend(t.detach().cpu().numpy())

            val_accuracy = 100 * val_correct / val_total
            if args.task !='binary-class':
                val_f1 = f1_score(val_targets, val_f1_predictions, average = 'macro')
                val_auc = roc_auc_score(val_targets, val_predictions, average = 'macro', multi_class = 'ovr')
            else:
                val_f1 = f1_score(val_targets, val_f1_predictions)
                val_auc = roc_auc_score(val_targets, val_predictions)
            val_loss = val_loss / len(val_loader.dataset)
            # Print or log validation metrics
            print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.2f}%, F1-Score: {val_f1:.4f}, AUC: {val_auc:.4f}, Loss: {val_loss:.4f}")

            # Save the model if it has the best F1-score
            if val_auc > best_auc:
                best_auc = val_auc
                checkpoint = {
                    'model_state_dict': full_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'hyperparameters': {'learning_rate': args.lr,'batch_size': BATCH_SIZE, 'epochs':NUM_EPOCHS, 'best_epoch': epoch},
                    'results':{'train_loss':train_loss,'val_loss':val_loss,'train_acc':train_accuracy,'val_acc':val_accuracy,'train_F1':train_f1,'val_F1':val_f1, 'train_AUC':train_auc,'val_AUC':val_auc},
                    }
                torch.save(checkpoint,args.output_model_path)

        scheduler.step()