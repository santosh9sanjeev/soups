from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator
from dataset.medmnist_dataset import PneumoniaDataset
import clip
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--test', default=False, help='trian (False) or test (True)')
    parser.add_argument('--reload_ckpt_dir', default=None, type=str, help='ckpt_dir')
    parser.add_argument('--data_flag', default='pneumoniamnist', type=str, help='dataset')
    parser.add_argument('--download', default='True', type=bool, help='flag')
    parser.add_argument('--model', default='True', type=str, help='RN50')

    parser.add_argument('--n_gpus', default=1, type=int)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=4.5e-6, type=float, help='learning rate')
    parser.add_argument('--accumulate_grad_batches', default=1, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')

    parser.add_argument('--img_root_dir', default='path/to/mimic-cxr-jpg', type=str)
    parser.add_argument('--text_root_dir', default='path/to/mimic-cxr-database', type=str)

    args = parser.parse_args()


    data_flag = args.data_flag
    download = args.download

    NUM_EPOCHS = args.nepochs
    BATCH_SIZE = args.batch_size
    lr = args.lr

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)


    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)


    print(train_dataset)
    print("===================")
    print(val_dataset)
    print("===================")
    print(test_dataset)

    base_model, preprocess = clip.load(args.model, 'cuda', jit=False)
    full_model = model.

    if args.task == "multi-label" or args.task =="binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(full_model.parameters(), lr=args.lr, momentum=0.9)


    full_model.
    for epoch in range(NUM_EPOCHS):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        
        full_model.train()
        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if args.task == 'multi-label' or args.task == 'binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()

        
    def test(split):
        model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        
        data_loader = train_loader_at_eval if split == 'train' else test_loader

        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)

                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)

            y_true = y_true.numpy()
            y_score = y_score.detach().numpy()
            
            evaluator = Evaluator(data_flag, split)
            metrics = evaluator.evaluate(y_score)
        
            print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))

            
    print('==> Evaluating ...')
    test('train')
    test('test')