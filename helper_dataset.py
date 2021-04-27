import torch
from torch.utils.data import sampler
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms


def get_dataloaders(batch_size, num_workers=0,
                            validation_fraction=None,
                            train_transforms=None,
                            test_transforms=None):

    train_dataset = datasets.ImageFolder('C:/Users/James Thomason/Stat453/Project/Data', transform=train_transforms)

    valid_dataset = datasets.ImageFolder('C:/Users/James Thomason/Stat453/Project/Data', transform=test_transforms)

    test_dataset = datasets.ImageFolder('C:/Users/James Thomason/Stat453/Project/Data', transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * 744)
        train_indices = torch.arange(0, 744 - num)
        valid_indices = torch.arange(744 - num, 744)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader