import torch
import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


def get_cifar10_input_transformations(cifar10_normalize_inputs: bool, img_size: int, data_augmentations: str):
    if cifar10_normalize_inputs:
        if data_augmentations == 'extra':
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(p=0.3),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.)
                        ]),
                        p=0.2
                    ),
                    torchvision.transforms.RandomGrayscale(p=0.1),
                    torchvision.transforms.RandomVerticalFlip(p=0.3),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.01, 0.2))
                        ]),
                        p=0.2
                    ),
                    cifar10_normalization(),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif data_augmentations == 'basic':
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    cifar10_normalization(),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    cifar10_normalization(),
                    torchvision.transforms.ToTensor(),
                ]
            )
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(img_size, img_size)),
                cifar10_normalization(),
                torchvision.transforms.ToTensor(),
            ]
        )
    # No cifar10 normalization
    else:
        if data_augmentations == 'extra':
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(p=0.3),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.)
                        ]),
                        p=0.2
                    ),
                    torchvision.transforms.RandomGrayscale(p=0.1),
                    torchvision.transforms.RandomVerticalFlip(p=0.3),
                    torchvision.transforms.RandomApply(
                        torch.nn.ModuleList([
                            torchvision.transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.01, 0.2))
                        ]),
                        p=0.2
                    ),
                    torchvision.transforms.ToTensor(),
                ]
            )
        elif data_augmentations == 'basic':
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.RandomCrop(img_size, padding=int(img_size / 8)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            train_transforms = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(img_size, img_size)),
                    torchvision.transforms.ToTensor(),
                ]
            )
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=(img_size, img_size)),
                torchvision.transforms.ToTensor(),
            ]
        )

    return train_transforms, test_transforms
