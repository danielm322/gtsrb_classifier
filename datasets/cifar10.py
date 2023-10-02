import torch
import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_cifar10_input_transformations(cifar10_normalize_inputs: bool,
                                      img_size: int,
                                      data_augmentations: str,
                                      anomalies: bool):
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
        if anomalies:
            anomaly_transforms = A.Compose(
                [
                    A.Resize(img_size, img_size, p=1),
                    A.OneOf([
                        # A.MotionBlur(blur_limit=16, p=1.0),
                        A.RandomFog(fog_coef_lower=0.7,
                                    fog_coef_upper=0.9,
                                    alpha_coef=0.8,
                                    p=1.0),
                        A.RandomSunFlare(flare_roi=(0.3, 0.3, 0.7, 0.7),
                                         src_radius=int(img_size * 0.8),
                                         num_flare_circles_lower=8,
                                         num_flare_circles_upper=12,
                                         angle_lower=0.5,
                                         p=1.0),
                        A.RandomSnow(brightness_coeff=2.5,
                                     snow_point_lower=0.6,
                                     snow_point_upper=0.8,
                                     p=1.0)
                    ], p=1.0),
                    cifar10_normalization(),
                    ToTensorV2()
                ]
            )
            return anomaly_transforms, anomaly_transforms
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
        if anomalies:
            anomaly_transforms = A.Compose(
                [
                    A.Resize(img_size, img_size, p=1),
                    A.OneOf([
                        # A.MotionBlur(blur_limit=16, p=1.0),
                        A.RandomFog(fog_coef_lower=0.7,
                                    fog_coef_upper=0.9,
                                    alpha_coef=0.8,
                                    p=1.0),
                        A.RandomSunFlare(flare_roi=(0.3, 0.3, 0.7, 0.7),
                                         src_radius=int(img_size * 0.8),
                                         num_flare_circles_lower=8,
                                         num_flare_circles_upper=12,
                                         angle_lower=0.5,
                                         p=1.0),
                        A.RandomSnow(brightness_coeff=2.5,
                                     snow_point_lower=0.6,
                                     snow_point_upper=0.8,
                                     p=1.0)
                    ], p=1.0),
                    ToTensorV2()
                ]
            )
            return anomaly_transforms, anomaly_transforms

    return train_transforms, test_transforms
