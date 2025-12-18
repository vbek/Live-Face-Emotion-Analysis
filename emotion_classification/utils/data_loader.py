import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(data_dir, batch_size=16, num_workers=1, model_type="cnn"):
    """
    Handles different image sizes: 96px for custom CNN/ResNet, 224px for ResNet18.
    """
    if model_type in ["cnn", "resnet"]:
        img_size = 96  
    elif model_type == "resnet18":
        img_size = 224  
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),  
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform_test)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, len(train_ds.classes)