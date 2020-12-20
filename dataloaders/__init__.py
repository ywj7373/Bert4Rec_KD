from datasets import dataset_factory
from .bert import BertDataloader


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
}


def dataloader_factory(args):
    dataset = dataset_factory(args) # Get dataset
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset) # Initialize dataloader with the dataset
    train, val, test = dataloader.get_pytorch_dataloaders() 
    return train, val, test


def get_train_dataset(args):
    dataset = dataset_factory(args) # Get dataset
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset) # Initialize dataloader with the dataset
    return dataloader.get_train_dataset()