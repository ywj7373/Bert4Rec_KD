from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)  # Create output directory and file
    model = model_factory(args)  # Initialize model
    train_loader, val_loader, test_loader = dataloader_factory(args)  # Load data
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()  # Train

    test_result = trainer.test()  # Evaluation
    save_test_result(export_root, test_result)  # Save result


def distill():
    pass


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'distill':
        distill()
    else:
        raise ValueError('Invalid mode')
