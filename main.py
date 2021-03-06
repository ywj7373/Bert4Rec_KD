import torch.nn as nn
import torch.nn.functional as F

from dataloaders import dataloader_factory
from models import model_factory
from options import args
from trainers import trainer_factory
from trainers.utils import extract_layers
from utils import *


def train():
    export_root = setup_train(args)  # Create output directory and file
    model = model_factory("bert", args)  # Initialize model
    train_loader, val_loader, test_loader = dataloader_factory(args)  # Load data
    trainer = trainer_factory(args, None, model, train_loader, val_loader, test_loader, export_root)

    def calculate_loss(model, batch):
        seqs, labels = batch
        logits = model(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        ce = nn.CrossEntropyLoss(ignore_index=0)
        loss = ce(logits, labels)

        return loss

    trainer.train(calculate_loss)  # Train

    test_result = trainer.test()  # Evaluation
    save_test_result(export_root, test_result)  # Save result
    torch.save(model, export_root + "_model")


def distill():
    # Prepare dataset
    train_loader, val_loader, test_loader = dataloader_factory(args)

    # Get Teacher Model
    teacher_model = model_factory("bert", args)
    export_root = get_name_of_last_experiment_path(args, args.experiment_dir, args.experiment_description)
    best_model = torch.load(os.path.join(export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
    teacher_model.load_state_dict(best_model)
    export_root = create_experiment_distill_folder(args)

    # Fetch teacher output and put it into dataset
    teacher_model.eval()
    teacher_model.to(args.device)

    teacher_logits = []
    for batch in train_loader:
        batch = [t.to(args.device) for t in batch]
        with torch.no_grad():
            seqs, labels = batch
            teacher_logits.append(teacher_model(seqs))

    # Get Student Model
    model = model_factory("small bert", args)

    # Load weights from teacher model
    layers = args.distill_layers
    assert (len(layers) == args.distill_num_blocks)
    distill_sd = extract_layers(layers, best_model)
    model.load_state_dict(distill_sd)

    # Train
    trainer = trainer_factory(args, teacher_logits, model, train_loader, val_loader, test_loader, export_root)

    def calculate_loss(model, batch, teacher_logits):
        seqs, labels = batch
        logits = model(seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        teacher_logits = teacher_logits.view(-1, logits.size(-1))

        ce = nn.CrossEntropyLoss(ignore_index=0)
        loss = args.distill_alpha * ce(logits, labels)

        if args.distill_loss_func == "soft":
            T = 1.0
            loss += (1 - args.distill_alpha) * nn.KLDivLoss()(
                F.log_softmax(logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1)
            )
        elif args.distill_loss_func == "mse":
            loss += (1 - args.distill_alpha) * nn.MSELoss()(logits, teacher_logits)

        return loss

    trainer.train(calculate_loss)

    # Evaluate and save test result
    test_result = trainer.test()
    save_test_result(export_root, test_result)
    torch.save(model, export_root + "_model")


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'distill':
        distill()
    else:
        raise ValueError('Invalid mode')
