from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, teacher_logits, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, teacher_logits, model, train_loader, val_loader, test_loader, export_root)

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics
