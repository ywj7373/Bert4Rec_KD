from .base import BaseModel
from .bert_modules.bert import BERT, SMALLBERT

import torch.nn as nn


class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        self.out = nn.Linear(self.bert.hidden, args.bert_num_items + 1)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)


class SmallBERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = SMALLBERT(args)
        self.out = nn.Linear(self.bert_hidden, args.bert_num_items + 1)

    @classmethod
    def code(cls):
        return 'small bert'

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)
