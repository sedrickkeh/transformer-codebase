import torch.nn as nn
import torch.nn.functional as F

from onmt.encoders.transformer import TransformerEncoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.modules.embeddings import Embeddings
from transformers import BertModel, BertConfig

class TransformerBase(nn.Module):
    def __init__(self, ntokens, n_layer, n_head, d_model, 
        d_head, d_inner, dropout, dropatt):
        super(TransformerBase, self).__init__()

        # Initializing a BERT bert-base-uncased style configuration
        self.configuration = BertConfig()
        self.configuration.vocab_size = ntokens
        self.configuration.hidden_size = d_inner
        self.configuration.num_hidden_layers = n_layer
        self.configuration.num_attention_heads = n_head
        self.configuration.hidden_dropout_prob = dropout
        self.configuration.attention_probs_dropout_prob = dropatt
        
        # Initializing a model from the bert-base-uncased style configuration
        self.bertmodel = BertModel(self.configuration)
        self.generator = Generator(d_inner, ntokens)

    def forward(self, data, target, *mems):
        tgt_len = target.size(0)
        a = self.bertmodel(data)
        pred_hid = a[0][-tgt_len:]
        b = self.generator(a[0], target)
        return b

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, target):
        return F.log_softmax(self.proj(x), dim=-1)