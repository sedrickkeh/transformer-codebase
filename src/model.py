import torch.nn as nn
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
        self.configuration["num_attention_heads"] = n_head
        self.configuration["num_hidden_layers"] = n_layer
        
        # Initializing a model from the bert-base-uncased style configuration
        self.model = BertModel(self.configuration)

    def forward(self, data, target, *mems):
        a = self.model(data)
        print(len(a))
        print(a[0].shape)
        print(a[1].shape)
        print(self.configuration)
