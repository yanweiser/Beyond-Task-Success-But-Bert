import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.wrap_var import to_var
from transformers import BertModel

use_cuda = torch.cuda.is_available()

class Encoder(nn.Module):
    """docstring for EncoderBasic."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        """Short summary.

        Parameters
        ----------
        kwargs : dict
            'vocab_size' : size of the vocabulary
            'word_embedding_dim' : Dimension of the word embeddings
            'hidden_dim' : Dimension of hidden state of the encoder LSTM
            'word_pad_token' : Pad token in the vocabulary
            'num_layers' : Number of layers of the encoder LSTM
            'visual_features_dim' : Dimension of avg pool layer of the Resnet 152
            'scale_to' : Used to scale the concatenated visual features and LSTM hidden state to be used as input to next modules
            'decider' : Depending on decider the return from the forward pass changes

        """

        self.encoder_args = kwargs

        # self.word_embeddings = nn.Embedding(self.encoder_args['vocab_size'], self.encoder_args['word_embedding_dim'], padding_idx=self.encoder_args['word_pad_token'])

        # self.rnn = nn.LSTM(self.encoder_args['word_embedding_dim'], self.encoder_args['hidden_dim'], num_layers=self.encoder_args['num_layers'], batch_first=True)

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Looking for better variable name here
        self.scale_to = nn.Linear(self.encoder_args['hidden_dim']+self.encoder_args['visual_features_dim'], self.encoder_args['scale_to'])

        # Using tanh to keep the input to all other modules to be between 1 and -1
        self.tanh = nn.Tanh()

    def forward(self, **kwargs):
        """Short summary.

        Parameters
        ----------
        kwargs : dict
            'visual_features' : avg pool layer of the Resnet 152
            'history' : Dialogue history
            'history_len' : Length of dialogue history for pack_padded_sequence
            'history_types' : input types for the bert model
            'history_atts' : attention mask for the bert model

        Returns
        -------
        output : dict
            'encoder_hidden' : final output from Encoder for all other modules as input

        """

        history, history_len = kwargs['history'], kwargs['history_len']

        history_types = kwargs['history_types']

        history_atts = kwargs['history_atts']

        visual_features = kwargs['visual_features']

        batch_size = history.size(0)


        if history_atts.shape != history.shape:
            history_atts = history_atts.reshape(history.size())
        if history_types.shape != history.shape:
            history_types = history_types.reshape(history.size())
            
        if isinstance(history_len, Variable):
            history_len = history_len.data

        with torch.no_grad():
            encoding = self.bert(
                input_ids = history.view(-1,200),
                token_type_ids = history_types.view(-1,200),
                attention_mask = history_atts.view(-1,200)
            )[1]
        # batchsize x 768

        _hidden = encoding.unsqueeze(1)

        # batchsize x 1 x 768

        # This is similar to Best of both worlds paper
        encoder_hidden = self.tanh(self.scale_to(torch.cat(
            [
                _hidden.view(-1,1,self.encoder_args['hidden_dim']), 
                 visual_features.view(-1,1,self.encoder_args['visual_features_dim'])
            ], dim=2)))

        return encoder_hidden
