from common import *
from typing import Tuple, Dict

from fairseq import utils
from fairseq.models import *
from fairseq.modules import *

#https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

# ------------------------------------------------------
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need

class PositionEncode1D(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert (dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2)* (-math.log(10000.0) / dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        pos[0, :, 0::2] = torch.sin(position * d)
        pos[0, :, 1::2] = torch.cos(position * d)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, T, dim = x.shape
        x = x + self.pos[:,:T]
        return x

# https://gitlab.maastrichtuniversity.nl/dsri-examples/dsri-pytorch-workspace/-/blob/c8a88cdeb8e1a0f3a2ccd3c6119f43743cbb01e9/examples/transformer/fairseq/models/transformer.py
#https://github.com/pytorch/fairseq/issues/568
# fairseq/fairseq/models/fairseq_encoder.py

# https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py
class TransformerEncode(FairseqEncoder):

    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})
        #print('my TransformerEncode()')

        self.layer = nn.ModuleList([
            TransformerEncoderLayer(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'encoder_normalize_before': True,
                'encoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):# T x B x C
        #print('my TransformerEncode forward()')
        for layer in self.layer:
            x = layer(x)
        x = self.layer_norm(x)
        return x


# https://mt.cs.upc.edu/2020/12/21/the-transformer-fairseq-edition/
# for debug
# class TransformerDecode(FairseqDecoder):
#     def __init__(self, dim, ff_dim, num_head, num_layer):
#         super().__init__({})
#         print('my TransformerDecode()')
#
#         self.layer = nn.ModuleList([
#             TransformerDecoderLayer(Namespace({
#                 'decoder_embed_dim': dim,
#                 'decoder_attention_heads': num_head,
#                 'attention_dropout': 0.1,
#                 'dropout': 0.1,
#                 'decoder_normalize_before': True,
#                 'decoder_ffn_embed_dim': ff_dim,
#             })) for i in range(num_layer)
#         ])
#         self.layer_norm = nn.LayerNorm(dim)
#
#
#     def forward(self, x, mem, x_mask):# T x B x C
#         print('my TransformerDecode forward()')
#         for layer in self.layer:
#             x = layer(x, mem, self_attn_mask=x_mask)[0]
#         x = self.layer_norm(x)
#         return x  # T x B x C

# https://fairseq.readthedocs.io/en/latest/tutorial_simple_lstm.html
# see https://gitlab.maastrichtuniversity.nl/dsri-examples/dsri-pytorch-workspace/-/blob/c8a88cdeb8e1a0f3a2ccd3c6119f43743cbb01e9/examples/transformer/fairseq/models/transformer.py
class TransformerDecode(FairseqIncrementalDecoder):
    def __init__(self, encoder_dim, decoder_dim, ff_dim, num_head, num_layer):
        super().__init__({})
        #print('my TransformerDecode()')

        self.layer = nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'encoder_embed_dim': encoder_dim,
                'decoder_embed_dim': decoder_dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(decoder_dim)


    def forward(self, x, mem, x_mask, x_pad_mask, mem_pad_mask):
            #print('my TransformerDecode forward()')
            for layer in self.layer:
                x = layer(
                    x,
                    mem,
                    self_attn_mask=x_mask,
                    self_attn_padding_mask=x_pad_mask,
                    encoder_padding_mask=mem_pad_mask,
                )[0]
            x = self.layer_norm(x)
            return x  # T x B x C

    #def forward_one(self, x, mem, incremental_state):
    def forward_one(self,
            x   : Tensor,
            mem : Tensor,
            mem_pad_mask : Tensor,
            incremental_state : Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    )-> Tensor:
        x = x[-1:]
        for layer in self.layer:
            x = layer(
                x,
                mem,
                encoder_padding_mask=mem_pad_mask,
                incremental_state=incremental_state
            )[0]
        x = self.layer_norm(x)
        return x

'''
https://fairseq.readthedocs.io/en/latest/_modules/fairseq/modules/transformer_layer.html
class TransformerDecoderLayer(nn.Module):
def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        
        
https://github.com/pytorch/fairseq/blob/05b86005bcca0155319fa9b81abfd69f63c06906/examples/simultaneous_translation/utils/data_utils.py#L31      
'''
