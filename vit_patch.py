from common import *
import configure

# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# from timm.models.vision_transformer import *

from copy import deepcopy
from functools import partial
from timm.models.layers import DropPath, trunc_normal_


# from timm.models.layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_
# from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
#
#

#-------------------------------------------------

# Rescale the grid of position embeddings when loading from state_dict. Adapted from
# https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

# def resize_pos_embed(posemb, posemb_new, num_tokens=1):
#     _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
#     ntok_new = posemb_new.shape[1]
#     if num_tokens:
#         posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
#         ntok_new -= num_tokens
#     else:
#         posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
#     gs_old = int(math.sqrt(len(posemb_grid)))
#     gs_new = int(math.sqrt(ntok_new))
#     _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
#     posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
#     posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
#     posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
#     posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
#     return posemb


#-------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,
        x: Tensor,
        mask: Optional[Tensor] = None
    )-> Tensor:

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale # B x self.num_heads x NxN
        if mask is not None:
            #mask = mask.unsqueeze(1).repeat(1,self.num_heads,1,1)
            mask = mask.unsqueeze(1).expand(-1,self.num_heads,-1,-1)
            attn = attn.masked_fill(mask == 0, -6e4)
            # attn = attn.masked_fill(mask == 0, -half('inf'))
            # https://github.com/NVIDIA/apex/issues/93
            # How to use fp16 training with masked operations

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x),mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=1, embed_dim=768, patch_size=16, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

#Vision Transformer
# A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
#     - https://arxiv.org/abs/2010.11929
#
# Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
#     - https://arxiv.org/abs/2012.12877


class VisionTransformer(nn.Module):

    # url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
    # vit_deit_base_distilled_patch16_384
    def __init__(self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
    ):

        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer  = nn.GELU


        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed( patch_size=patch_size, embed_dim=embed_dim)

        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Embedding(max_patch_row_col*max_patch_row_col, embed_dim)
        self.pos_drop  = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        # if representation_size and not distilled:
        #     self.num_features = representation_size
        #     self.pre_logits = nn.Sequential(OrderedDict([
        #         ('fc', nn.Linear(embed_dim, representation_size)),
        #         ('act', nn.Tanh())
        #     ]))
        # else:
        #     self.pre_logits = nn.Identity()
        trunc_normal_(self.pos_embed.weight, std=.02)


    @torch.jit.ignore
    #todo: ensure not decay in optimizer
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}


    def forward(self, patch, coord, patch_pad_mask):
        max_patch_row_col = configure.max_patch_row_col
        batch_size, max_of_num_patch, patch_size, patch_size = patch.shape
        x = patch.reshape(batch_size*max_of_num_patch, 1, patch_size, patch_size)

        x = self.patch_embed(x).reshape(batch_size,max_of_num_patch,-1)
        #cls_token = self.cls_token.expand(batch_size,max_of_num_patch, -1)
        #x = torch.cat((cls_token, x), dim=1) #cls token is already in patch
        x = x + self.pos_embed(coord[:, :, 0] * max_patch_row_col + coord[:, :, 1])
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x,patch_pad_mask)
        x = self.norm(x)

        return x


#################################################################3
from patch import *


def make_dummy_data():
    # make dummy data
    # image_id,width,height,scale,orientation
    meta = [
        ['000011a64c74', 325, 229, 2, 0, ],
        ['000019cc0cd2', 288, 148, 1, 0, ],
        ['0000252b6d2b', 509, 335, 2, 0, ],
        ['000026b49b7e', 243, 177, 1, 0, ],
        ['000026fc6c36', 294, 112, 1, 0, ],
        ['000028818203', 402, 328, 2, 0, ],
        ['000029a61c01', 395, 294, 2, 0, ],
        ['000035624718', 309, 145, 1, 0, ],
    ]
    batch_size = 8

    # <todo> check border for padding
    # <todo> pepper noise

    batch = {
        'num_patch': [],
        'patch': [],
        'coord': [],
    }
    for b in range(batch_size):
        image_id = meta[b][0]
        scale = meta[b][3]

        image_file = data_dir + '/%s/%s/%s/%s/%s.png' % ('train', image_id[0], image_id[1], image_id[2], image_id)
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        image = resize_image(image, scale)
        image = repad_image(image, patch_size)  # remove border and repad
        # print(image.shape)

        k, yx = image_to_patch(image, patch_size, pixel_pad=0, threshold=0)

        for y, x in yx:
            # cv2.circle(image,(x,y),8,128,1)
            x = x * patch_size
            y = y * patch_size
            cv2.rectangle(image, (x, y), (x + patch_size, y + patch_size), 128, 1)

        image_show('image-%d' % b, image, resize=1)
        cv2.waitKey(1)

        batch['patch'].append(k)
        batch['coord'].append(yx)
        batch['num_patch'].append(len(k))

    # ----
    max_of_num_patch = max(batch['num_patch'])
    mask = np.zeros((batch_size, max_of_num_patch, max_of_num_patch))
    patch = np.zeros((batch_size, max_of_num_patch, patch_size, patch_size))
    coord = np.zeros((batch_size, max_of_num_patch, 2))
    for b in range(batch_size):
        N = batch['num_patch'][b]
        patch[b, :N] = batch['patch'][b]
        coord[b, :N] = batch['coord'][b]
        mask[b, :N, :N] = 1

    num_patch = batch['num_patch']
    patch = torch.from_numpy(patch).float()
    coord = torch.from_numpy(coord).long()
    patch_pad_mask = torch.from_numpy(mask).byte()

    return patch,coord,patch_pad_mask


def run_check_vit_patch():

    patch, coord, patch_pad_mask = make_dummy_data()
    net = VisionTransformer()
    y = net(patch, coord, patch_pad_mask)
    print(y.shape)




# main #################################################################
if __name__ == '__main__':
     run_check_vit_patch()