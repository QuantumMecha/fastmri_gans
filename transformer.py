import torch
import torch.nn as nn
import math
import numpy as np
import scipy.signal

from transformer_helpers import *

class Discriminator(nn.Module):
    def __init__(self, diff_aug,d_depth=3,d_act=None,d_norm=None,df_dim=384,d_window_size=4, img_size=32, patch_size=2, in_chans=1, num_classes=2,  depth=7,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = df_dim
        self.embed_dim = df_dim  
        self.diff_aug=diff_aug
        self.depth = d_depth
        self.patch_size = patch_size
        self.norm_layer = norm_layer
        act_layer = d_act
        self.window_size = d_window_size
        self.img_size=img_size
        
        act_layer = d_act
        if patch_size != 6:
            self.fRGB_1 = nn.Conv2d(1, self.embed_dim//4, kernel_size=patch_size*2, stride=patch_size, padding=patch_size//2)
            self.fRGB_2 = nn.Conv2d(1, self.embed_dim//4, kernel_size=patch_size*2, stride=patch_size*2, padding=0)
            self.fRGB_3 = nn.Conv2d(1, self.embed_dim//2, kernel_size=patch_size*4, stride=patch_size*4, padding=0)
            num_patches_1 = (self.img_size // patch_size)**2
            num_patches_2 = ((self.img_size//2) // patch_size)**2
            num_patches_3 = ((self.img_size//4) // patch_size)**2
        else:
            self.fRGB_1 = nn.Conv2d(1, self.embed_dim//4, kernel_size=6, stride=4, padding=1)
            self.fRGB_2 = nn.Conv2d(1, self.embed_dim//4, kernel_size=10, stride=8, padding=1)
            self.fRGB_3 = nn.Conv2d(1, self.embed_dim//2, kernel_size=18, stride=16, padding=1)
            num_patches_1 = (self.img_size // patch_size)**2
            num_patches_2 = ((self.img_size//2) // patch_size)**2
            num_patches_3 = ((self.img_size//4) // patch_size)**2
            self.patch_size = 4

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, num_patches_1, self.embed_dim//4))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, num_patches_2, self.embed_dim//2))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, num_patches_3, self.embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks_1 = nn.ModuleList([
            DisBlock(
                dim=self.embed_dim//4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])
        self.blocks_2 = nn.ModuleList([
            DisBlock(
                dim=self.embed_dim//2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth-1)])
        self.blocks_21 = nn.ModuleList([
            DisBlock(
                dim=self.embed_dim//2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(1)])
        self.blocks_3 = nn.ModuleList([
            DisBlock(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth+1)])
        self.last_block = nn.Sequential(
            Blockdis(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], act_layer=act_layer, norm_layer=norm_layer)
            )
        
        self.norm = CustomNorm(norm_layer, self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed_1, std=.02)
        trunc_normal_(self.pos_embed_2, std=.02)
        trunc_normal_(self.pos_embed_3, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
        if 'filter' in self.diff_aug:
            Hz_lo = np.asarray(wavelets['sym2'])            # H(z)
            Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
            Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
            Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
            Hz_fbank = np.eye(4, 1)                         # Bandpass(H(z), b_i)
            for i in range(1, Hz_fbank.shape[0]):
                Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
                Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
                Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
            Hz_fbank = torch.as_tensor(Hz_fbank, dtype=torch.float32)
            self.register_buffer('Hz_fbank', torch.as_tensor(Hz_fbank, dtype=torch.float32))
        else:
            self.Hz_fbank = None
        if 'geo' in self.diff_aug:
            self.register_buffer('Hz_geom', upfirdn2d.setup_filter(wavelets['sym6']))
        else:
            self.Hz_geom = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x, aug=True, epoch=400):
        if "None" not in self.diff_aug and aug:
            x = DiffAugment(x, self.diff_aug, True, [self.Hz_geom, self.Hz_fbank])
        elif "None" not in self.diff_aug:
            x = DiffAugment(x, "translation", True, [self.Hz_geom, self.Hz_fbank])
        B, _, H, W = x.size()
        H = W = H//self.patch_size
        #print("Discriminator 0 Clearing cache")
        
        x_1 = self.fRGB_1(x).flatten(2).permute(0,2,1)
        x_2 = self.fRGB_2(x).flatten(2).permute(0,2,1)
        x_3 = self.fRGB_3(x).flatten(2).permute(0,2,1)

        B = x.shape[0]
        #print("Discriminator 1 Clearing cache")
        
        x = x_1 + self.pos_embed_1
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size*self.window_size, C)
        for blk in self.blocks_1:
            x = blk(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B,H*W,C)
        #print("Discriminator 2 Clearing cache")
            
        _, _, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = nn.AvgPool2d(2)(x)
        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_2], dim=-1)
        x = x + self.pos_embed_2
        #print("Discriminator 3 Clearing cache")
        
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size*self.window_size, C)
        for blk in self.blocks_2:
            x = blk(x)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, H, W).view(B,H*W,C)
        for blk in self.blocks_21:
            x = blk(x)
        #print("Discriminator 4 Clearing cache")
        
        _, _, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = nn.AvgPool2d(2)(x)
        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_3], dim=-1)
        x = x + self.pos_embed_3
        #print("Discriminator 5 Clearing cache")
        
        for blk in self.blocks_3:
            x = blk(x)
            
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.last_block(x)
        x = self.norm(x)
        #print("Discriminator 6 Clearing cache")
        
        return x[:,0]

    def forward(self, x, aug=True, epoch=400):
        x = self.forward_features(x, aug=aug, epoch=epoch)
        x = self.head(x)
        return x

class Generator(nn.Module):
    def __init__(self, g_act=None,g_depth='5,4,4,4,4,4', mlp_ratio=4, drop_rate=0.5,img_size=224, patch_size=16, in_chans=1,embed_dim=384, depth=5,num_heads=4, qkv_bias=False, qk_scale=None, attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super(Generator, self).__init__()
        
        assert(img_size % 8 == 0)
        assert(img_size >= 32)

        self.img_size = img_size
        self.ch = embed_dim
        self.bottom_width = img_size // 32
        self.embed_dim = embed_dim 
        self.window_size = min(max(self.bottom_width, 4), 16)
        # assert(self.window_size <= self.bottom_width ** 2)

        self.norm_layer = norm_layer
        self.mlp_ratio = mlp_ratio
        latent_dim = img_size * img_size

        depth = [int(i) for i in g_depth.split(",")]
        act_layer = g_act
        self.l2_size = 0
        
        if self.l2_size == 0:
            self.l1 = nn.Linear(latent_dim, (self.bottom_width ** 2) * self.embed_dim)
        elif self.l2_size > 1000:
            self.l1 = nn.Linear(latent_dim, (self.bottom_width ** 2) * self.l2_size//16)
            self.l2 = nn.Sequential(
                        nn.Linear(self.l2_size//16, self.l2_size),
                        nn.Linear(self.l2_size, self.embed_dim)
                      )
        else:
            self.l1 = nn.Linear(latent_dim, (self.bottom_width ** 2) * self.l2_size)
            self.l2 = nn.Linear(self.l2_size, self.embed_dim)
        #print("1 Clearing cache")
        
        self.embedding_transform = nn.Linear(latent_dim, (self.bottom_width ** 2) * self.embed_dim)
        #print("2 Clearing cache")
        
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.bottom_width**2, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (self.bottom_width*2)**2, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (self.bottom_width*4)**2, embed_dim))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, (self.bottom_width*8)**2, embed_dim//4))
        self.pos_embed_5 = nn.Parameter(torch.zeros(1, (self.bottom_width*16)**2, embed_dim//16))
        self.pos_embed_6 = nn.Parameter(torch.zeros(1, (self.bottom_width*32)**2, embed_dim//64))
        
        self.embed_pos = nn.Parameter(torch.zeros(1, self.bottom_width**2, embed_dim))
                                        
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4,
            self.pos_embed_5,
            self.pos_embed_6
        ]
        #print("3 Clearing cache")
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[0])]  # stochastic depth decay rule

        self.blocks_2 = StageBlock(
                        depth=depth[1],
                        dim=embed_dim, 
                        embedding_dim=embed_dim,
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size//4
                        )
        #print("4 Clearing cache")

        self.blocks_3 = StageBlock(
                        depth=depth[2],
                        dim=embed_dim, 
                        embedding_dim=embed_dim,
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size//2
                        )
        self.blocks_4 = StageBlock(
                        depth=depth[3],
                        dim=embed_dim//4, 
                        embedding_dim=embed_dim,
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size
                        )
        #print("Clearing cache")
        
        self.blocks_5 = StageBlock(
                        depth=depth[4],
                        dim=embed_dim//16,
                        embedding_dim=embed_dim,
                        num_heads=num_heads, 
                        mlp_ratio=mlp_ratio, 
                        qkv_bias=qkv_bias, 
                        qk_scale=qk_scale,
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=0, 
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        window_size=self.window_size
                        )
        #print("Clearing cache")
                                        
        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim//64, 1, 1, 1, 0)
        )
        
    def forward(self, z):
        #print(self.latent_dim,self.l1,z.shape)
        if self.l2_size == 0:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.embed_dim)
        elif self.l2_size > 1000:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size//16)
            x = self.l2(x)
        else:
            x = self.l1(z).view(-1, self.bottom_width ** 2, self.l2_size)
            x = self.l2(x)
        #print("Generator 1 Clearing cache")
        
        # input noise
        x = x + self.pos_embed[0].to(x.get_device())
        B = x.size(0)
        H, W = self.bottom_width, self.bottom_width
        #print("2 Clearing cache")
        
        # embedding
        embedding = self.embedding_transform(z).view(-1, self.bottom_width ** 2, self.embed_dim)
        embedding = embedding + self.embed_pos.to(embedding.get_device())
        
        #print("3 Clearing cache")
        
        x, H, W = bicubic_upsample(x, H, W)
        x = x + self.pos_embed[1].to(x.get_device())
        B, _, C = x.size()
        x, _ = self.blocks_2(x, embedding)
        #print("3 Clearing cache")

        
        x, H, W = bicubic_upsample(x, H, W)
        x = x + self.pos_embed[2].to(x.get_device())
        B, _, C = x.size()

        x, _ = self.blocks_3(x, embedding)
        #print("4 Clearing cache")
        
        x, H, W = pixel_upsample(x, H, W)
        x = x + self.pos_embed[3].to(x.get_device())
        B, _, C = x.size()

        x, _ = self.blocks_4(x, embedding)
        #print("5 Clearing cache")
        
        x, H, W = updown(x, H, W)
        
        x, H, W = pixel_upsample(x, H, W)
        x = x + self.pos_embed[4].to(x.get_device())
        B, _, C = x.size()
        #print("6 Clearing cache")
        
        x, _ = self.blocks_5(x, embedding)
        x, H, W = updown(x, H, W)
        
        x, H, W = pixel_upsample(x, H, W)
        x = x + self.pos_embed[5].to(x.get_device())
        B, _, C = x.size()

        x = x.permute(0,2,1).view(B, C, self.img_size, self.img_size)
        output = self.deconv(x)
        #print("7 Clearing cache")
        
        return output