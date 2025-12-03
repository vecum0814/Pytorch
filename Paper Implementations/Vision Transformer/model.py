import torch
import torch.nn as nn
import math  # 추가

def get_1d_sincos_pos_embed(embed_dim: int, positions: torch.Tensor):
    """
    1D sinusoidal positional encoding
    positions: (N,) 0,1,2,... 위치 인덱스
    return: (N, embed_dim)
    """
    assert embed_dim % 2 == 0
    # 각 차원마다 다른 주파수
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))  # (dim/2,)

    out = positions.unsqueeze(1) * omega.unsqueeze(0)      # (N, dim/2)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (N, dim)
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int):
    """
    2D sinusoidal positional encoding
    grid_h, grid_w: 패치 그리드의 세로/가로 길이
    return: (grid_h * grid_w, embed_dim)
    """
    assert embed_dim % 4 == 0, "embed_dim은 4의 배수여야 2D sin/cos를 깔끔하게 나눌 수 있음"

    half_dim = embed_dim // 2  # 절반은 H용, 절반은 W용

    # 0,1,2,... 형태의 위치 인덱스
    pos_h = torch.arange(grid_h, dtype=torch.float32)  # (H,)
    pos_w = torch.arange(grid_w, dtype=torch.float32)  # (W,)

    # H축 1D sin/cos -> (H, half_dim)
    emb_h = get_1d_sincos_pos_embed(half_dim, pos_h)
    # W축 1D sin/cos -> (W, half_dim)
    emb_w = get_1d_sincos_pos_embed(half_dim, pos_w)

    # (H, W, half_dim)
    emb_h = emb_h[:, None, :].expand(grid_h, grid_w, half_dim)
    emb_w = emb_w[None, :, :].expand(grid_h, grid_w, half_dim)

    # 최종 (H, W, embed_dim)
    emb = torch.cat([emb_h, emb_w], dim=2)

    # (H*W, embed_dim)
    return emb.reshape(grid_h * grid_w, embed_dim)


def build_2d_sincos_position_embedding(embed_dim: int, num_patches: int, add_cls_token: bool = True):
    """
    ViT에서 바로 쓸 수 있는 형태로 만드는 helper
    num_patches: 패치 개수 (H * W, 정사각형이라면 sqrt(num_patches) == H == W)
    return: (1, num_patches + add_cls_token, embed_dim)
    """
    grid_size = int(math.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, "num_patches는 정사각형 그리드(H*W)여야 함"

    # (N, D)
    pos = get_2d_sincos_pos_embed(embed_dim, grid_size, grid_size)
    pos = pos.unsqueeze(0)  # (1, N, D)

    if add_cls_token:
        # CLS 토큰용 위치벡터는 0으로 두는 경우가 많음
        cls_pos = torch.zeros(1, 1, embed_dim)
        pos = torch.cat([cls_pos, pos], dim=1)  # (1, N+1, D)

    return pos



class LinearProjection(nn.Module):

    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super().__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim) # (1 * p^2 c) -> (1 x D)
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim)) # (1xD)의 클래스 토큰을 가장 왼쪽에 concat. 시켜줘야 하기 때문에 우선 (1xD) 사이즈의 학습 가능한 파라미터로 정의.
        pos_embedding = build_2d_sincos_position_embedding(
            embed_dim=latent_vec_dim,
            num_patches=num_patches,
            add_cls_token=True,
        )  # (1, num_patches + 1, latent_vec_dim)

        # 고정 positional encoding이므로 buffer로 등록 (학습 X, .to(device)될 때 같이 이동)
        self.register_buffer("pos_embedding", pos_embedding, persistent=False)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x): # 배치 단위로 (b x N x p^2 c) 들어온다.
        batch_size = x.size(0)
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1) 
        # linear_proj 이후엔 (b x N x D), repeat 함수를 통해 cls_token의 사이즈를 (b x 1 x D)로 조정.
        x += self.pos_embedding
        x = self.dropout(x)
        return x # Transformer의 Input을 return. 

class MultiheadedSelfAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads)
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim) # D -> D_h, D = k x D_h, 즉 D_h를 헤드 수만큼 계산 했다.
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.scale = torch.sqrt(self.head_dim * torch.ones(1)).to(device)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (b x N x h x D_h) -> (b x h x N x D_h), 헤드 수에 따라 qkv 값이 달라지니까 permute로 조정.
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1) # k^Transpose
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attention = torch.softmax(q @ k / self.scale, dim = -1)
        x = self.dropout(attention) @ v
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim)

        return x, attention

class TFencoderLayer(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiheadedSelfAttention(latent_vec_dim = latent_vec_dim, num_heads = num_heads, drop_rate = drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = nn.Sequential(nn.Linear(latent_vec_dim, mlp_hidden_dim),
                                 nn.GELU(), nn.Dropout(drop_rate),
                                 nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                 nn.Dropout(drop_rate))

    def forward(self, x):
        z = self.ln1(x)
        z, att = self.msa(z)
        z = self.dropout(z)
        x = x + z
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z

        return x, att

class VisionTransformer(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes):
        super().__init__()
        self.patchembedding = LinearProjection(patch_vec_size = patch_vec_size, num_patches = num_patches, # patch_vec_size = p^2 * c
                                               latent_vec_dim = latent_vec_dim, drop_rate = drop_rate)
        self.transformer = nn.ModuleList([TFencoderLayer(latent_vec_dim = latent_vec_dim, num_heads = num_heads,
                                                         mlp_hidden_dim = mlp_hidden_dim, drop_rate = drop_rate)
                                          for _ in range(num_layers)]) # Transformer의 Layer가 반복되는 과정을 리스트 안에 넣어서 처리. num_layer만큼 list에 append 한다.

        self.mlp_head = nn.Sequential(nn.LayerNorm(latent_vec_dim), nn.Linear(latent_vec_dim, num_classes))

    def forward(self, x):
        att_list = []
        x = self.patchembedding(x)
        for layer in self.transformer: # List로 구성되어 있는 레이어를 하나씩 불러
            x, att = layer(x)
            att_list.append(att)
        x = self.mlp_head(x[:,0]) # class token에 해당되는 부분의 벡터만 사용

        return x, att_list
