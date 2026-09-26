import json
import math
import os
import random
import re
import subprocess
import sys

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.vit_model import create_vit
from utils.common import IMAGENET_MEAN, IMAGENET_STD, build_default_transforms


DEFAULT_VIT_WEIGHT = os.path.join(ROOT_DIR, 'weights', 'vit_best.pth')
DEFAULT_SAMPLE_IMAGE = os.path.join(
    ROOT_DIR,
    'dataset',
    'ship_fine',
    'FGSCR',
    '001.Nimitz-class_aircraft_carrier',
    'P0002.bmp',
)


def resolve_default_sample_image():
    if os.path.exists(DEFAULT_SAMPLE_IMAGE):
        return DEFAULT_SAMPLE_IMAGE
    dataset_root = os.path.join(ROOT_DIR, 'dataset', 'ship_fine', 'test')
    for root, _, files in os.walk(dataset_root):
        for filename in files:
            if filename.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                return os.path.join(root, filename)
    return DEFAULT_SAMPLE_IMAGE


def _pick_font_family(candidates, available_names):
    for name in candidates:
        if name and name in available_names:
            return name
    return None


def _contains_cjk(text):
    if not text:
        return False
    return re.search(r'[\u4e00-\u9fff]', str(text)) is not None


def configure_matplotlib_cjk_font(preferred_font=None):
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties

    available_names = {f.name for f in font_manager.fontManager.ttflist}
    en_candidates = ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif']
    zh_candidates = [
        preferred_font,
        'SimSun',
        'Songti SC',
        'STSong',
        'Noto Serif CJK SC',
        'Noto Serif CJK JP',
        'Noto Serif CJK TC',
        'Noto Sans CJK SC',
        'Source Han Serif SC',
        'Source Han Sans SC',
        'WenQuanYi Zen Hei',
        'AR PL UMing CN',
        'SimHei',
    ]
    zh_candidates = [x for x in zh_candidates if x]

    en_font = _pick_font_family(en_candidates, available_names) or 'DejaVu Serif'

    if preferred_font and preferred_font in available_names:
        plt.rcParams['font.family'] = [preferred_font, en_font]
        plt.rcParams['font.sans-serif'] = [preferred_font, 'DejaVu Sans']
        plt.rcParams['font.serif'] = [en_font, 'DejaVu Serif']
        plt.rcParams['font.size'] = 10.5
        plt.rcParams['axes.unicode_minus'] = False
        return preferred_font, en_font

    selected = _pick_font_family(zh_candidates, available_names)
    if selected is None:
        patterns = ('SimSun', 'Songti', 'STSong', 'Noto CJK', 'Source Han', 'WenQuanYi', 'YaHei', 'SimHei', 'Heiti', 'PingFang')
        for name in sorted(available_names):
            if any(pat in name for pat in patterns):
                selected = name
                break

    if selected is not None:
        plt.rcParams['font.family'] = [selected, en_font]
        plt.rcParams['font.sans-serif'] = [selected, 'DejaVu Sans']
        plt.rcParams['font.serif'] = [en_font, 'DejaVu Serif']
        plt.rcParams['font.size'] = 10.5
        plt.rcParams['axes.unicode_minus'] = False
        return selected, en_font

    try:
        out = subprocess.check_output(['fc-list', ':lang=zh', 'file', 'family'], text=True, stderr=subprocess.STDOUT)
        font_file = None
        for line in out.splitlines():
            parts = line.split(':', 1)
            if parts and parts[0].strip().lower().endswith(('.ttf', '.ttc', '.otf')):
                font_file = parts[0].strip()
                break

        if font_file and os.path.exists(font_file):
            font_manager.fontManager.addfont(font_file)
            loaded_name = FontProperties(fname=font_file).get_name()
            plt.rcParams['font.family'] = [loaded_name, en_font]
            plt.rcParams['font.sans-serif'] = [loaded_name, 'DejaVu Sans']
            plt.rcParams['font.serif'] = [en_font, 'DejaVu Serif']
            plt.rcParams['font.size'] = 10.5
            plt.rcParams['axes.unicode_minus'] = False
            return loaded_name, en_font
    except Exception:
        pass

    plt.rcParams['font.family'] = [en_font]
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['font.serif'] = [en_font, 'DejaVu Serif']
    plt.rcParams['font.size'] = 10.5
    plt.rcParams['axes.unicode_minus'] = False
    return None, en_font


def set_paper_style():
    from matplotlib import font_manager

    zh_font, en_font = configure_matplotlib_cjk_font()
    available_names = {f.name for f in font_manager.fontManager.ttflist}

    sans_candidates = [
        zh_font,
        'Noto Sans CJK SC',
        'Noto Serif CJK SC',
        'Noto Sans CJK JP',
        'Noto Serif CJK JP',
        'WenQuanYi Zen Hei',
        'DejaVu Sans',
    ]
    serif_candidates = [en_font, 'DejaVu Serif']

    sans_stack = [x for x in sans_candidates if x and x in available_names]
    serif_stack = [x for x in serif_candidates if x and x in available_names]

    if not sans_stack:
        sans_stack = ['DejaVu Sans']
    if not serif_stack:
        serif_stack = ['DejaVu Serif']

    plt.rcParams.update({
        'font.family': sans_stack,
        'font.sans-serif': sans_stack,
        'font.serif': serif_stack,
        'axes.unicode_minus': False,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
    })


def enhance_attention_map(attention_map, low_q=0.60, high_q=0.995, gamma=0.80, blur_kernel=5):
    """Enhance attention map readability for paper figures.

    This function only changes visualization contrast, not model outputs.
    """
    attn = np.asarray(attention_map, dtype=np.float32)
    low = float(np.quantile(attn, float(low_q)))
    high = float(np.quantile(attn, float(high_q)))
    if high - low < 1e-8:
        return np.zeros_like(attn, dtype=np.float32)

    attn = np.clip(attn, low, high)
    attn = (attn - low) / (high - low)
    attn = np.power(attn, float(gamma))

    kernel = int(blur_kernel)
    if kernel % 2 == 0:
        kernel += 1
    if kernel >= 3:
        attn = cv2.GaussianBlur(attn, (kernel, kernel), 0)

    return min_max_normalize(attn)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_figure(fig, path, dpi=300):
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    return path


def get_device(device_name=None):
    if isinstance(device_name, torch.device):
        return device_name
    if device_name:
        return torch.device(device_name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_class_names(root_dir=ROOT_DIR):
    class_path = os.path.join(root_dir, 'class_indices.json')
    if not os.path.exists(class_path):
        raise FileNotFoundError(f'class_indices.json not found: {class_path}')
    with open(class_path, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    return [class_map[str(i)] for i in range(len(class_map))]


def load_test_dataset(dataset_subdir='dataset/ship_fine', split='test'):
    test_dir = os.path.join(ROOT_DIR, dataset_subdir, split)
    transform = build_default_transforms(val_occlusion_mode='none')['val']
    return datasets.ImageFolder(test_dir, transform)


def build_inference_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def read_rgb_image(image_path, image_size=224):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img


def image_to_tensor(image_rgb):
    transform = build_inference_transform()
    return transform(image_rgb).unsqueeze(0)


def load_checkpoint_state_dict(weight_path, map_location='cpu'):
    state = torch.load(weight_path, map_location=map_location)
    if isinstance(state, dict):
        for key in ('state_dict', 'model_state_dict', 'model', 'state'):
            if key in state and isinstance(state[key], dict):
                return state[key]
        sample_keys = list(state.keys())[:5]
        if any(isinstance(state[k], torch.Tensor) for k in sample_keys if k in state):
            return state
    return state


def resolve_vit_weight(weight_path=None):
    if weight_path:
        return weight_path
    candidates = [
        DEFAULT_VIT_WEIGHT,
        os.path.join(ROOT_DIR, 'weights', 'rank2', 'vit_best.pth'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return DEFAULT_VIT_WEIGHT


def build_vit_model(num_classes, weight_path=None, device=None):
    device = get_device(device)
    weight_path = resolve_vit_weight(weight_path)

    model = create_vit(num_classes=num_classes, pretrained=False)
    state_dict = load_checkpoint_state_dict(weight_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model, weight_path, {}


def prepare_vit_tokens(model, x):
    tokens = model.patch_embed(x)
    if hasattr(model, '_pos_embed'):
        tokens = model._pos_embed(tokens)
    else:
        cls_token = getattr(model, 'cls_token', None)
        if cls_token is not None:
            cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
            dist_token = getattr(model, 'dist_token', None)
            if dist_token is not None:
                tokens = torch.cat((cls_tokens, dist_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
            else:
                tokens = torch.cat((cls_tokens, tokens), dim=1)
        pos_embed = getattr(model, 'pos_embed', None)
        if pos_embed is not None:
            tokens = tokens + pos_embed
        pos_drop = getattr(model, 'pos_drop', None)
        if pos_drop is not None:
            tokens = pos_drop(tokens)

    if hasattr(model, 'patch_drop'):
        tokens = model.patch_drop(tokens)
    if hasattr(model, 'norm_pre'):
        tokens = model.norm_pre(tokens)
    return tokens


def get_block_attention(block, tokens):
    attn_module = getattr(block, 'attn', None)
    if attn_module is None or not hasattr(attn_module, 'qkv'):
        return None

    norm_tokens = block.norm1(tokens) if hasattr(block, 'norm1') else tokens
    qkv = attn_module.qkv(norm_tokens)
    batch_size, token_count, _ = qkv.shape
    num_heads = int(getattr(attn_module, 'num_heads', 0))
    if num_heads <= 0:
        return None

    head_dim = qkv.shape[-1] // (3 * num_heads)
    q, k, _v = qkv.reshape(batch_size, token_count, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    scale = float(getattr(attn_module, 'scale', head_dim ** -0.5))
    return (q * scale @ k.transpose(-2, -1)).softmax(dim=-1)


def rollout_from_attentions(attentions, device):
    if not attentions:
        return None

    token_count = attentions[0].shape[-1]
    eye = torch.eye(token_count, device=device).unsqueeze(0)
    rollout = eye.expand(attentions[0].shape[0], -1, -1).clone()

    for attn in attentions:
        attn_mean = attn.mean(dim=1)
        attn_aug = attn_mean + eye
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        rollout = torch.bmm(attn_aug, rollout)

    return rollout[:, 0, 1:]


def extract_layer_attentions(model, x, layer_indices):
    backbone = getattr(model, 'backbone', model)
    tokens = prepare_vit_tokens(backbone, x)
    selected = {}

    with torch.no_grad():
        for idx, block in enumerate(backbone.blocks, start=1):
            attn = get_block_attention(block, tokens)
            if attn is not None and idx in layer_indices:
                selected[idx] = attn.detach()
            tokens = block(tokens)
    return selected


def get_rollout_score(model, x, rollout_layers=None):
    backbone = getattr(model, 'backbone', model)
    tokens = prepare_vit_tokens(backbone, x)
    attentions = []
    rollout_layers = int(rollout_layers) if rollout_layers is not None else 4

    with torch.no_grad():
        for block in backbone.blocks:
            attn = get_block_attention(block, tokens)
            if attn is not None:
                attentions.append(attn)
            tokens = block(tokens)

    if not attentions:
        raise RuntimeError('Failed to capture attention maps for rollout')
    return rollout_from_attentions(attentions[-rollout_layers:], x.device)


def reshape_attention_vector(attention_vector):
    vector = attention_vector.detach().float().cpu().numpy().reshape(-1)
    patch_count = int(vector.shape[0])
    side = int(round(math.sqrt(patch_count)))
    if side * side != patch_count:
        side = int(math.ceil(math.sqrt(patch_count)))
        pad = side * side - patch_count
        if pad > 0:
            vector = np.concatenate([vector, np.zeros(pad, dtype=vector.dtype)])
    return vector.reshape(side, side)


def min_max_normalize(array):
    array = np.asarray(array, dtype=np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum)


def resize_attention_map(attention_map, size=224):
    return cv2.resize(attention_map, (size, size), interpolation=cv2.INTER_LINEAR)


def sample_random_test_images(dataset, count=1, seed=None):
    rng = random.Random(seed)
    indices = list(range(len(dataset.samples)))
    if count >= len(indices):
        chosen = indices
    else:
        chosen = rng.sample(indices, count)
    return [dataset.samples[i][0] for i in chosen]


def class_markers_and_colors(num_classes):
    markers = ['o', 's', '^', 'v', '<', '>', 'P', 'X', 'D', '*', 'p', 'h', 'H', '1', '2', '3', '4']
    cmap = plt.get_cmap('tab20', max(num_classes, 1))
    colors = [cmap(i) for i in range(num_classes)]
    marker_map = [markers[i % len(markers)] for i in range(num_classes)]
    return marker_map, colors


def tensor_to_numpy_feature(tensor):
    return tensor.detach().cpu().numpy().astype(np.float32)
