import os
import sys
from functools import partial

import torch.hub
from efficientnet_pytorch import EfficientNet
from timm.models.efficientnet import tf_efficientnet_b5_ns, tf_efficientnet_b3_ns, tf_efficientnet_b4_ns, \
    tf_efficientnet_b7_ns, tf_efficientnet_b6_ns, tf_efficientnet_b7_ap, tf_efficientnet_b6_ap, tf_efficientnet_b8
from timm.models.resnet import seresnext26tn_32x4d, ecaresnet50d, swsl_resnext101_32x8d, ecaresnet101d
from torch.nn import Dropout2d, UpsamplingBilinear2d, Sequential, ModuleList, BatchNorm2d, ReLU
from torch.utils import model_zoo

from zoo.dpn import dpn92
from zoo.senet import SCSEModule, senet154, se_resnext50_32x4d



encoder_params = {
    'dpn92':
        {'filters': [64, 336, 704, 1552, 2688],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': dpn92,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth'},
    'se26':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': partial(seresnext26tn_32x4d, pretrained=True),
         'url': None},
    'senet154':
        {'filters': [128, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': partial(senet154, pretrained="imagenet"),
         'url': None},
    'r101':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 192, 256],
         'last_upsample': 64,
         'init_op': partial(swsl_resnext101_32x8d, pretrained=True),
         'url': None},
    'se50':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': partial(se_resnext50_32x4d, pretrained="imagenet"),
         'url': None},
    'eca50':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': partial(ecaresnet50d, pretrained=True),
         'url': None},
    'ecaresnet101d':
        {'filters': [64, 256, 512, 1024, 2048],
         'decoder_filters': [64, 128, 256, 256],
         'last_upsample': 64,
         'init_op': partial(ecaresnet101d, pretrained=True),
         'url': None},
    "tf_efficientnet_b3_ns": {
        'last_upsample': 64,
        "filters": [40, 32, 48, 136, 1536],
        "decoder_filters": [64, 128, 256, 256],
        "init_op": partial(tf_efficientnet_b3_ns, pretrained=True, drop_path_rate=0.2),
        'url': None,
    },
    "tf_efficientnet_b4_ns": {
        'last_upsample': 64,
        "filters": [48, 32, 56, 160, 1792],
        "decoder_filters": [64, 128, 256, 256],
        "init_op": partial(tf_efficientnet_b4_ns, pretrained=True, drop_path_rate=0.2),
        'url': None,
    },
    "tf_efficientnet_b5_ns": {
        'last_upsample': 64,
        "filters": [48, 40, 64, 176, 2048],
        "decoder_filters": [64, 128, 176, 256],
        "init_op": partial(tf_efficientnet_b5_ns, pretrained=True, drop_path_rate=0.1),
        'url': None,
    },
    "tf_efficientnet_b7_ns": {
        'last_upsample': 64,
        "filters": [64, 48, 80, 224, 2560],
        "decoder_filters": [64, 128, 256, 256],
        "init_op": partial(tf_efficientnet_b7_ns, pretrained=True, drop_path_rate=0.2),
        'url': None,
    },
    "tf_efficientnet_b7_ap": {
        'last_upsample': 64,
        "filters": [64, 48, 80, 224, 2560],
        "decoder_filters": [64, 128, 256, 256],
        "init_op": partial(tf_efficientnet_b7_ap, pretrained=True, drop_path_rate=0.2),
        'url': None,
    },
    "tf_efficientnet_b6_ns": {
        'last_upsample': 64,
        "filters": [56, 40, 72, 200, 2304],
        "decoder_filters": [64, 128, 200, 256],
        "init_op": partial(tf_efficientnet_b6_ns, pretrained=True, drop_path_rate=0.2),
        'url': None,
    },
    "tf_efficientnet_b6_ap": {
        'last_upsample': 64,
        "filters": [56, 40, 72, 200, 2304],
        "decoder_filters": [64, 128, 200, 256],
        "init_op": partial(tf_efficientnet_b6_ap, pretrained=True, drop_path_rate=0.2),
        'url': None,
    },
    "tf_efficientnet_b8": {
        'last_upsample': 64,
        "filters": [72, 56, 88, 248, 2816],
        "decoder_filters": [64, 128, 200, 256],
        "init_op": partial(tf_efficientnet_b8, pretrained=True, drop_path_rate=0.2),
        'url': None,
    },
    "efficientnet_b7": {
        'last_upsample': 64,
        "filters": [64, 48, 80, 224, 2560],
        "decoder_filters": [64, 128, 256, 256],
        "init_op": partial(EfficientNet.from_pretrained, "efficientnet-b7", advprop=True),
        "stage_idxs": (11, 18, 38, 55),
        'url': None,
    },
    "efficientnet_b5": {
        'last_upsample': 64,
        "filters": [48, 40, 64, 176, 2048],
        "decoder_filters": [64, 128, 176, 256],
        "init_op": partial(EfficientNet.from_pretrained, "efficientnet-b5"),
        "stage_idxs": (8, 13, 27, 39),
        'url': None,
    },
}

import torch
from torch import nn
import torch.nn.functional as F


class BasicConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, activation=nn.ReLU, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                            bias=bias)
        self.use_act = activation is not None
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class Conv1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=None, bias=bias)


class Conv3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=None)


class ConvReLu1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=nn.ReLU)


class ConvReLu3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=nn.ReLU)


class BasicUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.ReLU, mode='nearest'):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * 1
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1)
        self.use_act = activation is not None
        self.mode = mode
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode=self.mode)
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_names[0] + '.weight'][:, :3, ...] = pretrained_dict[
                self.first_layer_params_names[0] + '.weight'].data
            skip_layers = self.first_layer_params_names
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               not any(k.startswith(s) for s in skip_layers)}
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_names(self):
        return ['conv1.conv']


class MultiHead(nn.Module):
    def __init__(self, in_filters: int, num_classes: list):
        super().__init__()
        self.heads = ModuleList([nn.Conv2d(in_filters, classes, 1, padding=0) for classes in num_classes])

    def forward(self, x):
        return [head(x) for head in self.heads]


class MultiHeadSeparate(nn.Module):
    def __init__(self, in_filters: int, num_classes: list, num_filters: int = 64):
        super().__init__()
        self.heads = ModuleList([
            Sequential(
                nn.Conv2d(in_filters, num_filters, 3, padding=1),
                ReLU(),
                nn.Conv2d(num_filters, classes, 1, padding=0)
            )
            for classes in num_classes])

    def forward(self, xs):
        return [head(x) for head, x in zip(self.heads, xs)]


class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34', use_last_decoder=False):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        self.filters = encoder_params[encoder_name]['filters']
        self.decoder_filters = encoder_params[encoder_name].get('decoder_filters', self.filters[:-1])
        self.last_upsample_filters = encoder_params[encoder_name].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])

        if self.first_layer_stride_two:
            if use_last_decoder:
                self.last_upsample = self.decoder_block(self.decoder_filters[0], self.last_upsample_filters,
                                                        self.last_upsample_filters)
            else:
                self.last_upsample = UpsamplingBilinear2d(scale_factor=2)
        self.final = self.make_final_classifier(
            self.last_upsample_filters if self.first_layer_stride_two else self.decoder_filters[0], num_classes)
        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        encoder = encoder_params[encoder_name]['init_op']()
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x, **kwargs):
        # Encoder
        enc_results = []
        for stage in self.encoder_stages:
            #            x = self.dropout(x)
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
        # if self.use_bilinear_4x:
        x = self.dropout(x)

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)
        return f

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def make_final_classifier(self, in_filters, num_classes):
        if isinstance(num_classes, int):
            return nn.Sequential(
                nn.Conv2d(in_filters, num_classes, 1, padding=0)
            )
        elif isinstance(num_classes, list):
            return MultiHead(in_filters, num_classes)
        else:
            raise ValueError("unknown numclasses type: " + type(num_classes))

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def first_layer_params_names(self):
        raise NotImplementedError

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [self.bottlenecks, self.decoder_stages, self.final]
        return _get_layers_params(layers)


def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class SCSeResneXt(EncoderDecoder):

    def __init__(self, seg_classes, backbone_arch, reduction=2, mode='concat'):
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = partial(ConvSCSEBottleneckNoBn, reduction=reduction, mode=mode)
        self.first_layer_stride_two = True
        self.concat_scse = mode == 'concat'

        super().__init__(seg_classes, 3, backbone_arch)
        self.last_upsample = self.decoder_block(
            self.decoder_filters[0] * 2 if self.concat_scse else self.decoder_filters[0],
            self.last_upsample_filters,
            self.last_upsample_filters)

    def calc_dec_filters(self, d_filters):
        return d_filters * 2 if self.concat_scse else d_filters

    def forward(self, x, **kwargs):
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(x.clone())
        dec_results = []

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
            dec_results.append(x)

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        mask = self.final(x)
        return mask

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        if self.concat_scse and layer + 1 < len(self.decoder_filters):
            in_channels *= 2

        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.layer0
        elif layer == 1:
            return nn.Sequential(
                encoder.pool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class ConvSCSEBottleneckNoBn(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, mode='concat'):
        print("bottleneck ", in_channels, out_channels)
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            SCSEModule(out_channels, reduction=reduction, mode=mode)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class DPNUnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='dpn92', use_last_decoder=True):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 3, backbone_arch, use_last_decoder)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.blocks['conv1_1'].conv,  # conv
                encoder.blocks['conv1_1'].bn,  # bn
                encoder.blocks['conv1_1'].act,  # relu
            )
        elif layer == 1:
            return nn.Sequential(
                encoder.blocks['conv1_1'].pool,  # maxpool
                *[b for k, b in encoder.blocks.items() if k.startswith('conv2_')]
            )
        elif layer == 2:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv3_')])
        elif layer == 3:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv4_')])
        elif layer == 4:
            return nn.Sequential(*[b for k, b in encoder.blocks.items() if k.startswith('conv5_')])

    @property
    def first_layer_params_names(self):
        return ['features.conv1_1.conv']


class SEUnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='senet154'):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return encoder.layer0
        elif layer == 1:
            return nn.Sequential(
                encoder.pool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class TimmResnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch, use_last_decoder=True):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 3, backbone_arch, use_last_decoder)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.act1)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4


class TimmEffnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch, use_last_decoder=True):
        self.first_layer_stride_two = True
        super().__init__(seg_classes, 3, backbone_arch, use_last_decoder)


    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv_stem,
                encoder.bn1,
                encoder.act1
            )
        elif layer == 1:
            return nn.Sequential(encoder.blocks[:2])
        elif layer == 2:
            return nn.Sequential(encoder.blocks[2:3])
        elif layer == 3:
            return nn.Sequential(encoder.blocks[3:5])
        elif layer == 4:
            return nn.Sequential(
                *encoder.blocks[5:],
                encoder.conv_head,
                encoder.bn2,
                encoder.act2
            )



class SCSEDragon(SCSeResneXt):
    def __init__(self, seg_classes, backbone_arch, **kwargs):
        super().__init__(seg_classes, backbone_arch)
        self.final = self.make_final_classifier(in_filters=self.last_upsample_filters, num_classes=seg_classes)
        self.last_upsample = ModuleList([self.decoder_block(self.decoder_filters[0] * 2, self.last_upsample_filters,
                                                self.last_upsample_filters) for _ in seg_classes])

    def make_final_classifier(self, in_filters, num_classes):
        return MultiHeadSeparate(in_filters, num_classes, self.last_upsample_filters)

    def forward(self, x, month=None):
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(x.clone())
        dec_results = []

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
            dec_results.append(x)

        xs = [m(x) for m in self.last_upsample]

        f = self.final(xs)
        return f

class TimmEffnetDragon(TimmEffnet):
    def __init__(self, seg_classes, backbone_arch, use_last_decoder=False):
        super().__init__(seg_classes, backbone_arch, use_last_decoder)
        self.final = self.make_final_classifier(in_filters=self.last_upsample_filters, num_classes=seg_classes)
        self.last_upsample = ModuleList([self.decoder_block(self.decoder_filters[0], self.last_upsample_filters,
                                                self.last_upsample_filters) for _ in seg_classes])
    def make_final_classifier(self, in_filters, num_classes):
        return MultiHeadSeparate(in_filters, num_classes, self.last_upsample_filters)

    def forward(self, x, month=None):
        # Encoder
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        x = self.dropout(x)
        xs = [m(x) for m in self.last_upsample]

        f = self.final(xs)
        return f

class TimmResnetDragon(TimmResnet):
    def __init__(self, seg_classes, backbone_arch, use_last_decoder=True):
        super().__init__(seg_classes, backbone_arch, use_last_decoder)
        self.final = self.make_final_classifier(in_filters=self.last_upsample_filters, num_classes=seg_classes)
        self.last_upsample = ModuleList([self.decoder_block(self.decoder_filters[0], self.last_upsample_filters,
                                                self.last_upsample_filters) for _ in seg_classes])
    def make_final_classifier(self, in_filters, num_classes):
        return MultiHeadSeparate(in_filters, num_classes, self.last_upsample_filters)

    def forward(self, x, month=None):
        # Encoder
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        x = self.dropout(x)
        xs = [m(x) for m in self.last_upsample]

        f = self.final(xs)
        return f

class DPNDragon(DPNUnet):
    def __init__(self, seg_classes, backbone_arch, use_last_decoder=True):
        super().__init__(seg_classes, backbone_arch, use_last_decoder)
        self.final = self.make_final_classifier(in_filters=self.last_upsample_filters, num_classes=seg_classes)
        self.last_upsample = ModuleList([self.decoder_block(self.decoder_filters[0], self.last_upsample_filters,
                                                self.last_upsample_filters) for _ in seg_classes])
    def make_final_classifier(self, in_filters, num_classes):
        return MultiHeadSeparate(in_filters, num_classes, self.last_upsample_filters)

    def forward(self, x, month=None):
        # Encoder
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())

        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        x = self.dropout(x)
        xs = [m(x) for m in self.last_upsample]

        f = self.final(xs)
        return f

class TimmEffnetDragonSiamese(TimmEffnet):
    def __init__(self, seg_classes, backbone_arch, use_last_decoder=False):
        super().__init__(seg_classes, backbone_arch, use_last_decoder)
        self.final = self.make_final_classifier(in_filters=self.last_upsample_filters, num_classes=seg_classes)
        self.last_upsample = ModuleList([self.decoder_block(self.decoder_filters[0], self.last_upsample_filters,
                                                self.last_upsample_filters) for _ in seg_classes])
        self.discriminators = ModuleList([
            Sequential(ConvBNReLU(f * 2, f, kernel_size=1),
                       ConvBNReLU(f, f, kernel_size=1)
                       ) for i, f in enumerate(self.filters)
        ])
    def make_final_classifier(self, in_filters, num_classes):
        return MultiHeadSeparate(in_filters, num_classes, self.last_upsample_filters)

    def forward(self, x1, x2):
        # Encoder
        enc_results = []
        for i, stage in enumerate(self.encoder_stages):
            x1 = stage(x1)
            x1 = torch.cat(x1, dim=1) if isinstance(x1, tuple) else x1.clone()
            x2 = stage(x2)
            x2 = torch.cat(x2, dim=1) if isinstance(x2, tuple) else x2.clone()
            enc_results.append(self.discriminators[i](torch.cat([x1, x2], dim=1)))
        x = enc_results[-1]
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        xs = [m(x) for m in self.last_upsample]

        f = self.final(xs)
        return f

class EfficientUnet(EncoderDecoder):
    def __init__(self, seg_classes, backbone_arch='efficientnet-b7'):
        self.first_layer_stride_two = True
        self._stage_idxs = encoder_params[backbone_arch]['stage_idxs']
        super().__init__(seg_classes, 3, backbone_arch)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(encoder._conv_stem, encoder._bn0, encoder._swish)
        elif layer == 1:
            return Sequential(*encoder._blocks[:self._stage_idxs[0]])
        elif layer == 2:
            return Sequential(*encoder._blocks[self._stage_idxs[0]:self._stage_idxs[1]])
        elif layer == 3:
            return Sequential(*encoder._blocks[self._stage_idxs[1]:self._stage_idxs[2]])
        elif layer == 4:
            return Sequential(*encoder._blocks[self._stage_idxs[2]:], encoder._conv_head, encoder._bn1, encoder._swish)

    def forward(self, x):
        # Encoder
        enc_results = []
        block_idx = 0
        drop_connect_rate = 0.2
        for i, stage in enumerate(self.encoder_stages):
            if i > 0:
                for block in stage:
                    block_idx += 1
                    drop_connect_rate *= float(block_idx) / self._stage_idxs[-1]
                    if block_idx <= self._stage_idxs[-1]:
                        x = block(x, drop_connect_rate=drop_connect_rate)
                    else:
                        x = block(x)
            else:
                x = stage(x)
            enc_results.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x.clone())
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])
        # if self.use_bilinear_4x:
        x = self.dropout(x)

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)
        return f


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=BatchNorm2d):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

setattr(sys.modules[__name__], 'scse_unet', partial(SCSeResneXt))
setattr(sys.modules[__name__], 'scse_dragon', partial(SCSEDragon))
setattr(sys.modules[__name__], 'dpn_unet', partial(DPNUnet))
setattr(sys.modules[__name__], 'timm_resnet', partial(TimmResnet))
setattr(sys.modules[__name__], 'timm_effnet', partial(TimmEffnet))
setattr(sys.modules[__name__], 'timm_effnet_dragon', partial(TimmEffnetDragon))
setattr(sys.modules[__name__], 'timm_resnet_dragon', partial(TimmResnetDragon))
setattr(sys.modules[__name__], 'dpn_dragon', partial(DPNDragon))
setattr(sys.modules[__name__], 'timm_effnet_dragon_siamese', partial(TimmEffnetDragonSiamese))
setattr(sys.modules[__name__], 'effnet', partial(EfficientUnet))

__all__ = ['scse_unet',
           'timm_resnet',
           'timm_effnet',
           'timm_effnet_dragon',
           'timm_resnet_dragon',
           'timm_effnet_dragon_siamese',
           'dpn_unet',
           'dpn_dragon',
           'effnet',
           'scse_dragon',
           ]