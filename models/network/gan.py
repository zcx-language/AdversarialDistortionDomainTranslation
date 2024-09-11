#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : AdversarialDistortionTranslation
# @File         : gan_utils.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2022/5/10 下午8:28

# Import lib here
import random
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
# from fastai.layers import ConvLayer
# from fastai.vision.models.unet import DynamicUnet
from models.network.basic_blocks import ConvBNRelu
from typing import Tuple, List


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
            #                              kernel_size=3, stride=2,
            #                              padding=1, output_padding=1,
            #                              bias=use_bias),
            #           norm_layer(int(ngf * mult / 2)),
            #           nn.ReLU(True)]
            model += [nn.Upsample(scale_factor=2, mode='nearest'),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)
            upconv = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=0),
            ]
            down = [downconv]
            up = [uprelu, *upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1, bias=use_bias)
            upconv = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=0, bias=use_bias),
            ]
            down = [downrelu, downconv]
            up = [uprelu, *upconv, upnorm]
            model = down + up
        else:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1, bias=use_bias)
            upconv = [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=0, bias=use_bias),
            ]
            down = [downrelu, downconv, downnorm]
            up = [uprelu, *upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class DefaultUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False):
        super().__init__()

        self.conv1 = ConvBNRelu(in_channels, 32, 3)
        self.conv2 = ConvBNRelu(32, 32, 3, stride=2)
        self.conv3 = ConvBNRelu(32, 64, 3, stride=2)
        self.conv4 = ConvBNRelu(64, 128, 3, stride=2)
        self.conv5 = ConvBNRelu(128, 256, 3, stride=2)
        self.up6 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBNRelu(256, 128, 3)
        )
        self.conv6 = ConvBNRelu(256, 128, 3)
        self.up7 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBNRelu(128, 64, 3)
        )
        self.conv7 = ConvBNRelu(128, 64, 3)
        self.up8 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBNRelu(64, 32, 3)
        )
        self.conv8 = ConvBNRelu(64, 32, 3)
        self.up9 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBNRelu(32, 32, 3)
        )
        self.conv9 = ConvBNRelu(64, 32, 3)
        self.last_conv = nn.Conv2d(32, out_channels, 1)

        # Use F.dropout instead of nn.Dropout, so that the dropout still works during evaluation.
        self.use_dropout = use_dropout

    def forward(self, inputs: torch.Tensor):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = F.dropout(self.up6(conv5), p=0.5, training=self.use_dropout)
        conv6 = self.conv6(torch.cat([conv4, up6], dim=1))
        up7 = F.dropout(self.up7(conv6), p=0.5, training=self.use_dropout)
        conv7 = self.conv7(torch.cat([conv3, up7], dim=1))
        up8 = F.dropout(self.up8(conv7), p=0.5, training=self.use_dropout)
        conv8 = self.conv8(torch.cat([conv2, up8], dim=1))
        up9 = F.dropout(self.up9(conv8), p=0.5, training=self.use_dropout)
        conv9 = self.conv9(torch.cat([conv1, up9], dim=1))
        return self.last_conv(conv9).clamp(-1, 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class ImagePool:
    """
    This class implements an image buffer that stores previously generated images! This buffer enables to update
    discriminators using a history of generated image rather than the latest ones produced by generator.
    """

    def __init__(self, pool_sz: int = 50):

        """
        Parameters:
            pool_sz: Size of the image buffer
        """

        self.nb_images = 0
        self.image_pool = []
        self.pool_sz = pool_sz

    def push_and_pop(self, images):

        """
        Parameters:
            images: latest images generated by the generator
        Returns a batch of images from pool!
        """
        if self.pool_sz == 0:
            return images

        images_to_return = []
        for image in images:
            image = torch.unsqueeze(image, 0)

            if self.nb_images < self.pool_sz:
                self.image_pool.append(image)
                images_to_return.append(image)
                self.nb_images += 1
            else:
                if random.uniform(0, 1) > 0.5:
                    rand_int = random.randint(0, self.pool_sz-1)
                    temp_img = self.image_pool[rand_int].clone()
                    self.image_pool[rand_int] = image
                    images_to_return.append(temp_img)
                else:
                    images_to_return.append(image)

        return torch.cat(images_to_return, 0)


def cal_gradient_penalty(discriminator, real_data, fake_data, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        discriminator (network)     -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand_like(real_data)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = discriminator(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones_like(disc_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.reshape(real_data.shape[0], -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class DiscriminatorLoss(nn.Module):
    def __init__(self, input_nc: int, loss_mode: str = 'lsgan', image_pool_size: int = 50):
        """Discriminator loss for GAN

        Parameters:
            input_nc: input channel number
            loss_mode: loss mode, support vanilla, lsgan, and wgan-gp
            image_pool_size: image pool size
        """
        super().__init__()
        self.discriminator = NLayerDiscriminator(input_nc=input_nc)
        self.image_pool = ImagePool(image_pool_size)
        self.loss_mode = loss_mode
        self.optimizer = torch.optim.Adam(self.discriminator.parameters())

    def _update_discriminator(self, fake_data: torch.Tensor, real_data: torch.Tensor) -> float:
        fake_data = self.image_pool.push_and_pop(fake_data.detach())
        fake_logit = self.discriminator(fake_data)
        real_logit = self.discriminator(real_data)

        if self.loss_mode == 'vanilla':
            loss = F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit)) + \
                     F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
        elif self.loss_mode == 'lsgan':
            loss = F.mse_loss(fake_logit, torch.zeros_like(fake_logit)) + \
                        F.mse_loss(real_logit, torch.ones_like(real_logit))
        elif self.loss_mode == 'wgan-gp':
            gp, _ = cal_gradient_penalty(self.discriminator, real_data, fake_data)
            loss = torch.mean(fake_logit) - torch.mean(real_logit) + gp
        else:
            raise NotImplementedError('{} not implemented'.format(self.loss_mode))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def forward(self, fake_data: torch.Tensor, real_data: torch.Tensor,
                update_discriminator_frequency: int = 1,
                return_discriminator_loss: bool = False):
        dis_loss = self._update_discriminator(fake_data, real_data)
        for _ in range(update_discriminator_frequency-1):
            dis_loss = self._update_discriminator(fake_data, real_data)

        fake_logit = self.discriminator(fake_data)
        if self.loss_mode == 'vanilla':
            gen_loss = F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
        elif self.loss_mode == 'lsgan':
            gen_loss = F.mse_loss(fake_logit, torch.ones_like(fake_logit))
        elif self.loss_mode == 'wgan-gp':
            gen_loss = -torch.mean(fake_logit)
        else:
            raise NotImplementedError('{} not implemented'.format(self.loss_mode))

        if return_discriminator_loss:
            return gen_loss, dis_loss
        return gen_loss


def run():
    from torchinfo import summary
    # generator = DefaultGenerator(image_shape=(3, 128, 128), out_channels=3)
    # summary(generator, (4, 3, 128, 128))
    pass


if __name__ == '__main__':
    run()
