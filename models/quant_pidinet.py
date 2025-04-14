"""
Quantized PiDiNet model using Brevitas.
Based on the converted (vanilla CNN) version of PiDiNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant # Example bias quantizer

from .config import config_model_converted # To get PDC types for kernel size logic

# Default bit widths (can be overridden in factory functions or args)
DEFAULT_WEIGHT_BIT_WIDTH = 8
DEFAULT_ACT_BIT_WIDTH = 8

class QuantCSAM(nn.Module):
    """ Quantized Compact Spatial Attention Module """
    def __init__(self, channels, act_bit_width=DEFAULT_ACT_BIT_WIDTH, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantCSAM, self).__init__()
        mid_channels = 4
        # Use QuantIdentity for input activation quantization if needed, or rely on previous layer's output quant
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(channels, mid_channels, kernel_size=1, padding=0,
                                     weight_bit_width=weight_bit_width,
                                     bias_quant=BiasQuant,
                                     cache_inference_quant_bias=True) # Bias default True
        self.conv2 = qnn.QuantConv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False,
                                     weight_bit_width=weight_bit_width)
        self.sigmoid = qnn.QuantSigmoid(bit_width=act_bit_width, return_quant_tensor=True) # Sigmoid activation quantization

    def forward(self, x):
        # Assuming x is already a QuantTensor from a previous layer
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        # Element-wise multiplication should handle QuantTensors correctly
        return x * y

class QuantCDCM(nn.Module):
    """ Quantized Compact Dilation Convolution based Module """
    def __init__(self, in_channels, out_channels, act_bit_width=DEFAULT_ACT_BIT_WIDTH, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantCDCM, self).__init__()
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=1, padding=0,
                                     weight_bit_width=weight_bit_width,
                                     bias_quant=BiasQuant,
                                     cache_inference_quant_bias=True)
        # Using standard Conv2d for dilated as Brevitas might not directly support quantized dilated conv easily,
        # or need specific setup. For simplicity, keeping these standard for now.
        # If full quantization is needed here, these need replacement and careful handling.
        # Alternatively, use qnn.QuantConv2d if supported and tested for dilation.
        # Let's try qnn.QuantConv2d assuming it works.
        self.conv2_1 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False, weight_bit_width=weight_bit_width)
        self.conv2_2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False, weight_bit_width=weight_bit_width)
        self.conv2_3 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False, weight_bit_width=weight_bit_width)
        self.conv2_4 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False, weight_bit_width=weight_bit_width)
        # Addition might require explicit requantization if inputs have different scales
        self.requant_add = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)


    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        # Summing QuantTensors: Brevitas typically handles this, but explicit requantization might be safer
        return self.requant_add(x1 + x2 + x3 + x4)


class QuantMapReduce(nn.Module):
    """ Quantized Reduce feature maps into a single edge map """
    def __init__(self, channels, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantMapReduce, self).__init__()
        self.conv = qnn.QuantConv2d(channels, 1, kernel_size=1, padding=0,
                                    weight_bit_width=weight_bit_width,
                                    bias_quant=BiasQuant,
                                    cache_inference_quant_bias=True) # Bias default True

    def forward(self, x):
        # Output of this might need sigmoid and quantization, handled later
        return self.conv(x)


class QuantPDCBlock(nn.Module):
    """ Quantized PDCBlock based on converted structure """
    def __init__(self, pdc_type, inplane, ouplane, stride=1, act_bit_width=DEFAULT_ACT_BIT_WIDTH, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantPDCBlock, self).__init__()
        self.stride = stride
        self.act_bit_width = act_bit_width # Store for requantization

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            # Add quantizer after pooling
            self.quant_pool = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
            self.shortcut = qnn.QuantConv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False,
                                            weight_bit_width=weight_bit_width)
        else:
            # If inplane != ouplane, need a shortcut, but original doesn't seem to have it for stride=1
             if inplane != ouplane:
                 self.shortcut = qnn.QuantConv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False,
                                                 weight_bit_width=weight_bit_width)
             else:
                 self.shortcut = qnn.QuantIdentity(return_quant_tensor=True) # Identity for skip connection


        # Determine kernel size based on converted PDC type
        if pdc_type == 'rd':
            conv1_kernel_size = 5
            conv1_padding = 2
        else: # 'cv', 'cd', 'ad' convert to 3x3
            conv1_kernel_size = 3
            conv1_padding = 1

        self.conv1 = qnn.QuantConv2d(inplane, inplane, kernel_size=conv1_kernel_size, padding=conv1_padding, groups=inplane, bias=False,
                                     weight_bit_width=weight_bit_width)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False,
                                     weight_bit_width=weight_bit_width)
        # Addition requantization - Temporarily remove for testing
        # self.requant_add = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)

    def forward(self, x):
        # Input x assumed to be QuantTensor
        identity = x
        if self.stride > 1:
            x_pooled = self.pool(x.value) # Pool the float value
            x = self.quant_pool(x_pooled) # Requantize the result
            # identity = self.shortcut(x) # Shortcut now operates on QuantTensor - Not needed for this test
        # elif hasattr(self, 'shortcut'): # Not needed for this test
             # identity = self.shortcut(identity) # Apply shortcut if it exists (e.g., channel change)


        y = self.conv1(x) # conv1 now always receives QuantTensor
        y = self.relu2(y)
        y = self.conv2(y)

        # Temporarily remove residual connection for testing
        # out = self.requant_add(y + identity)
        out = y # Just return the output of conv2
        return out


class QuantPiDiNet(nn.Module):
    def __init__(self, inplane, pdcs_types, dil=None, sa=False, act_bit_width=DEFAULT_ACT_BIT_WIDTH, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantPiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []

        # Input Quantization
        self.quant_inp = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)

        # Initial Block (based on converted type)
        init_pdc_type = pdcs_types[0]
        if init_pdc_type == 'rd':
            init_kernel_size = 5
            init_padding = 2
        else:
            init_kernel_size = 3
            init_padding = 1
        self.init_block = qnn.QuantConv2d(3, inplane,
                                          kernel_size=init_kernel_size, padding=init_padding, bias=False,
                                          weight_bit_width=weight_bit_width)
        self.fuseplanes.append(inplane) # C

        # Define Blocks
        block_class = QuantPDCBlock
        cur_inplane = inplane
        self.block1_1 = block_class(pdcs_types[1], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block1_2 = block_class(pdcs_types[2], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block1_3 = block_class(pdcs_types[3], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)

        cur_inplane = inplane * 2
        self.block2_1 = block_class(pdcs_types[4], inplane, cur_inplane, stride=2, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block2_2 = block_class(pdcs_types[5], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block2_3 = block_class(pdcs_types[6], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block2_4 = block_class(pdcs_types[7], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.fuseplanes.append(cur_inplane) # 2C

        inplane = cur_inplane
        cur_inplane = cur_inplane * 2
        self.block3_1 = block_class(pdcs_types[8], inplane, cur_inplane, stride=2, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block3_2 = block_class(pdcs_types[9], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block3_3 = block_class(pdcs_types[10], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block3_4 = block_class(pdcs_types[11], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.fuseplanes.append(cur_inplane) # 4C

        inplane = cur_inplane
        # Note: Original PiDiNet keeps 4C here, check if typo in original or intended
        # Assuming it stays 4C based on original code structure
        self.block4_1 = block_class(pdcs_types[12], inplane, cur_inplane, stride=2, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block4_2 = block_class(pdcs_types[13], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block4_3 = block_class(pdcs_types[14], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block4_4 = block_class(pdcs_types[15], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.fuseplanes.append(cur_inplane) # 4C

        # Fusion Layers (Quantized)
        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(QuantCDCM(self.fuseplanes[i], self.dil, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width))
                self.attentions.append(QuantCSAM(self.dil, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width))
                self.conv_reduces.append(QuantMapReduce(self.dil, weight_bit_width=weight_bit_width))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(QuantCSAM(self.fuseplanes[i], act_bit_width=act_bit_width, weight_bit_width=weight_bit_width))
                self.conv_reduces.append(QuantMapReduce(self.fuseplanes[i], weight_bit_width=weight_bit_width))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(QuantCDCM(self.fuseplanes[i], self.dil, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width))
                self.conv_reduces.append(QuantMapReduce(self.dil, weight_bit_width=weight_bit_width))
        else:
            for i in range(4):
                self.conv_reduces.append(QuantMapReduce(self.fuseplanes[i], weight_bit_width=weight_bit_width))

        # Add QuantIdentity layer to handle concatenation before classifier
        self.quant_cat = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)

        # Final Classifier
        self.classifier = qnn.QuantConv2d(4, 1, kernel_size=1,
                                          weight_bit_width=weight_bit_width,
                                          bias_quant=BiasQuant,
                                          cache_inference_quant_bias=True) # Bias default True
        # Initialize classifier weights and bias if needed, similar to original
        # nn.init.constant_(self.classifier.weight, 0.25) # May need adjustment for QuantTensor
        # nn.init.constant_(self.classifier.bias, 0)

        # Final Sigmoid Activation Quantization
        self.final_sigmoid = qnn.QuantSigmoid(bit_width=act_bit_width, return_quant_tensor=False) # Return float for loss calculation

        print('Quantized PiDiNet initialization done')

    def forward(self, x):
        H, W = x.size()[2:]

        # Quantize input
        x = self.quant_inp(x)

        # Blocks
        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        # Feature Fusion / Attention / Dilation
        x_fuses = []
        if self.sa and self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](self.dilations[i](xi)))
        elif self.sa:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.attentions[i](xi))
        elif self.dil is not None:
            for i, xi in enumerate([x1, x2, x3, x4]):
                x_fuses.append(self.dilations[i](xi))
        else:
            x_fuses = [x1, x2, x3, x4]

        # Reduce and Upsample
        e1 = self.conv_reduces[0](x_fuses[0])
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)

        e2 = self.conv_reduces[1](x_fuses[1])
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3 = F.interpolate(e3, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4 = F.interpolate(e4, (H, W), mode="bilinear", align_corners=False)

        # Classifier and Final Output
        # Concatenate the interpolated (standard) tensors
        cat_out = torch.cat([e1, e2, e3, e4], dim=1)
        # Requantize the concatenated tensor before feeding to the classifier
        quant_cat_out = self.quant_cat(cat_out)
        output = self.classifier(quant_cat_out)

        # Apply final sigmoid quantization
        # The intermediate outputs e1..e4 might also need sigmoid for training loss
        # Note: e1..e4 are standard tensors here after interpolation
        outputs = [self.final_sigmoid(e) for e in [e1, e2, e3, e4]]
        outputs.append(self.final_sigmoid(output)) # output is already processed by final_sigmoid

        # Return float tensors for compatibility with existing loss functions
        # Brevitas QuantTensor can be used directly if loss supports it,
        # otherwise .value extracts the dequantized float tensor.
        # Setting return_quant_tensor=False in final sigmoid handles this.
        return outputs


# Factory functions for different sizes
def quant_pidinet_micro(args, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH, act_bit_width=DEFAULT_ACT_BIT_WIDTH):
    pdcs_types = config_model_converted(args.config) # Get pdc types ('cv', 'cd', 'ad', 'rd')
    dil = 4 if args.dil else None
    return QuantPiDiNet(12, pdcs_types, dil=dil, sa=args.sa, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

def quant_pidinet_tiny(args, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH, act_bit_width=DEFAULT_ACT_BIT_WIDTH):
    pdcs_types = config_model_converted(args.config)
    dil = 8 if args.dil else None
    return QuantPiDiNet(20, pdcs_types, dil=dil, sa=args.sa, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

def quant_pidinet_small(args, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH, act_bit_width=DEFAULT_ACT_BIT_WIDTH):
    pdcs_types = config_model_converted(args.config)
    dil = 12 if args.dil else None
    return QuantPiDiNet(30, pdcs_types, dil=dil, sa=args.sa, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

def quant_pidinet(args, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH, act_bit_width=DEFAULT_ACT_BIT_WIDTH):
    pdcs_types = config_model_converted(args.config)
    dil = 24 if args.dil else None
    return QuantPiDiNet(60, pdcs_types, dil=dil, sa=args.sa, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

