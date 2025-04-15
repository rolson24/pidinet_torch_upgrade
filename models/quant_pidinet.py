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
        self.relu1 = nn.ReLU()
        # Remove QuantIdentity after ReLU
        # self.quant_relu_out = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(channels, mid_channels, kernel_size=1, padding=0,
                                     weight_bit_width=weight_bit_width,
                                     bias=True, # Ensure bias parameter exists
                                     bias_quant=None, # Keep bias quantization disabled for this test
                                     cache_inference_quant_bias=False)
        if self.conv1.bias is not None:
             nn.init.constant_(self.conv1.bias, 0)

        self.conv2 = qnn.QuantConv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False,
                                     weight_bit_width=weight_bit_width)
        self.sigmoid = nn.Sigmoid() # Keep standard Sigmoid

    def forward(self, x): # Input x: QuantTensor
        # Apply standard ReLU -> float Tensor
        y_float = self.relu1(x)
        # Pass float Tensor directly to conv1
        y = self.conv1(y_float) # y: QuantTensor (conv1 handles float input)
        y = self.conv2(y) # y: QuantTensor
        # Apply standard Sigmoid to the float value of y
        y_sigmoid_float = self.sigmoid(y)
        # Multiply original QuantTensor x by float sigmoid output
        return x * y_sigmoid_float

class QuantCDCM(nn.Module):
    """ Quantized Compact Dilation Convolution based Module """
    def __init__(self, in_channels, out_channels, act_bit_width=DEFAULT_ACT_BIT_WIDTH, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantCDCM, self).__init__()
        # Replace QuantReLU with nn.ReLU
        self.relu1 = nn.ReLU()
        self.conv1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=1, padding=0,
                                     weight_bit_width=weight_bit_width,
                                     bias=True, # Ensure bias exists
                                     bias_quant=None, # Disable bias quantization
                                     cache_inference_quant_bias=False)
        # Initialize conv1 bias to 0
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)

        # Keep dilated convs as QuantConv2d (bias=False)
        self.conv2_1 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False, weight_bit_width=weight_bit_width)
        self.conv2_2 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False, weight_bit_width=weight_bit_width)
        self.conv2_3 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False, weight_bit_width=weight_bit_width)
        self.conv2_4 = qnn.QuantConv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False, weight_bit_width=weight_bit_width)
        # Remove requant_add layer
        # self.requant_add = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)


    def forward(self, x): # Input x: QuantTensor
        # Apply standard ReLU -> float Tensor
        x_float = self.relu1(x)
        # Pass float Tensor directly to conv1
        x = self.conv1(x_float) # x: QuantTensor
        # Pass QuantTensor to dilated convs
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        # Sum QuantTensors directly
        return x1 + x2 + x3 + x4 # Output: QuantTensor

class QuantMapReduce(nn.Module):
    """ Quantized Reduce feature maps into a single edge map """
    def __init__(self, channels, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantMapReduce, self).__init__()
        self.conv = qnn.QuantConv2d(channels, 1, kernel_size=1, padding=0,
                                    weight_bit_width=weight_bit_width,
                                    bias=True, # Ensure bias exists
                                    bias_quant=None, # Disable bias quantization
                                    cache_inference_quant_bias=False)
        # Initialize bias to 0, matching original MapReduce
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x): # Input x: QuantTensor or float Tensor
        # conv handles QuantTensor or float Tensor input
        # Output will be QuantTensor
        return self.conv(x)


class QuantPDCBlock(nn.Module):
    """ Quantized PDCBlock based on converted structure """
    def __init__(self, pdc_type, inplane, ouplane, stride=1, act_bit_width=DEFAULT_ACT_BIT_WIDTH, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantPDCBlock, self).__init__()
        self.stride = stride
        self.act_bit_width = act_bit_width # Store for requantization

        # Define shortcut only if needed (stride > 1 or channels change)
        self.shortcut = None # Initialize shortcut to None
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = qnn.QuantConv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False,
                                            weight_bit_width=weight_bit_width)
        elif inplane != ouplane: # Stride is 1, channels are different
             self.shortcut = qnn.QuantConv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False,
                                             weight_bit_width=weight_bit_width)
        # else: No explicit shortcut layer if stride=1 and inplane==ouplane

        # Determine kernel size based on converted PDC type
        if pdc_type == 'rd':
            conv1_kernel_size = 5
            conv1_padding = 2
        else: # 'cv', 'cd', 'ad' convert to 3x3
            conv1_kernel_size = 3
            conv1_padding = 1

        self.conv1 = qnn.QuantConv2d(inplane, inplane, kernel_size=conv1_kernel_size, padding=conv1_padding, groups=inplane, bias=False,
                                     weight_bit_width=weight_bit_width)
        # Use standard ReLU (Reverted change)
        self.relu2 = nn.ReLU()
        # Remove explicit quantization after ReLU
        # self.quant_relu_out = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False,
                                     weight_bit_width=weight_bit_width)
        # Addition requantization - Keep removed
        # self.requant_add = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)

    def forward(self, x):
        # Input x assumed to be QuantTensor
        identity = x # Store original input QuantTensor

        if self.stride > 1:
            # Pool the float value
            x_pooled = self.pool(x)
            # Pass the float pooled output directly to conv1
            input_to_conv1 = x_pooled
            # Apply shortcut (must exist) to the pooled float
            identity_processed = self.shortcut(x_pooled)
        else: # Stride = 1
            # Pass float value to conv1
            input_to_conv1 = x
            # Apply shortcut only if it exists (channels changed)
            if self.shortcut is not None: # Check if shortcut layer exists
                 identity_processed = self.shortcut(identity.value)
            else: # No shortcut layer (stride=1, inplane==ouplane), use original QuantTensor
                 identity_processed = identity

        # conv1 receives float Tensor in both stride=1 and stride=2 cases
        y = self.conv1(input_to_conv1)
        # Apply standard ReLU
        y_relu = self.relu2(y)
        # Pass ReLU output (potentially float Tensor) directly to conv2
        y = self.conv2(y_relu)

        # Add residual connection directly
        # y is QuantTensor, identity_processed is QuantTensor
        out = y + identity_processed
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
                                          bias=True, # Ensure bias exists
                                          bias_quant=None, # Disable bias quantization
                                          cache_inference_quant_bias=False)
        # Initialize classifier weights and bias, matching original PiDiNet
        if self.classifier.weight is not None:
            nn.init.constant_(self.classifier.weight, 0.25)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

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
        # Classifier now handles QuantTensor input, uses float bias
        output = self.classifier(quant_cat_out) # Output is QuantTensor

        # Apply final sigmoid quantization
        # Note: e1..e4 are standard tensors here after interpolation
        outputs = [self.final_sigmoid(e) for e in [e1, e2, e3, e4]]
        # Apply final_sigmoid to the classifier output (which is QuantTensor)
        outputs.append(self.final_sigmoid(output))

        # Return float tensors
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

