"""
Quantized PiDiNet model using Brevitas.
Based on the converted (vanilla CNN) version of PiDiNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant # Example bias quantizer

from .config import config_model_converted, nets # To get PDC types for kernel size logic, Import nets

# Default bit widths (can be overridden in factory functions or args)
DEFAULT_WEIGHT_BIT_WIDTH = 8
DEFAULT_ACT_BIT_WIDTH = 8

class QuantCSAM(nn.Module):
    """ Quantized Compact Spatial Attention Module """
    def __init__(self, channels, act_bit_width=DEFAULT_ACT_BIT_WIDTH, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantCSAM, self).__init__()
        mid_channels = 4
        self.relu1 = nn.ReLU()
        # Keep QuantIdentity after ReLU since conv1 uses BiasQuant
        self.quant_relu_out = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(channels, mid_channels, kernel_size=1, padding=0,
                                     weight_bit_width=weight_bit_width,
                                     bias=True, # Ensure bias parameter exists
                                     bias_quant=BiasQuant, # Re-enable bias quantization
                                     cache_inference_quant_bias=True) # Re-enable cache
        if self.conv1.bias is not None:
             nn.init.constant_(self.conv1.bias, 0)

        self.conv2 = qnn.QuantConv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False,
                                     weight_bit_width=weight_bit_width)
        # Replace nn.Sigmoid with qnn.QuantSigmoid
        self.sigmoid = qnn.QuantSigmoid(bit_width=act_bit_width, return_quant_tensor=True) # Return QuantTensor

    def forward(self, x): # Input x: QuantTensor or float
        # Ensure input is QuantTensor if it's not already
        if not isinstance(x, torch.Tensor) or not hasattr(x, 'is_quantized'):
             # Add an input quantizer if needed, assuming x might be float
             # This might require defining self.quant_input in __init__
             # For now, assume x is QuantTensor or handle upstream
             pass # Placeholder

        # Apply standard ReLU -> float Tensor
        y_float = self.relu1(x)
        # Requantize before conv1 because BiasQuant needs scale
        y = self.quant_relu_out(y_float) # y: QuantTensor
        y = self.conv1(y) # y: QuantTensor
        y = self.conv2(y) # y: QuantTensor
        # Apply QuantSigmoid
        y_sigmoid = self.sigmoid(y) # y_sigmoid is now QuantTensor
        # Multiply original QuantTensor x by QuantTensor sigmoid output
        # Brevitas handles QuantTensor * QuantTensor multiplication
        return x * y_sigmoid

class QuantCDCM(nn.Module):
    """ Quantized Compact Dilation Convolution based Module """
    def __init__(self, in_channels, out_channels, act_bit_width=DEFAULT_ACT_BIT_WIDTH, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantCDCM, self).__init__()
        # Replace QuantReLU with nn.ReLU
        self.relu1 = nn.ReLU()
        # Keep QuantIdentity after ReLU since conv1 uses BiasQuant
        self.quant_relu_out = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(in_channels, out_channels, kernel_size=1, padding=0,
                                     weight_bit_width=weight_bit_width,
                                     bias=True, # Ensure bias exists
                                     bias_quant=BiasQuant, # Re-enable bias quantization
                                     cache_inference_quant_bias=True) # Re-enable cache
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


    def forward(self, x): # Input x: QuantTensor or float
        # Ensure input is QuantTensor if it's not already
        if not isinstance(x, torch.Tensor) or not hasattr(x, 'is_quantized'):
             # Add an input quantizer if needed
             pass # Placeholder

        # Apply standard ReLU -> float Tensor
        x_float = self.relu1(x)
        # Requantize before conv1 because BiasQuant needs scale
        x = self.quant_relu_out(x_float) # x: QuantTensor
        # Pass QuantTensor to conv1
        x = self.conv1(x) # x: QuantTensor
        # Pass QuantTensor to dilated convs
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)

        # Use QuantAdd for summation if available and desired for full quantization
        # For simplicity, standard add is often sufficient if followed by requantization
        # If using standard add, the output might be float or require careful scale handling
        # Let's assume standard add works and rely on subsequent layers' input quantizers
        # If issues arise, replace with cascaded qnn.QuantAdd
        out = x1 + x2 + x3 + x4

        return out # Output: QuantTensor (if using QuantAdd) or potentially float

class QuantMapReduce(nn.Module):
    """ Quantized Reduce feature maps into a single edge map """
    def __init__(self, channels, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH, act_bit_width=DEFAULT_ACT_BIT_WIDTH): # act_bit_width is needed
        super(QuantMapReduce, self).__init__()
        # Re-add input requantization layer to ensure scale is available for BiasQuant
        self.requant_input = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(channels, 1, kernel_size=1, padding=0,
                                    weight_bit_width=weight_bit_width,
                                    bias=True, # Ensure bias exists
                                    bias_quant=BiasQuant, # Bias quantization enabled
                                    cache_inference_quant_bias=True) # Cache enabled
        # Initialize bias to 0, matching original MapReduce
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x): # Input x: QuantTensor or float Tensor
        # Ensure input is explicitly quantized before conv layer
        x_quant = self.requant_input(x)
        # Now conv receives a QuantTensor with a scale from requant_input
        # Output will be QuantTensor
        return self.conv(x_quant)


class QuantPDCBlock(nn.Module):
    """ Quantized PDCBlock based on converted structure """
    def __init__(self, pdc_type, inplane, ouplane, stride=1, act_bit_width=DEFAULT_ACT_BIT_WIDTH, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantPDCBlock, self).__init__()
        self.stride = stride
        self.act_bit_width = act_bit_width # Store for requantization

        # Input quantizer for the block
        self.quant_input = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)

        # Define shortcut logic
        self.shortcut = None # Initialize shortcut to None
        self.requant_pool = None # Initialize pool requantizer
        if self.stride > 1:
            # MaxPool outputs float, need requantization before shortcut/conv1
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.requant_pool = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
            self.shortcut = qnn.QuantConv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False,
                                            weight_bit_width=weight_bit_width)
        elif inplane != ouplane: # Stride is 1, channels are different
             self.shortcut = qnn.QuantConv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False,
                                             weight_bit_width=weight_bit_width)
        else: # Stride is 1, channels are the same
             # Use QuantIdentity to ensure QuantTensor passthrough for residual
             self.shortcut = qnn.QuantIdentity(return_quant_tensor=True)


        # Determine kernel size based on converted PDC type
        if pdc_type == 'rd':
            conv1_kernel_size = 5
            conv1_padding = 2
        else: # 'cv', 'cd', 'ad' convert to 3x3
            conv1_kernel_size = 3
            conv1_padding = 1

        self.conv1 = qnn.QuantConv2d(inplane, inplane, kernel_size=conv1_kernel_size, padding=conv1_padding, groups=inplane, bias=False,
                                     weight_bit_width=weight_bit_width)
        # Use QuantReLU
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.conv2 = qnn.QuantConv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False,
                                     weight_bit_width=weight_bit_width)

        # Quantized addition for the residual connection
        self.residual_add = qnn.QuantAdd(
            bit_width=act_bit_width, # Output bitwidth usually matches input activation bitwidth
            return_quant_tensor=True)

    def forward(self, x):
        # Ensure input is QuantTensor
        x_quant = self.quant_input(x)
        identity = x_quant # Store quantized identity

        input_to_conv1 = x_quant # Default input for conv1 (stride=1 case)

        if self.stride > 1:
            x_pooled = self.pool(x_quant) # pool outputs float
            x_quant_pooled = self.requant_pool(x_pooled) # requantize before conv1/shortcut
            input_to_conv1 = x_quant_pooled # Use requantized pooled tensor for conv1
            identity = self.shortcut(x_quant_pooled) # Apply shortcut to requantized pooled input
        elif self.shortcut is not None: # Apply shortcut if it exists (stride=1 cases)
             identity = self.shortcut(identity)
        # else: identity remains x_quant (stride=1, same channels)

        y = self.conv1(input_to_conv1) # Input is QuantTensor
        y = self.relu2(y)       # Input/Output are QuantTensor
        y = self.conv2(y)       # Input/Output are QuantTensor

        # Use QuantAdd for residual connection
        out = self.residual_add(y, identity) # Both inputs must be QuantTensor

        return out # Output is QuantTensor


class QuantPiDiNet(nn.Module):
    def __init__(self, inplane, pdcs_types, dil=None, sa=False, act_bit_width=DEFAULT_ACT_BIT_WIDTH, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH):
        super(QuantPiDiNet, self).__init__()
        self.sa = sa
        if dil is not None:
            assert isinstance(dil, int), 'dil should be an int'
        self.dil = dil

        self.fuseplanes = []
        self.inplane = inplane # Store initial inplane

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
        self.init_block = qnn.QuantConv2d(3, self.inplane,
                                          kernel_size=init_kernel_size, padding=init_padding, bias=False,
                                          weight_bit_width=weight_bit_width)


        # Define Blocks
        block_class = QuantPDCBlock
        cur_inplane = self.inplane
        self.block1_1 = block_class(pdcs_types[1], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block1_2 = block_class(pdcs_types[2], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block1_3 = block_class(pdcs_types[3], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.fuseplanes.append(cur_inplane) # C

        inplane_prev = cur_inplane
        cur_inplane = cur_inplane * 2
        self.block2_1 = block_class(pdcs_types[4], inplane_prev, cur_inplane, stride=2, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block2_2 = block_class(pdcs_types[5], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block2_3 = block_class(pdcs_types[6], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block2_4 = block_class(pdcs_types[7], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.fuseplanes.append(cur_inplane) # 2C

        inplane_prev = cur_inplane
        cur_inplane = cur_inplane * 2
        self.block3_1 = block_class(pdcs_types[8], inplane_prev, cur_inplane, stride=2, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block3_2 = block_class(pdcs_types[9], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block3_3 = block_class(pdcs_types[10], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block3_4 = block_class(pdcs_types[11], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.fuseplanes.append(cur_inplane) # 4C

        inplane_prev = cur_inplane
        # Block 4 keeps the same number of channels (4C)
        self.block4_1 = block_class(pdcs_types[12], inplane_prev, cur_inplane, stride=2, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block4_2 = block_class(pdcs_types[13], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block4_3 = block_class(pdcs_types[14], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.block4_4 = block_class(pdcs_types[15], cur_inplane, cur_inplane, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width)
        self.fuseplanes.append(cur_inplane) # 4C

        # Fusion Layers (Quantized)
        # ... existing side path module definitions (QuantCDCM, QuantCSAM, QuantMapReduce) ...
        self.conv_reduces = nn.ModuleList()
        if self.sa and self.dil is not None:
            self.attentions = nn.ModuleList()
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(QuantCDCM(self.fuseplanes[i], self.dil, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width))
                self.attentions.append(QuantCSAM(self.dil, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width))
                self.conv_reduces.append(QuantMapReduce(self.dil, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width))
        elif self.sa:
            self.attentions = nn.ModuleList()
            for i in range(4):
                self.attentions.append(QuantCSAM(self.fuseplanes[i], act_bit_width=act_bit_width, weight_bit_width=weight_bit_width))
                self.conv_reduces.append(QuantMapReduce(self.fuseplanes[i], weight_bit_width=weight_bit_width, act_bit_width=act_bit_width))
        elif self.dil is not None:
            self.dilations = nn.ModuleList()
            for i in range(4):
                self.dilations.append(QuantCDCM(self.fuseplanes[i], self.dil, act_bit_width=act_bit_width, weight_bit_width=weight_bit_width))
                self.conv_reduces.append(QuantMapReduce(self.dil, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width))
        else:
            for i in range(4):
                self.conv_reduces.append(QuantMapReduce(self.fuseplanes[i], weight_bit_width=weight_bit_width, act_bit_width=act_bit_width))


        # Add QuantIdentity layer to handle concatenation before classifier
        self.quant_cat = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)

        # Final Classifier
        self.classifier = qnn.QuantConv2d(4, 1, kernel_size=1,
                                          weight_bit_width=weight_bit_width,
                                          bias=True, # Ensure bias exists
                                          bias_quant=BiasQuant, # Re-enable bias quantization
                                          # Add input quantizer to handle potential float input from concat fallback
                                          input_quant=qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True),
                                          cache_inference_quant_bias=True) # Re-enable cache
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

        # Blocks now consume and produce QuantTensors
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

        # Feature Fusion / Attention / Dilation (Inputs/Outputs are QuantTensors)
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
            x_fuses = [x1, x2, x3, x4] # These are QuantTensors

        # Reduce (Outputs QuantTensor) and Upsample (Outputs float)
        e1 = self.conv_reduces[0](x_fuses[0])
        e1_interp = F.interpolate(e1.value, (H, W), mode="bilinear", align_corners=False) # Interpolate float value

        e2 = self.conv_reduces[1](x_fuses[1])
        e2_interp = F.interpolate(e2.value, (H, W), mode="bilinear", align_corners=False)

        e3 = self.conv_reduces[2](x_fuses[2])
        e3_interp = F.interpolate(e3.value, (H, W), mode="bilinear", align_corners=False)

        e4 = self.conv_reduces[3](x_fuses[3])
        e4_interp = F.interpolate(e4.value, (H, W), mode="bilinear", align_corners=False)

        # Classifier and Final Output
        # Concatenate the interpolated float tensors
        cat_out = torch.cat([e1_interp, e2_interp, e3_interp, e4_interp], dim=1)
        # Requantize the concatenated tensor before feeding to the classifier
        # This is handled by self.classifier.input_quant
        output = self.classifier(cat_out) # Classifier handles input quantization, output is QuantTensor

        # Apply final sigmoid quantization to classifier output
        final_output = self.final_sigmoid(output) # Returns float

        # Prepare side outputs (apply sigmoid to interpolated float values)
        s1 = torch.sigmoid(e1_interp)
        s2 = torch.sigmoid(e2_interp)
        s3 = torch.sigmoid(e3_interp)
        s4 = torch.sigmoid(e4_interp)

        outputs = [s1, s2, s3, s4, final_output] # All should be float tensors

        return outputs


# Factory functions for different sizes
def quant_pidinet_micro(args, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH, act_bit_width=DEFAULT_ACT_BIT_WIDTH):
    pdcs_types = config_model_converted(args.config) # Get pdc types ('cv', 'cd', 'ad', 'rd')
    dil = 4 if args.dil else None
    # Pass args directly to QuantPiDiNet constructor if needed, or just necessary attributes
    model_args = argparse.Namespace(sa=args.sa, dil=args.dil) # Create a simple namespace if needed
    return QuantPiDiNet(12, pdcs_types, dil=dil, sa=args.sa, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

def quant_pidinet_tiny(args, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH, act_bit_width=DEFAULT_ACT_BIT_WIDTH):
    pdcs_types = config_model_converted(args.config)
    dil = 8 if args.dil else None
    model_args = argparse.Namespace(sa=args.sa, dil=args.dil)
    return QuantPiDiNet(20, pdcs_types, dil=dil, sa=args.sa, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

def quant_pidinet_small(args, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH, act_bit_width=DEFAULT_ACT_BIT_WIDTH):
    pdcs_types = config_model_converted(args.config)
    dil = 12 if args.dil else None
    model_args = argparse.Namespace(sa=args.sa, dil=args.dil)
    return QuantPiDiNet(30, pdcs_types, dil=dil, sa=args.sa, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

def quant_pidinet(args, weight_bit_width=DEFAULT_WEIGHT_BIT_WIDTH, act_bit_width=DEFAULT_ACT_BIT_WIDTH):
    pdcs_types = config_model_converted(args.config)
    dil = 24 if args.dil else None
    model_args = argparse.Namespace(sa=args.sa, dil=args.dil)
    return QuantPiDiNet(60, pdcs_types, dil=dil, sa=args.sa, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)

