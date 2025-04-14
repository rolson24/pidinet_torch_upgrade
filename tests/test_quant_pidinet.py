import torch
import argparse
import sys
import os
import brevitas.nn as qnn # Import qnn for input quantization

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.pidinet import pidinet_micro_converted, pidinet_tiny_converted, pidinet_small_converted, pidinet_converted
from models.quant_pidinet import quant_pidinet_micro, quant_pidinet_tiny, quant_pidinet_small, quant_pidinet
from models.config import config_model_converted

# Dictionary to store intermediate activations
quant_activations = {}
converted_activations = {}

def get_activation(name, activations_dict):
    """Hook function to capture activations"""
    def hook(model, input, output):
        # Store the output tensor (detach if needed)
        # For QuantTensor, store its float value for comparison
        if hasattr(output, 'value'):
            activations_dict[name] = output.value.detach()
        else:
            activations_dict[name] = output.detach()
    return hook

def test_quant_vs_converted(args):
    """
    Tests if the QuantPiDiNet.block1_1 output matches the converted
    PiDiNet.block1_1 output when using equivalent weights and the same input.
    """
    print("Starting test: QuantPiDiNet.block1_1 vs Converted PiDiNet.block1_1")
    print(f"Args: {args}")

    # --- Instantiate Full Models (to get configured submodules) ---
    print("Instantiating full models to extract block1_1...")
    # Quantized Model
    if args.model == 'quant_pidinet_micro':
        quant_model_full = quant_pidinet_micro(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_micro_converted
        inplane = 12 # Input channels for block1_1 in micro model
    elif args.model == 'quant_pidinet_tiny':
        quant_model_full = quant_pidinet_tiny(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_tiny_converted
        inplane = 20
    elif args.model == 'quant_pidinet_small':
        quant_model_full = quant_pidinet_small(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_small_converted
        inplane = 30
    elif args.model == 'quant_pidinet':
        quant_model_full = quant_pidinet(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_converted
        inplane = 60
    else:
        raise ValueError(f"Unknown quantized model type: {args.model}")

    # Converted Vanilla Model
    converted_model_full = converted_model_func(args)

    # --- Extract Block1_1 Submodules ---
    quant_block = quant_model_full.block1_1
    converted_block = converted_model_full.block1_1

    quant_block.eval()
    converted_block.eval()

    # --- Synchronize Weights for Block1_1 ---
    print("Synchronizing weights for block1_1...")
    quant_block_state_dict = quant_block.state_dict()
    converted_block_state_dict = converted_block.state_dict()
    new_converted_block_state_dict = {}

    # Iterate through the *quantized block's* state dict
    for name, param_quant in quant_block_state_dict.items():
        # Check if the parameter exists in the *converted block's* state dict
        if name in converted_block_state_dict:
            # Extract float value
            if hasattr(param_quant, 'value'):
                 float_param = param_quant.value.detach().clone()
            else:
                 float_param = param_quant.detach().clone()

            # Directly assign if shapes match
            if float_param.shape == converted_block_state_dict[name].shape:
                new_converted_block_state_dict[name] = float_param
            else:
                # This case should ideally not happen if structures match
                print(f"Warning: Shape mismatch for {name} in block1_1. "
                      f"Quant: {float_param.shape}, "
                      f"Target: {converted_block_state_dict[name].shape}. Skipping.")
                new_converted_block_state_dict[name] = converted_block_state_dict[name]
        # else:
        #     print(f"Parameter {name} from quant_block not found in converted_block.")

    # Ensure all keys from converted block are present
    for key in converted_block_state_dict.keys():
        if key not in new_converted_block_state_dict:
            print(f"Warning: Key {key} from converted_block not found in synchronized dict, using original random weights.")
            new_converted_block_state_dict[key] = converted_block_state_dict[key]

    converted_block.load_state_dict(new_converted_block_state_dict)
    print("Weight synchronization complete for block1_1.")

    # --- Create Dummy Input for Block1_1 ---
    print("Creating dummy input for block1_1...")
    # Input shape should match the output of init_block: (batch, inplane, H, W)
    dummy_input_float = torch.randn(1, inplane, 256, 256)

    # Quantize the input for the quantized block
    # Use the activation bit width specified in args
    input_quantizer = qnn.QuantIdentity(bit_width=args.act_bits, return_quant_tensor=True)
    dummy_input_quant = input_quantizer(dummy_input_float)

    # --- Run Inference on Block1_1 ---
    print("Running inference on block1_1...")
    with torch.no_grad():
        quant_output = quant_block(dummy_input_quant)
        converted_output = converted_block(dummy_input_float) # Use the float input

    # --- Compare Outputs ---
    print("Comparing outputs of block1_1...")
    # Extract float value from potential QuantTensor output
    if hasattr(quant_output, 'value'):
        quant_output_float = quant_output.value.detach()
    else:
        quant_output_float = quant_output.detach() # Should be QuantTensor based on QuantPDCBlock

    # Use a slightly higher tolerance due to potential quantization effects
    are_close = torch.allclose(quant_output_float, converted_output, atol=1e-5)
    max_diff = torch.max(torch.abs(quant_output_float - converted_output)).item()

    print(f"Block 1_1 outputs are close: {are_close}")
    print(f"Maximum absolute difference for Block 1_1: {max_diff:.6e}")

    if are_close:
        print("Test PASSED for Block 1_1 isolation!")
    else:
        print("Test FAILED for Block 1_1 isolation!")

    return are_close

if __name__ == "__main__":
    # Update description
    parser = argparse.ArgumentParser(description='Test Quantized PiDiNet.block1_1 vs Converted PiDiNet.block1_1')
    # ... (rest of argument parsing remains the same) ...
    parser.add_argument('--model', type=str, default='quant_pidinet_micro',
                        choices=['quant_pidinet_micro', 'quant_pidinet_tiny', 'quant_pidinet_small', 'quant_pidinet'],
                        help='Quantized model type to test')
    parser.add_argument('--config', type=str, default='carv4',
                        help='PiDiNet configuration (e.g., carv4, baseline)')
    parser.add_argument('--sa', action='store_true', default=False,
                        help='Enable Spatial Attention (CSAM)')
    parser.add_argument('--dil', action='store_true', default=False,
                        help='Enable Dilation (CDCM)')
    # Add quantization specific args if needed, otherwise use defaults
    parser.add_argument('--weight-bits', type=int, default=8, help='Default weight bit width')
    parser.add_argument('--act-bits', type=int, default=8, help='Default activation bit width')

    args = parser.parse_args()

    # Ensure 'convert' flag is True for the converted model factory
    args.convert = True

    test_quant_vs_converted(args)
