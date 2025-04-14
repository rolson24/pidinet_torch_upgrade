import torch
import argparse
import sys
import os

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
    Tests if the QuantPiDiNet output matches the converted PiDiNet output
    after the first block (block1_3) when using equivalent weights.
    """
    print("Starting test: QuantPiDiNet vs Converted PiDiNet (Block 1 Output)")
    print(f"Args: {args}")

    # Clear activation dictionaries for a fresh run
    quant_activations.clear()
    converted_activations.clear()

    # --- Instantiate Models ---
    print("Instantiating models...")
    # Quantized Model
    if args.model == 'quant_pidinet_micro':
        quant_model = quant_pidinet_micro(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_micro_converted
    elif args.model == 'quant_pidinet_tiny':
        quant_model = quant_pidinet_tiny(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_tiny_converted
    elif args.model == 'quant_pidinet_small':
        quant_model = quant_pidinet_small(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_small_converted
    elif args.model == 'quant_pidinet':
        quant_model = quant_pidinet(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_converted
    else:
        raise ValueError(f"Unknown quantized model type: {args.model}")

    # Converted Vanilla Model
    converted_model = converted_model_func(args)

    quant_model.eval()
    converted_model.eval()

    # --- Synchronize Weights ---
    print("Synchronizing weights...")
    quant_state_dict = quant_model.state_dict()
    converted_state_dict = converted_model.state_dict()
    new_converted_state_dict = {}

    for name, param_quant in quant_state_dict.items():
        if name in converted_state_dict:
            # Extract float value from Brevitas quantized tensor
            if hasattr(param_quant, 'value'):
                 float_param = param_quant.value.detach().clone()
            else:
                 float_param = param_quant.detach().clone()

            # Directly assign the float parameter if shapes match
            if float_param.shape == converted_state_dict[name].shape:
                new_converted_state_dict[name] = float_param
            else:
                print(f"Warning: Shape mismatch for {name}. "
                      f"Quant: {float_param.shape}, "
                      f"Target: {converted_state_dict[name].shape}. Skipping.")
                new_converted_state_dict[name] = converted_state_dict[name]

    for key in converted_state_dict.keys():
        if key not in new_converted_state_dict:
            print(f"Warning: Key {key} not found in synchronized dict, using original random weights.")
            new_converted_state_dict[key] = converted_state_dict[key]

    converted_model.load_state_dict(new_converted_state_dict)
    print("Weight synchronization complete.")

    # --- Register Hooks ---
    print("Registering hooks for block1_3...")
    # Ensure module names match exactly how they are defined in the classes
    # Accessing through module attributes if DataParallel is not used here
    hook_handle_quant = quant_model.block1_3.register_forward_hook(get_activation('block1_3', quant_activations))
    hook_handle_converted = converted_model.block1_3.register_forward_hook(get_activation('block1_3', converted_activations))

    # --- Create Dummy Input ---
    print("Creating dummy input...")
    dummy_input = torch.randn(1, 3, 256, 256) # Example size

    # --- Run Inference up to Block 1 ---
    print("Running inference up to block1_3...")
    with torch.no_grad():
        # Quant Model Path
        x_q = quant_model.quant_inp(dummy_input)
        x_q = quant_model.init_block(x_q)
        x_q = quant_model.block1_1(x_q)
        x_q = quant_model.block1_2(x_q)
        quant_model.block1_3(x_q) # Run final block to trigger hook

        # Converted Model Path
        x_c = converted_model.init_block(dummy_input)
        x_c = converted_model.block1_1(x_c)
        x_c = converted_model.block1_2(x_c)
        converted_model.block1_3(x_c) # Run final block to trigger hook

    # --- Remove Hooks ---
    hook_handle_quant.remove()
    hook_handle_converted.remove()
    print("Hooks removed.")

    # --- Compare Outputs ---
    print("Comparing outputs after block1_3...")
    if 'block1_3' not in quant_activations or 'block1_3' not in converted_activations:
        print("Error: Failed to capture activations from block1_3.")
        return False

    quant_block1_output = quant_activations['block1_3']
    converted_block1_output = converted_activations['block1_3']

    # Use a slightly higher tolerance due to potential quantization effects
    are_close = torch.allclose(quant_block1_output, converted_block1_output, atol=1e-5)
    max_diff = torch.max(torch.abs(quant_block1_output - converted_block1_output)).item()

    print(f"Block 1 outputs are close: {are_close}")
    print(f"Maximum absolute difference after Block 1: {max_diff:.6e}")

    if are_close:
        print("Test PASSED for Block 1!")
    else:
        print("Test FAILED for Block 1!")

    return are_close

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Quantized PiDiNet vs Converted PiDiNet (Block 1)')
    # Add arguments needed by the model factory functions and config
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
