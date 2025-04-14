import torch
import argparse
import sys
import os
import brevitas.nn as qnn
import torch.nn as nn # Import nn for nn.Conv2d

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.pidinet import PDCBlock_converted
from models.quant_pidinet import QuantPDCBlock
from models.config import config_model_converted # To get pdc_types

def test_pdc_blocks(args):
    """
    Tests if QuantPDCBlock output matches PDCBlock_converted output
    when using equivalent weights and the same input.
    Tests both stride=1 and stride=2 cases.
    """
    print(f"Testing PDC Block equivalence for type: {args.pdc_type}")
    print(f"Args: {args}")

    inplane = args.inplane
    outplane_same = args.inplane # For stride=1, same channel test
    outplane_diff = args.inplane * 2 # For stride=2 test

    # --- Create a consistent initial dummy input ---
    # Use the larger size that causes failure in the other test
    initial_input_channels = 3 # Simulate input to init_block
    initial_dummy_input_float = torch.randn(1, initial_input_channels, 256, 256)

    # --- Simulate Previous Layer (e.g., init_block) ---
    print("Simulating preceding layer...")
    # Use kernel size 3, padding 1 as a common case (adjust if needed based on config)
    dummy_quant_prev_layer = qnn.QuantConv2d(initial_input_channels, inplane, kernel_size=3, padding=1, bias=False,
                                             weight_bit_width=args.weight_bits)
    dummy_float_prev_layer = nn.Conv2d(initial_input_channels, inplane, kernel_size=3, padding=1, bias=False)

    # Synchronize weights of the dummy previous layers
    sync_weights(dummy_quant_prev_layer, dummy_float_prev_layer, "Dummy Previous Layer")

    # Generate inputs for the actual blocks to be tested
    dummy_quant_prev_layer.eval()
    dummy_float_prev_layer.eval()
    with torch.no_grad():
        # Input for QuantPDCBlock comes from the dummy QuantConv2d
        input_for_quant_block = dummy_quant_prev_layer(initial_dummy_input_float)
        # Input for PDCBlock_converted comes from the dummy nn.Conv2d
        input_for_converted_block = dummy_float_prev_layer(initial_dummy_input_float)
    print("Preceding layer simulation complete.")


    # --- Test Stride = 1 ---
    print("\n--- Testing Stride = 1 ---")
    quant_block_s1 = QuantPDCBlock(args.pdc_type, inplane, outplane_same, stride=1,
                                   act_bit_width=args.act_bits, weight_bit_width=args.weight_bits)
    converted_block_s1 = PDCBlock_converted(args.pdc_type, inplane, outplane_same, stride=1)

    # Synchronize weights
    sync_weights(quant_block_s1, converted_block_s1, "Block Stride=1") # Use generic sync function

    # Use generated inputs
    dummy_input_quant_s1 = input_for_quant_block
    dummy_input_float_s1 = input_for_converted_block

    # Run inference
    quant_block_s1.eval()
    converted_block_s1.eval()
    with torch.no_grad():
        quant_output_s1 = quant_block_s1(dummy_input_quant_s1)
        converted_output_s1 = converted_block_s1(dummy_input_float_s1)

    # Compare
    compare_outputs(quant_output_s1, converted_output_s1, "Stride=1")

    # --- Test Stride = 2 ---
    print("\n--- Testing Stride = 2 ---")
    quant_block_s2 = QuantPDCBlock(args.pdc_type, inplane, outplane_diff, stride=2,
                                   act_bit_width=args.act_bits, weight_bit_width=args.weight_bits)
    converted_block_s2 = PDCBlock_converted(args.pdc_type, inplane, outplane_diff, stride=2)

    # Synchronize weights
    sync_weights(quant_block_s2, converted_block_s2, "Block Stride=2") # Use generic sync function

    # Use generated inputs (same inputs as stride 1 test)
    dummy_input_quant_s2 = input_for_quant_block
    dummy_input_float_s2 = input_for_converted_block

    # Run inference
    quant_block_s2.eval()
    converted_block_s2.eval()
    with torch.no_grad():
        quant_output_s2 = quant_block_s2(dummy_input_quant_s2)
        converted_output_s2 = converted_block_s2(dummy_input_float_s2)

    # Compare
    compare_outputs(quant_output_s2, converted_output_s2, "Stride=2")


# Rename sync_block_weights to be more generic
def sync_weights(quant_module, float_module, module_name="Module"):
    """Synchronizes weights between a quantized module and its float equivalent."""
    print(f"Synchronizing weights for {module_name}...")
    quant_state_dict = quant_module.state_dict()
    float_state_dict = float_module.state_dict()
    new_float_state_dict = {}

    # Use float module's keys as reference to handle potential extra keys in quant state_dict (like scaling factors)
    for name, param_float in float_state_dict.items():
        # Find corresponding parameter in quant_state_dict
        # Parameter names should match exactly between qnn.Conv2d and nn.Conv2d (e.g., 'weight')
        if name in quant_state_dict:
            param_quant = quant_state_dict[name]
            if hasattr(param_quant, 'value'):
                 float_param_from_quant = param_quant.value.detach().clone()
            else:
                 # Handle cases where quant param might not have .value (e.g., bias if not quantized)
                 # Or if the layer itself isn't a Brevitas layer (though less likely here)
                 float_param_from_quant = param_quant.detach().clone()

            if float_param_from_quant.shape == param_float.shape:
                new_float_state_dict[name] = float_param_from_quant
            else:
                print(f"Warning: Shape mismatch for {name} in {module_name}. "
                      f"Quant: {float_param_from_quant.shape}, Float: {param_float.shape}. Using original float param.")
                new_float_state_dict[name] = param_float # Keep original if shapes mismatch
        else:
             print(f"Warning: Parameter {name} from float module not found in quant module {module_name}. Using original float param.")
             new_float_state_dict[name] = param_float # Keep original if not found

    # Load the synchronized state dict
    float_module.load_state_dict(new_float_state_dict)
    print(f"Weight synchronization complete for {module_name}.")


def compare_outputs(quant_output, converted_output, test_name):
    """Compares the float outputs of the quantized and converted blocks."""
    print(f"Comparing outputs for {test_name}...")
    if hasattr(quant_output, 'value'):
        quant_output_float = quant_output.value.detach()
    else:
        quant_output_float = quant_output.detach()

    are_close = torch.allclose(quant_output_float, converted_output, atol=1e-5)
    max_diff = torch.max(torch.abs(quant_output_float - converted_output)).item()

    print(f"{test_name} outputs are close: {are_close}")
    print(f"Maximum absolute difference for {test_name}: {max_diff:.6e}")

    if not are_close:
        print(f"Test FAILED for {test_name}!")
    else:
        print(f"Test PASSED for {test_name}!")
    return are_close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test QuantPDCBlock vs PDCBlock_converted')
    parser.add_argument('--pdc-type', type=str, default='ad', choices=['cv', 'cd', 'ad', 'rd'],
                        help='PDC type to test')
    parser.add_argument('--inplane', type=int, default=12, help='Input channels for the block')
    parser.add_argument('--weight-bits', type=int, default=32, help='Default weight bit width') # Default to 32 for testing
    parser.add_argument('--act-bits', type=int, default=32, help='Default activation bit width') # Default to 32 for testing

    args = parser.parse_args()
    test_pdc_blocks(args)
