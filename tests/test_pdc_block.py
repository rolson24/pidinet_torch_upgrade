import torch
import argparse
import sys
import os
import brevitas.nn as qnn

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

    # --- Test Stride = 1 ---
    print("\n--- Testing Stride = 1 ---")
    quant_block_s1 = QuantPDCBlock(args.pdc_type, inplane, outplane_same, stride=1,
                                   act_bit_width=args.act_bits, weight_bit_width=args.weight_bits)
    converted_block_s1 = PDCBlock_converted(args.pdc_type, inplane, outplane_same, stride=1)

    # Synchronize weights
    sync_block_weights(quant_block_s1, converted_block_s1)

    # Create inputs
    dummy_input_float_s1 = torch.randn(1, inplane, 256, 256) # Smaller size for block test
    input_quantizer_s1 = qnn.QuantIdentity(bit_width=args.act_bits, return_quant_tensor=True)
    dummy_input_quant_s1 = input_quantizer_s1(dummy_input_float_s1)

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
    sync_block_weights(quant_block_s2, converted_block_s2)

    # Create inputs (use same float input as stride 1)
    dummy_input_float_s2 = dummy_input_float_s1
    input_quantizer_s2 = qnn.QuantIdentity(bit_width=args.act_bits, return_quant_tensor=True)
    dummy_input_quant_s2 = input_quantizer_s2(dummy_input_float_s2)

    # Run inference
    quant_block_s2.eval()
    converted_block_s2.eval()
    with torch.no_grad():
        quant_output_s2 = quant_block_s2(dummy_input_quant_s2)
        converted_output_s2 = converted_block_s2(dummy_input_float_s2)

    # Compare
    compare_outputs(quant_output_s2, converted_output_s2, "Stride=2")


def sync_block_weights(quant_block, converted_block):
    """Synchronizes weights between a QuantPDCBlock and PDCBlock_converted."""
    print("Synchronizing block weights...")
    quant_block_state_dict = quant_block.state_dict()
    converted_block_state_dict = converted_block.state_dict()
    new_converted_block_state_dict = {}

    for name, param_quant in quant_block_state_dict.items():
        if name in converted_block_state_dict:
            if hasattr(param_quant, 'value'):
                 float_param = param_quant.value.detach().clone()
            else:
                 float_param = param_quant.detach().clone()

            if float_param.shape == converted_block_state_dict[name].shape:
                new_converted_block_state_dict[name] = float_param
            else:
                print(f"Warning: Shape mismatch for {name}. Q:{float_param.shape}, C:{converted_block_state_dict[name].shape}. Skipping.")
                new_converted_block_state_dict[name] = converted_block_state_dict[name]

    for key in converted_block_state_dict.keys():
        if key not in new_converted_block_state_dict:
            print(f"Warning: Key {key} not found in sync dict, using original.")
            new_converted_block_state_dict[key] = converted_block_state_dict[key]

    converted_block.load_state_dict(new_converted_block_state_dict)
    print("Block weight synchronization complete.")


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
    parser.add_argument('--inplane', type=int, default=12, help='Input channels')
    parser.add_argument('--weight-bits', type=int, default=8, help='Default weight bit width')
    parser.add_argument('--act-bits', type=int, default=8, help='Default activation bit width')

    args = parser.parse_args()
    test_pdc_blocks(args)
