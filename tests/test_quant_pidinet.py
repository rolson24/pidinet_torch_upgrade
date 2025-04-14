import torch
import argparse
import sys
import os
import brevitas.nn as qnn
import torch.nn as nn # Import nn

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.pidinet import pidinet_micro_converted, pidinet_tiny_converted, pidinet_small_converted, pidinet_converted
from models.quant_pidinet import quant_pidinet_micro, quant_pidinet_tiny, quant_pidinet_small, quant_pidinet
from models.config import config_model_converted
# Import the generic sync_weights function (assuming it's defined elsewhere or copy it here)
# For simplicity, let's copy the sync_weights function from test_pdc_block.py
def sync_weights(quant_module, float_module, module_name="Module"):
    """Synchronizes weights between a quantized module and its float equivalent."""
    print(f"Synchronizing weights for {module_name}...")
    quant_state_dict = quant_module.state_dict()
    float_state_dict = float_module.state_dict()
    new_float_state_dict = {}

    # Use float module's keys as reference
    for name, param_float in float_state_dict.items():
        if name in quant_state_dict:
            param_quant = quant_state_dict[name]
            if hasattr(param_quant, 'value'):
                 float_param_from_quant = param_quant.value.detach().clone()
            else:
                 float_param_from_quant = param_quant.detach().clone()

            if float_param_from_quant.shape == param_float.shape:
                new_float_state_dict[name] = float_param_from_quant
            else:
                print(f"Warning: Shape mismatch for {name} in {module_name}. "
                      f"Quant: {float_param_from_quant.shape}, Float: {param_float.shape}. Using original float param.")
                new_float_state_dict[name] = param_float
        else:
             print(f"Warning: Parameter {name} from float module not found in quant module {module_name}. Using original float param.")
             new_float_state_dict[name] = param_float

    float_module.load_state_dict(new_float_state_dict, strict=False) # Use strict=False initially if unsure about all keys matching
    print(f"Weight synchronization complete for {module_name}.")


def test_quant_vs_converted(args):
    """
    Tests if the QuantPiDiNet.block1_1 output matches the converted
    PiDiNet.block1_1 output when using equivalent weights and inputs generated
    from their respective init_blocks.
    """
    print("Starting test: QuantPiDiNet.block1_1 vs Converted PiDiNet.block1_1")
    print(f"Args: {args}")

    # --- Instantiate Full Models ---
    print("Instantiating full models...")
    # Quantized Model
    if args.model == 'quant_pidinet_micro':
        quant_model_full = quant_pidinet_micro(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_micro_converted
        # inplane = 12 # Not needed directly here
    # ... (add other model sizes elif blocks as before) ...
    elif args.model == 'quant_pidinet_tiny':
        quant_model_full = quant_pidinet_tiny(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_tiny_converted
    elif args.model == 'quant_pidinet_small':
        quant_model_full = quant_pidinet_small(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_small_converted
    elif args.model == 'quant_pidinet':
        quant_model_full = quant_pidinet(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_converted
    else:
        raise ValueError(f"Unknown quantized model type: {args.model}")

    # Converted Vanilla Model
    converted_model_full = converted_model_func(args)

    # --- Extract Submodules ---
    quant_init_block = quant_model_full.init_block
    converted_init_block = converted_model_full.init_block
    quant_block = quant_model_full.block1_1
    converted_block = converted_model_full.block1_1

    quant_init_block.eval()
    converted_init_block.eval()
    quant_block.eval()
    converted_block.eval()

    # --- Synchronize Weights ---
    # Sync init_block
    sync_weights(quant_init_block, converted_init_block, "init_block")
    # Sync block1_1
    sync_weights(quant_block, converted_block, "block1_1")

    # --- Create Initial Dummy Input ---
    print("Creating initial dummy input...")
    initial_dummy_input_float = torch.randn(1, 3, 256, 256) # Input to the whole network

    # --- Generate Inputs for Block1_1 using Init_Blocks ---
    print("Generating inputs for block1_1 using init_blocks...")
    with torch.no_grad():
        # Input for quant_block comes from quant_init_block
        # Need to quantize the initial float input first for quant_init_block
        # Use the input quantizer from the full quant model
        initial_input_quant = quant_model_full.quant_inp(initial_dummy_input_float)
        input_for_quant_block = quant_init_block(initial_input_quant)

        # Input for converted_block comes from converted_init_block
        input_for_converted_block = converted_init_block(initial_dummy_input_float)

    # --- Run Inference on Block1_1 ---
    print("Running inference on block1_1...")
    with torch.no_grad():
        quant_output = quant_block(input_for_quant_block) # Use generated QuantTensor input
        converted_output = converted_block(input_for_converted_block) # Use generated float Tensor input

    # --- Compare Outputs ---
    print("Comparing outputs of block1_1...")
    if hasattr(quant_output, 'value'):
        quant_output_float = quant_output.value.detach()
    else:
        quant_output_float = quant_output.detach()

    # Use appropriate tolerance
    are_close = torch.allclose(quant_output_float, converted_output, atol=1e-5)
    max_diff = torch.max(torch.abs(quant_output_float - converted_output)).item()

    print(f"Block 1_1 outputs are close: {are_close}")
    print(f"Maximum absolute difference for Block 1_1: {max_diff:.6e}")

    if are_close:
        print("Test PASSED for Block 1_1 isolation!")
    else:
        print("Test FAILED for Block 1_1 isolation!")
        # Optional: Print shapes or slices for debugging
        # print("Quant output shape:", quant_output_float.shape)
        # print("Converted output shape:", converted_output.shape)
        # print("Quant output sample:", quant_output_float[0, 0, 0, :5])
        # print("Converted output sample:", converted_output[0, 0, 0, :5])


    return are_close

if __name__ == "__main__":
    # ... (argument parsing remains the same) ...
    parser = argparse.ArgumentParser(description='Test Quantized PiDiNet.block1_1 vs Converted PiDiNet.block1_1')
    parser.add_argument('--model', type=str, default='quant_pidinet_micro',
                        choices=['quant_pidinet_micro', 'quant_pidinet_tiny', 'quant_pidinet_small', 'quant_pidinet'],
                        help='Quantized model type to test')
    parser.add_argument('--config', type=str, default='carv4',
                        help='PiDiNet configuration (e.g., carv4, baseline)')
    parser.add_argument('--sa', action='store_true', default=False,
                        help='Enable Spatial Attention (CSAM)')
    parser.add_argument('--dil', action='store_true', default=False,
                        help='Enable Dilation (CDCM)')
    parser.add_argument('--weight-bits', type=int, default=32, help='Default weight bit width') # Default 32
    parser.add_argument('--act-bits', type=int, default=32, help='Default activation bit width') # Default 32

    args = parser.parse_args()

    # Ensure 'convert' flag is True for the converted model factory
    args.convert = True

    test_quant_vs_converted(args)
