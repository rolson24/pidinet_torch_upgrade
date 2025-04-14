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
    Tests if the QuantPiDiNet output matches the converted PiDiNet output
    sequentially up to block2_4, using equivalent weights and inputs generated
    from preceding blocks.
    """
    # Update description
    print("Starting test: QuantPiDiNet vs Converted PiDiNet (up to block2_4)")
    print(f"Args: {args}")

    # --- Instantiate Full Models ---
    print("Instantiating full models...")
    # ... (model instantiation code remains the same) ...
    if args.model == 'quant_pidinet_micro':
        quant_model_full = quant_pidinet_micro(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
        converted_model_func = pidinet_micro_converted
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
    converted_model_full = converted_model_func(args)


    # --- Extract Submodules ---
    quant_init_block = quant_model_full.init_block
    converted_init_block = converted_model_full.init_block
    quant_block1_1 = quant_model_full.block1_1
    converted_block1_1 = converted_model_full.block1_1
    quant_block1_2 = quant_model_full.block1_2
    converted_block1_2 = converted_model_full.block1_2
    quant_block1_3 = quant_model_full.block1_3
    converted_block1_3 = converted_model_full.block1_3
    # Extract block2 modules
    quant_block2_1 = quant_model_full.block2_1
    converted_block2_1 = converted_model_full.block2_1
    quant_block2_2 = quant_model_full.block2_2
    converted_block2_2 = converted_model_full.block2_2
    quant_block2_3 = quant_model_full.block2_3
    converted_block2_3 = converted_model_full.block2_3
    quant_block2_4 = quant_model_full.block2_4
    converted_block2_4 = converted_model_full.block2_4


    quant_init_block.eval()
    converted_init_block.eval()
    quant_block1_1.eval()
    converted_block1_1.eval()
    quant_block1_2.eval()
    converted_block1_2.eval()
    quant_block1_3.eval()
    converted_block1_3.eval()
    # Set block2 modules to eval mode
    quant_block2_1.eval()
    converted_block2_1.eval()
    quant_block2_2.eval()
    converted_block2_2.eval()
    quant_block2_3.eval()
    converted_block2_3.eval()
    quant_block2_4.eval()
    converted_block2_4.eval()


    # --- Synchronize Weights ---
    sync_weights(quant_init_block, converted_init_block, "init_block")
    sync_weights(quant_block1_1, converted_block1_1, "block1_1")
    sync_weights(quant_block1_2, converted_block1_2, "block1_2")
    sync_weights(quant_block1_3, converted_block1_3, "block1_3")
    # Sync block2 modules
    sync_weights(quant_block2_1, converted_block2_1, "block2_1")
    sync_weights(quant_block2_2, converted_block2_2, "block2_2")
    sync_weights(quant_block2_3, converted_block2_3, "block2_3")
    sync_weights(quant_block2_4, converted_block2_4, "block2_4")


    # --- Create Initial Dummy Input ---
    print("Creating initial dummy input...")
    initial_dummy_input_float = torch.randn(1, 3, 256, 256) # Input to the whole network

    # --- Run Inference Sequentially ---
    print("Running inference sequentially through init_block, block1_x, block2_x...")
    with torch.no_grad():
        # Quant Model Path
        quant_out_init = quant_init_block(initial_dummy_input_float)
        quant_out_b1_1 = quant_block1_1(quant_out_init)
        quant_out_b1_2 = quant_block1_2(quant_out_b1_1)
        quant_out_b1_3 = quant_block1_3(quant_out_b1_2)
        # Block 2 path
        quant_out_b2_1 = quant_block2_1(quant_out_b1_3)
        quant_out_b2_2 = quant_block2_2(quant_out_b2_1)
        quant_out_b2_3 = quant_block2_3(quant_out_b2_2)
        quant_out_b2_4 = quant_block2_4(quant_out_b2_3) # Final output for quant path

        # Converted Model Path
        converted_out_init = converted_init_block(initial_dummy_input_float)
        converted_out_b1_1 = converted_block1_1(converted_out_init)
        converted_out_b1_2 = converted_block1_2(converted_out_b1_1)
        converted_out_b1_3 = converted_block1_3(converted_out_b1_2)
        # Block 2 path
        converted_out_b2_1 = converted_block2_1(converted_out_b1_3)
        converted_out_b2_2 = converted_block2_2(converted_out_b2_1)
        converted_out_b2_3 = converted_block2_3(converted_out_b2_2)
        converted_out_b2_4 = converted_block2_4(converted_out_b2_3) # Final output for converted path


    # --- Compare Final Outputs (of Block2_4) ---
    print("Comparing final outputs of block2_4...")
    if hasattr(quant_out_b2_4, 'value'):
        quant_output_final_float = quant_out_b2_4.value.detach()
    else:
        quant_output_final_float = quant_out_b2_4.detach()

    # Use appropriate tolerance
    are_close_final = torch.allclose(quant_output_final_float, converted_out_b2_4, atol=1e-5)
    max_diff_final = torch.max(torch.abs(quant_output_final_float - converted_out_b2_4)).item()

    print(f"Block 2_4 final outputs are close: {are_close_final}")
    print(f"Maximum absolute difference for Block 2_4: {max_diff_final:.6e}")

    if are_close_final:
        print("Test PASSED up to Block 2_4!")
    else:
        print("Test FAILED at Block 2_4!")

    return are_close_final

if __name__ == "__main__":
    # ... (argument parsing remains the same) ...
    parser = argparse.ArgumentParser(description='Test Quantized PiDiNet vs Converted PiDiNet up to block2_4')
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
