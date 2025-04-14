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
    Tests if the full QuantPiDiNet output matches the full converted PiDiNet
    output using equivalent weights and the same input.
    """
    # Update description
    print("Starting test: Full QuantPiDiNet vs Converted PiDiNet")
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

    # --- Set models to eval mode ---
    quant_model_full.eval()
    converted_model_full.eval()

    # --- Synchronize Weights for the full models ---
    # Note: This assumes sync_weights handles potential mismatches gracefully (e.g., quant params)
    sync_weights(quant_model_full, converted_model_full, "Full Model")

    # --- Create Initial Dummy Input ---
    print("Creating initial dummy input...")
    # Use a fixed size consistent with training/evaluation if possible, or keep 256x256
    H, W = 256, 256
    initial_dummy_input_float = torch.randn(1, 3, H, W) # Input to the whole network

    # --- Run Full Forward Pass ---
    print("Running full forward pass...")
    with torch.no_grad():
        # Pass float input directly to both models
        # Quant model handles input quantization internally if needed (via init_block)
        quant_outputs = quant_model_full(initial_dummy_input_float)
        converted_outputs = converted_model_full(initial_dummy_input_float)

    # --- Compare Final Outputs ---
    print("Comparing final outputs...")
    all_outputs_close = True
    max_overall_diff = 0.0

    if not isinstance(quant_outputs, list) or not isinstance(converted_outputs, list):
        print("Error: Expected model outputs to be lists.")
        return False
    if len(quant_outputs) != len(converted_outputs):
        print(f"Error: Output list lengths differ! Quant: {len(quant_outputs)}, Converted: {len(converted_outputs)}")
        return False

    for i, (q_out, c_out) in enumerate(zip(quant_outputs, converted_outputs)):
        # Quant model's final sigmoid returns float, no need for .value
        quant_output_float = q_out.detach()
        converted_output_float = c_out.detach()

        # Check shapes
        if quant_output_float.shape != converted_output_float.shape:
            print(f"Output {i} shape mismatch! Quant: {quant_output_float.shape}, Converted: {converted_output_float.shape}")
            all_outputs_close = False
            continue # Skip comparison if shapes differ

        # Compare values
        are_close = torch.allclose(quant_output_float, converted_output_float, atol=1e-5)
        max_diff = torch.max(torch.abs(quant_output_float - converted_output_float)).item()
        max_overall_diff = max(max_overall_diff, max_diff)

        print(f"Output {i} is close: {are_close} (Max diff: {max_diff:.6e})")
        if not are_close:
            all_outputs_close = False
            # Optional: Print slices for debugging specific output
            # print(f"  Quant output {i} sample:", quant_output_float[0, 0, 0, :5])
            # print(f"  Converted output {i} sample:", converted_output_float[0, 0, 0, :5])


    print(f"\nOverall maximum absolute difference: {max_overall_diff:.6e}")
    if all_outputs_close:
        print("Test PASSED for Full Model!")
    else:
        print("Test FAILED for Full Model!")

    return all_outputs_close

if __name__ == "__main__":
    # ... (argument parsing remains the same) ...
    parser = argparse.ArgumentParser(description='Test Full Quantized PiDiNet vs Converted PiDiNet')
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
