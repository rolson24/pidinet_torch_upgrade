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

def test_quant_vs_converted(args):
    """
    Tests if the QuantPiDiNet output matches the converted PiDiNet output
    when using equivalent weights.
    """
    print("Starting test: QuantPiDiNet vs Converted PiDiNet")
    print(f"Args: {args}")

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
                # print(f"Synchronized {name}") # Optional: for debugging
            else:
                print(f"Warning: Shape mismatch for {name}. "
                      f"Quant: {float_param.shape}, "
                      f"Target: {converted_state_dict[name].shape}. Skipping.")
                # Keep original random weight in converted model if shapes mismatch
                new_converted_state_dict[name] = converted_state_dict[name]

        # else:
        #     print(f"Parameter {name} from quant_model not found in converted_model state_dict.")

    # Load the synchronized weights into the converted model
    # Make sure all keys are present; use original if missing from quant
    for key in converted_state_dict.keys():
        if key not in new_converted_state_dict:
            print(f"Warning: Key {key} not found in synchronized dict, using original random weights.")
            new_converted_state_dict[key] = converted_state_dict[key]

    converted_model.load_state_dict(new_converted_state_dict)
    print("Weight synchronization complete.")

    # --- Create Dummy Input ---
    print("Creating dummy input...")
    dummy_input = torch.randn(1, 3, 256, 256) # Example size

    # --- Run Inference ---
    print("Running inference...")
    with torch.no_grad():
        quant_outputs = quant_model(dummy_input)
        converted_outputs = converted_model(dummy_input)

    # Get the final output map (last element in the list)
    quant_final_output = quant_outputs[-1]
    converted_final_output = converted_outputs[-1]

    # --- Compare Outputs ---
    print("Comparing outputs...")
    # Use a slightly higher tolerance due to potential quantization effects
    are_close = torch.allclose(quant_final_output, converted_final_output, atol=1e-5)
    max_diff = torch.max(torch.abs(quant_final_output - converted_final_output)).item()

    print(f"Outputs are close: {are_close}")
    print(f"Maximum absolute difference: {max_diff:.6e}")

    if are_close:
        print("Test PASSED!")
    else:
        print("Test FAILED!")
        # Optionally print parts of the tensors for debugging
        # print("Quant output sample:", quant_final_output.flatten()[:10])
        # print("Converted output sample:", converted_final_output.flatten()[:10])
        # print("Difference sample:", (quant_final_output - converted_final_output).flatten()[:10])

    return are_close

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Quantized PiDiNet vs Converted PiDiNet')
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
