import torch
import argparse
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.pidinet import pidinet_micro_converted, pidinet_tiny_converted, pidinet_small_converted, pidinet_converted
from models.quant_pidinet import quant_pidinet_micro, quant_pidinet_tiny, quant_pidinet_small, quant_pidinet
from models.convert_pidinet import convert_pdc
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
    pdcs_types = config_model_converted(args.config)

    pdc_weight_map = {
        'init_block.weight': pdcs_types[0],
        'block1_1.conv1.weight': pdcs_types[1],
        'block1_2.conv1.weight': pdcs_types[2],
        'block1_3.conv1.weight': pdcs_types[3],
        'block2_1.conv1.weight': pdcs_types[4],
        'block2_2.conv1.weight': pdcs_types[5],
        'block2_3.conv1.weight': pdcs_types[6],
        'block2_4.conv1.weight': pdcs_types[7],
        'block3_1.conv1.weight': pdcs_types[8],
        'block3_2.conv1.weight': pdcs_types[9],
        'block3_3.conv1.weight': pdcs_types[10],
        'block3_4.conv1.weight': pdcs_types[11],
        'block4_1.conv1.weight': pdcs_types[12],
        'block4_2.conv1.weight': pdcs_types[13],
        'block4_3.conv1.weight': pdcs_types[14],
        'block4_4.conv1.weight': pdcs_types[15],
    }

    for name, param_quant in quant_state_dict.items():
        if name in converted_state_dict:
            # Extract float value from Brevitas quantized tensor
            # Brevitas parameters might be wrapped, access .value if it's a QuantTensor
            if hasattr(param_quant, 'value'):
                 float_param = param_quant.value.detach().clone()
            else:
                 # Handle cases where it might be a regular tensor (e.g. running mean/var in BN if used)
                 # Or if bias is not quantized explicitly
                 float_param = param_quant.detach().clone()


            # Check if this weight needs PDC conversion
            if name in pdc_weight_map:
                pdc_type = pdc_weight_map[name]
                print(f"Converting weight {name} using PDC type: {pdc_type}")
                converted_weight = convert_pdc(pdc_type, float_param)
                # Ensure the shape matches the target converted model's layer
                if converted_weight.shape == converted_state_dict[name].shape:
                    new_converted_state_dict[name] = converted_weight
                else:
                    print(f"Warning: Shape mismatch for {name} after conversion. "
                          f"Quant: {float_param.shape}, "
                          f"Converted PDC: {converted_weight.shape}, "
                          f"Target: {converted_state_dict[name].shape}. Skipping.")
                    # Keep original random weight in converted model if shapes mismatch
                    new_converted_state_dict[name] = converted_state_dict[name]

            # Handle other weights/biases (shortcuts, conv2, classifier, etc.)
            # Check for potential naming differences (e.g., bias vs .bias)
            elif name.endswith('.bias') and name in converted_state_dict:
                 new_converted_state_dict[name] = float_param
            elif name.endswith('.weight') and name in converted_state_dict:
                 # Check if it's a weight that doesn't need PDC conversion
                 if name not in pdc_weight_map:
                     new_converted_state_dict[name] = float_param
            else:
                 # Catch any other parameters that match directly
                 if name in converted_state_dict and float_param.shape == converted_state_dict[name].shape:
                    new_converted_state_dict[name] = float_param
                 # else:
                 #    print(f"Skipping parameter {name} (not found or shape mismatch in converted model)")


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
