import torch
import argparse
import sys
import os
import brevitas.nn as qnn
import torch.nn as nn

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.pidinet import CDCM
# Import the modified QuantCDCM
from models.quant_pidinet import QuantCDCM

# Copy the sync_weights function (same as in test_quant_csam.py)
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
             if 'bias' not in name: # Only warn if it's not a bias mismatch
                 print(f"Warning: Parameter {name} from float module not found in quant module {module_name}. Using original float param.")
             new_float_state_dict[name] = param_float

    keys_to_remove = []
    for name in new_float_state_dict.keys():
        if name not in float_state_dict:
             print(f"Notice: Parameter {name} found in quant module but not in float module {module_name}. Removing from sync dict.")
             keys_to_remove.append(name)
    for key in keys_to_remove:
        del new_float_state_dict[key]

    float_module.load_state_dict(new_float_state_dict, strict=False)
    print(f"Weight synchronization complete for {module_name}.")


# Copy the compare_outputs function (same as in test_quant_csam.py)
def compare_outputs(quant_output, converted_output, test_name):
    """Compares the float outputs."""
    print(f"Comparing outputs for {test_name}...")
    if hasattr(quant_output, 'value'):
        quant_output_float = quant_output.value.detach()
    else:
        quant_output_float = quant_output.detach()

    converted_output_float = converted_output.detach()

    are_close = torch.allclose(quant_output_float, converted_output_float, atol=1e-5)
    max_diff = torch.max(torch.abs(quant_output_float - converted_output_float)).item()

    print(f"{test_name} outputs are close: {are_close}")
    print(f"Maximum absolute difference for {test_name}: {max_diff:.6e}")

    if not are_close:
        print(f"Test FAILED for {test_name}!")
    else:
        print(f"Test PASSED for {test_name}!")
    return are_close

def test_cdcm_equivalence(args):
    """Tests QuantCDCM vs CDCM."""
    print(f"Testing CDCM equivalence with in_channels={args.in_channels}, out_channels={args.out_channels}")
    print(f"Args: {args}")

    in_channels = args.in_channels
    out_channels = args.out_channels
    act_bits = args.act_bits
    weight_bits = args.weight_bits

    # --- Instantiate Modules ---
    # QuantCDCM already modified in the source file
    quant_cdcm = QuantCDCM(in_channels, out_channels, act_bit_width=act_bits, weight_bit_width=weight_bits)
    float_cdcm = CDCM(in_channels, out_channels)

    # --- Synchronize Weights ---
    # sync_weights should handle the modified QuantCDCM correctly
    sync_weights(quant_cdcm, float_cdcm, "CDCM")

    # --- Create Dummy Input ---
    print("Simulating preceding layer input...")
    # Input channels for CDCM are `in_channels`
    prev_channels = in_channels
    dummy_input_shape = (1, prev_channels, 64, 64) # Example spatial size
    initial_dummy_input_float = torch.randn(*dummy_input_shape)

    # Simulate previous layers
    dummy_quant_prev_layer = qnn.QuantConv2d(prev_channels, in_channels, kernel_size=1,
                                             weight_bit_width=weight_bits,
                                             bias=False)
    dummy_float_prev_layer = nn.Conv2d(prev_channels, in_channels, kernel_size=1, bias=False)

    sync_weights(dummy_quant_prev_layer, dummy_float_prev_layer, "Dummy Previous Layer")

    dummy_quant_prev_layer.eval()
    dummy_float_prev_layer.eval()
    quant_cdcm.eval()
    float_cdcm.eval()

    with torch.no_grad():
        # Generate inputs
        input_for_quant_cdcm = dummy_quant_prev_layer(initial_dummy_input_float) # QuantTensor
        input_for_float_cdcm = dummy_float_prev_layer(initial_dummy_input_float) # float Tensor
        compare_outputs(input_for_quant_cdcm, input_for_float_cdcm, "Input")

        # --- Run Inference ---
        print("Running inference on CDCM modules...")
        quant_output = quant_cdcm(input_for_quant_cdcm) # Should output QuantTensor
        float_output = float_cdcm(input_for_float_cdcm) # Should output float Tensor

    # --- Compare Outputs ---
    # compare_outputs handles QuantTensor or float Tensor input
    compare_outputs(quant_output, float_output, "CDCM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test QuantCDCM vs CDCM')
    # Example channels for micro model, stage 1 -> dil=4
    parser.add_argument('--in-channels', type=int, default=12, help='Input channels for CDCM')
    parser.add_argument('--out-channels', type=int, default=4, help='Output channels for CDCM (dil value)')
    parser.add_argument('--weight-bits', type=int, default=32, help='Weight bit width')
    parser.add_argument('--act-bits', type=int, default=32, help='Activation bit width')

    args = parser.parse_args()
    test_cdcm_equivalence(args)
