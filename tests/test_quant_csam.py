import torch
import argparse
import sys
import os
import brevitas.nn as qnn
import torch.nn as nn

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.pidinet import CSAM
from models.quant_pidinet import QuantCSAM

# Copy the sync_weights function
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
             # Handle bias initialization difference: QuantConv2d might have bias even if float doesn't if bias_quant is set
             # If float doesn't have bias, but quant does, we can ignore the quant bias for sync purposes.
             if 'bias' not in name: # Only warn if it's not a bias mismatch
                 print(f"Warning: Parameter {name} from float module not found in quant module {module_name}. Using original float param.")
             new_float_state_dict[name] = param_float

    # Check for keys in quant_module not in float_module (like bias in QuantConv2d vs nn.Conv2d(bias=False))
    # This ensures load_state_dict doesn't complain about unexpected keys if strict=True
    keys_to_remove = []
    for name in new_float_state_dict.keys():
        if name not in float_state_dict:
             print(f"Notice: Parameter {name} found in quant module but not in float module {module_name}. Removing from sync dict.")
             keys_to_remove.append(name)
    for key in keys_to_remove:
        del new_float_state_dict[key]


    float_module.load_state_dict(new_float_state_dict, strict=False) # Use strict=False for flexibility
    print(f"Weight synchronization complete for {module_name}.")


def compare_outputs(quant_output, converted_output, test_name):
    """Compares the float outputs."""
    print(f"Comparing outputs for {test_name}...")
    if hasattr(quant_output, 'value'):
        quant_output_float = quant_output.value.detach()
    else:
        quant_output_float = quant_output.detach()

    # Ensure converted_output is also detached if it's not already
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

def test_csam_equivalence(args):
    """Tests QuantCSAM vs CSAM."""
    print(f"Testing CSAM equivalence with {args.channels} channels")
    print(f"Args: {args}")

    channels = args.channels
    act_bits = args.act_bits
    weight_bits = args.weight_bits

    # --- Instantiate Modules ---
    quant_csam = QuantCSAM(channels, act_bit_width=act_bits, weight_bit_width=weight_bits)
    float_csam = CSAM(channels)

    # --- Synchronize Weights ---
    sync_weights(quant_csam, float_csam, "CSAM")

    # --- Create Dummy Input ---
    # Simulate input from a previous quantized layer vs float layer
    print("Simulating preceding layer input...")
    prev_channels = channels # Assume input channels match CSAM channels for simplicity here
    dummy_input_shape = (1, prev_channels, 64, 64) # Example spatial size
    initial_dummy_input_float = torch.randn(*dummy_input_shape)

    # Simulate a previous QuantConv layer outputting QuantTensor
    # Use a simple 1x1 conv for simulation
    dummy_quant_prev_layer = qnn.QuantConv2d(prev_channels, channels, kernel_size=1,
                                             weight_bit_width=weight_bits,
                                             bias=False) # Keep it simple
    # Simulate a previous nn.Conv layer outputting float Tensor
    dummy_float_prev_layer = nn.Conv2d(prev_channels, channels, kernel_size=1, bias=False)

    # Sync weights of dummy previous layers
    sync_weights(dummy_quant_prev_layer, dummy_float_prev_layer, "Dummy Previous Layer")

    dummy_quant_prev_layer.eval()
    dummy_float_prev_layer.eval()
    quant_csam.eval()
    float_csam.eval()

    with torch.no_grad():
        # Generate input for QuantCSAM (QuantTensor)
        input_for_quant_csam = dummy_quant_prev_layer(initial_dummy_input_float)
        # Generate input for CSAM (float Tensor)
        input_for_float_csam = dummy_float_prev_layer(initial_dummy_input_float)

        # --- Run Inference ---
        print("Running inference on CSAM modules...")
        quant_output = quant_csam(input_for_quant_csam)
        float_output = float_csam(input_for_float_csam)

    # --- Compare Outputs ---
    compare_outputs(quant_output, float_output, "CSAM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test QuantCSAM vs CSAM')
    parser.add_argument('--channels', type=int, default=12, help='Input/output channels for CSAM')
    parser.add_argument('--weight-bits', type=int, default=32, help='Weight bit width')
    parser.add_argument('--act-bits', type=int, default=32, help='Activation bit width')

    args = parser.parse_args()
    test_csam_equivalence(args)
