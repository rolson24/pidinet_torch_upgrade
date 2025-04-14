import torch
import argparse
import sys
import os
import brevitas.nn as qnn
import torch.nn as nn # Import nn
import torch.nn.functional as F # Import F for interpolate

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
    output, checking backbone and fusion/classifier stages sequentially.
    """
    # Update description
    print("Starting test: Full QuantPiDiNet vs Converted PiDiNet (Backbone + Fusion/Classifier)")
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

    # --- Synchronize Backbone Weights ---
    print("\n--- Synchronizing Backbone Weights ---")
    sync_weights(quant_model_full.init_block, converted_model_full.init_block, "init_block")
    for i in range(1, 5): # Stages 1 to 4
        for j in range(1, 5): # Blocks 1 to 4 within stage
            quant_block_name = f"block{i}_{j}"
            converted_block_name = f"block{i}_{j}"
            if hasattr(quant_model_full, quant_block_name) and hasattr(converted_model_full, converted_block_name):
                 sync_weights(getattr(quant_model_full, quant_block_name), getattr(converted_model_full, converted_block_name), quant_block_name)
            else:
                 print(f"Warning: Block {quant_block_name} not found in one or both models.")


    # --- Create Initial Dummy Input ---
    print("\n--- Creating Initial Dummy Input ---")
    H, W = 256, 256 # Keep consistent size
    initial_dummy_input_float = torch.randn(1, 3, H, W)

    # --- Run Backbone Inference ---
    print("\n--- Running Backbone Inference ---")
    backbone_outputs_quant = {}
    backbone_outputs_converted = {}
    with torch.no_grad():
        # Quant Model Path
        quant_x = quant_model_full.init_block(initial_dummy_input_float)
        quant_x1 = quant_model_full.block1_3(quant_model_full.block1_2(quant_model_full.block1_1(quant_x)))
        quant_x2 = quant_model_full.block2_4(quant_model_full.block2_3(quant_model_full.block2_2(quant_model_full.block2_1(quant_x1))))
        quant_x3 = quant_model_full.block3_4(quant_model_full.block3_3(quant_model_full.block3_2(quant_model_full.block3_1(quant_x2))))
        quant_x4 = quant_model_full.block4_4(quant_model_full.block4_3(quant_model_full.block4_2(quant_model_full.block4_1(quant_x3))))
        backbone_outputs_quant = {'x1': quant_x1, 'x2': quant_x2, 'x3': quant_x3, 'x4': quant_x4}

        # Converted Model Path
        converted_x = converted_model_full.init_block(initial_dummy_input_float)
        converted_x1 = converted_model_full.block1_3(converted_model_full.block1_2(converted_model_full.block1_1(converted_x)))
        converted_x2 = converted_model_full.block2_4(converted_model_full.block2_3(converted_model_full.block2_2(converted_model_full.block2_1(converted_x1))))
        converted_x3 = converted_model_full.block3_4(converted_model_full.block3_3(converted_model_full.block3_2(converted_model_full.block3_1(converted_x2))))
        converted_x4 = converted_model_full.block4_4(converted_model_full.block4_3(converted_model_full.block4_2(converted_model_full.block4_1(converted_x3))))
        backbone_outputs_converted = {'x1': converted_x1, 'x2': converted_x2, 'x3': converted_x3, 'x4': converted_x4}

    # --- Compare Backbone Final Outputs (Prerequisite Check) ---
    print("\n--- Comparing Backbone Final Outputs (Block4_4) ---")
    if hasattr(quant_x4, 'value'):
        quant_backbone_final_float = quant_x4.value.detach()
    else:
        quant_backbone_final_float = quant_x4.detach()

    are_backbone_close = torch.allclose(quant_backbone_final_float, converted_x4, atol=1e-5)
    max_backbone_diff = torch.max(torch.abs(quant_backbone_final_float - converted_x4)).item()

    print(f"Backbone (Block 4_4) outputs close: {are_backbone_close} (Max diff: {max_backbone_diff:.6e})")

    if not are_backbone_close:
        print("Backbone test failed. Aborting fusion/classifier test.")
        return False

    # --- Synchronize Fusion/Classifier Weights ---
    print("\n--- Synchronizing Fusion/Classifier Weights ---")
    if args.sa:
        sync_weights(quant_model_full.attentions, converted_model_full.attentions, "attentions")
    if args.dil:
        sync_weights(quant_model_full.dilations, converted_model_full.dilations, "dilations")
    sync_weights(quant_model_full.conv_reduces, converted_model_full.conv_reduces, "conv_reduces")
    sync_weights(quant_model_full.classifier, converted_model_full.classifier, "classifier")
    # No need to sync quant_cat or final_sigmoid as they don't have weights

    # --- Run Fusion/Classifier Forward Pass ---
    print("\n--- Running Fusion/Classifier Forward Pass ---")
    with torch.no_grad():
        # Quant Model Fusion/Classifier Path
        quant_x_fuses = []
        quant_inputs = [backbone_outputs_quant[f'x{i+1}'] for i in range(4)]
        if quant_model_full.sa and quant_model_full.dil is not None:
            for i, xi in enumerate(quant_inputs):
                quant_x_fuses.append(quant_model_full.attentions[i](quant_model_full.dilations[i](xi)))
        elif quant_model_full.sa:
            for i, xi in enumerate(quant_inputs):
                quant_x_fuses.append(quant_model_full.attentions[i](xi))
        elif quant_model_full.dil is not None:
            for i, xi in enumerate(quant_inputs):
                quant_x_fuses.append(quant_model_full.dilations[i](xi))
        else:
            quant_x_fuses = quant_inputs

        quant_e = [F.interpolate(quant_model_full.conv_reduces[i](quant_x_fuses[i]), (H, W), mode="bilinear", align_corners=False) for i in range(4)]
        quant_cat_out = torch.cat(quant_e, dim=1)
        # Need to handle potential QuantTensor output from interpolate if input was QuantTensor
        # Assuming interpolate outputs float tensor based on typical behavior
        quant_cat_out_requant = quant_model_full.quant_cat(quant_cat_out) # Requantize before classifier
        quant_output_final_layer = quant_model_full.classifier(quant_cat_out_requant)
        quant_outputs = [quant_model_full.final_sigmoid(e) for e in quant_e]
        quant_outputs.append(quant_model_full.final_sigmoid(quant_output_final_layer))

        # Converted Model Fusion/Classifier Path
        converted_x_fuses = []
        converted_inputs = [backbone_outputs_converted[f'x{i+1}'] for i in range(4)]
        if converted_model_full.sa and converted_model_full.dil is not None:
            for i, xi in enumerate(converted_inputs):
                converted_x_fuses.append(converted_model_full.attentions[i](converted_model_full.dilations[i](xi)))
        elif converted_model_full.sa:
            for i, xi in enumerate(converted_inputs):
                converted_x_fuses.append(converted_model_full.attentions[i](xi))
        elif converted_model_full.dil is not None:
            for i, xi in enumerate(converted_inputs):
                converted_x_fuses.append(converted_model_full.dilations[i](xi))
        else:
            converted_x_fuses = converted_inputs

        converted_e = [F.interpolate(converted_model_full.conv_reduces[i](converted_x_fuses[i]), (H, W), mode="bilinear", align_corners=False) for i in range(4)]
        converted_cat_out = torch.cat(converted_e, dim=1)
        converted_output_final_layer = converted_model_full.classifier(converted_cat_out)
        # Apply sigmoid manually for converted model
        converted_outputs = [torch.sigmoid(e) for e in converted_e]
        converted_outputs.append(torch.sigmoid(converted_output_final_layer))


    # --- Compare Final Outputs ---
    print("\n--- Comparing Final Model Outputs ---")
    all_outputs_close = True
    max_overall_diff = 0.0

    if len(quant_outputs) != len(converted_outputs):
        print(f"Error: Final output list lengths differ! Quant: {len(quant_outputs)}, Converted: {len(converted_outputs)}")
        return False

    for i, (q_out, c_out) in enumerate(zip(quant_outputs, converted_outputs)):
        # Both should be float tensors now after final sigmoid
        quant_output_float = q_out.detach()
        converted_output_float = c_out.detach()

        if quant_output_float.shape != converted_output_float.shape:
            print(f"Final Output {i} shape mismatch! Quant: {quant_output_float.shape}, Converted: {converted_output_float.shape}")
            all_outputs_close = False
            continue

        are_close = torch.allclose(quant_output_float, converted_output_float, atol=1e-5) # Keep tolerance tight for 32-bit
        max_diff = torch.max(torch.abs(quant_output_float - converted_output_float)).item()
        max_overall_diff = max(max_overall_diff, max_diff)

        print(f"Final Output {i} is close: {are_close} (Max diff: {max_diff:.6e})")
        if not are_close:
            all_outputs_close = False

    print(f"\nOverall maximum absolute difference in final outputs: {max_overall_diff:.6e}")
    if all_outputs_close:
        print("Test PASSED for Full Model (Backbone + Fusion/Classifier)!")
    else:
        print("Test FAILED in Fusion/Classifier Stage!")

    return all_outputs_close

if __name__ == "__main__":
    # ... (argument parsing remains the same) ...
    parser = argparse.ArgumentParser(description='Test Full Quantized PiDiNet vs Converted PiDiNet (Backbone + Fusion)')
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
