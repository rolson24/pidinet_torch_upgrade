"""
Export a quantized PiDiNet model directly to QCDQ ONNX format.
"""
import argparse
import torch
import os
from brevitas.export import export_onnx_qcdq # Use QCDQ export function

# Import model factory functions
from models.quant_pidinet import quant_pidinet_micro, quant_pidinet_tiny, quant_pidinet_small, quant_pidinet
from models.config import config_model_converted

def main():
    parser = argparse.ArgumentParser(description="Export Quantized PiDiNet to QCDQ ONNX")

    # Model configuration
    parser.add_argument('--model', type=str, default='quant_pidinet_micro',
                      choices=['quant_pidinet_micro', 'quant_pidinet_tiny', 'quant_pidinet_small', 'quant_pidinet'],
                      help='Model architecture to export')
    parser.add_argument('--config', type=str, default='carv4', help='Model PDC configuration')
    parser.add_argument('--sa', action='store_true', help='Use CSAM in model')
    parser.add_argument('--dil', action='store_true', help='Use CDCM in model')

    # Quantization parameters
    parser.add_argument('--weight-bits', type=int, default=8, help='Weight bit width')
    parser.add_argument('--act-bits', type=int, default=8, help='Activation bit width')

    # Input/Output specs
    parser.add_argument('--input-shape', type=int, nargs='+', default=[1, 3, 480, 480], # NCHW format
                      help='Input shape in NCHW format')
    parser.add_argument('--qcdq-out', type=str, default='model_qcdq.onnx', help='Output QCDQ ONNX filename')
    parser.add_argument('--opset-version', type=int, default=13, help='ONNX opset version (13 is common for QCDQ)')

    # Checkpoint loading
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--random-weights', action='store_true', help='Use random weights (no checkpoint)')

    args = parser.parse_args()

    # Create argument namespace with required attributes for model factory
    class ModelArgs:
        def __init__(self, config, sa, dil):
            self.config = config
            self.sa = sa
            self.dil = dil

    model_args = ModelArgs(args.config, args.sa, args.dil)

    # Choose the appropriate model factory
    model_factories = {
        'quant_pidinet_micro': quant_pidinet_micro,
        'quant_pidinet_tiny': quant_pidinet_tiny,
        'quant_pidinet_small': quant_pidinet_small,
        'quant_pidinet': quant_pidinet
    }

    # Create model
    print(f"Creating {args.model} with config {args.config}")
    model_factory = model_factories[args.model]
    model = model_factory(model_args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)

    # Load weights if checkpoint is provided
    if args.checkpoint and not args.random_weights:
        print(f"Loading weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # Adjust key loading based on whether model was saved with DataParallel
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        print("Using random weights")

    # Set model to evaluation mode
    model.eval()


    # --- Count and Print Parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"------------------------------------")
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters:   {trainable_params:,}")
    print(f"------------------------------------")
    # --- End Parameter Counting ---

    
    # Prepare dummy input (NCHW format)
    dummy_input = torch.randn(args.input_shape)

    # Export directly to QCDQ ONNX
    print(f"Exporting to QCDQ ONNX: {args.qcdq_out} with opset {args.opset_version}")
    export_onnx_qcdq(
        module=model,
        input_t=dummy_input,
        export_path=args.qcdq_out,
        input_names=['input'],
        # Output names might need adjustment based on your model's actual outputs
        output_names=['output_s1', 'output_s2', 'output_s3', 'output_s4', 'output_final'],
        opset_version=args.opset_version
    )

    print(f"Export complete. You can now try importing {args.qcdq_out} with import_hls4ml.py.")
    print("Note: You might need to adjust the import script if it specifically relies on QONNX transformations that are not applicable to QCDQ.")

if __name__ == "__main__":
    main()
