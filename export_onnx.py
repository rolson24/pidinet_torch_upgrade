import torch
import argparse
import os
# Add necessary imports for model loading
import models.quant_pidinet as quant_models
from argparse import Namespace # Import Namespace

# Pass checkpoint path explicitly, make it optional
def load_model(args, checkpoint_path=None):
    model_factory = getattr(quant_models, args.model)
    model = model_factory(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)

    # Load checkpoint only if path is provided
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        # Use the passed checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Handle potential DataParallel wrapping in checkpoint
        state_dict = checkpoint['state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint load results: {load_result}") # Print missing/unexpected keys
    else:
        print("Initializing model with random weights (no checkpoint loaded).")

    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Export quantized PiDiNet to ONNX")
    parser.add_argument('--model', type=str, required=True, help='quantized model name (e.g. quant_pidinet_micro)')
    parser.add_argument('--config', type=str, required=True, help='model config (e.g. carv4)')
    parser.add_argument('--weight-bits', type=int, default=8)
    parser.add_argument('--act-bits', type=int, default=8)
    # Make checkpoint optional
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint .pth file (optional)')
    parser.add_argument('--onnx-out', type=str, required=True, help='output ONNX filename')
    parser.add_argument('--input-shape', type=int, nargs=3, default=[3, 480, 480], help='input shape as C H W')
    parser.add_argument('--sa', action='store_true', help='use CSAM in pidinet')
    parser.add_argument('--dil', action='store_true', help='use CDCM in pidinet')
    # Add flag for random weights
    parser.add_argument('--random-weights', action='store_true', help='Export model with random weights (ignore checkpoint)')
    args = parser.parse_args()

    # Validate arguments
    if not args.random_weights and args.checkpoint is None:
        parser.error("--checkpoint is required unless --random-weights is specified.")
    if args.random_weights and args.checkpoint is not None:
        print("Warning: --checkpoint provided but --random-weights is set. Checkpoint will be ignored.")
        args.checkpoint = None # Ensure checkpoint is not loaded

    # Prepare args namespace for model factory, using parsed args directly
    model_args = Namespace(
        config=args.config,
        sa=args.sa, # Use parsed sa flag
        dil=args.dil, # Use parsed dil flag
        # Add other args if the model factory expects them
    )

    # Pass the parsed args for model parameters and the checkpoint path separately
    model = load_model(
        args=Namespace( # Create Namespace with only necessary args for factory
             model=args.model,
             config=args.config,
             sa=args.sa,
             dil=args.dil,
             weight_bits=args.weight_bits,
             act_bits=args.act_bits
        ),
        checkpoint_path=args.checkpoint # Pass the checkpoint path (or None)
    )

    dummy_input = torch.randn(1, *args.input_shape)

    # Ensure model is on CPU for export
    model.cpu()

    # Use export_onnx_qcdq instead of export_finn_onnx
    from brevitas.onnx import export_onnx_qcdq # Import the correct function
    from brevitas.export import export_qonnx # Import export_qonnx if needed

    # print(f"Exporting model to ONNX (QCDQ format) at: {args.onnx_out}")
    # export_onnx_qcdq(
    #     model,
    #     args=dummy_input, # Pass dummy_input as the 'args' argument
    #     export_path=args.onnx_out,
    #     opset_version=13 # Use opset 13 as in the example, or adjust if needed
    #     # Add other relevant arguments for export_onnx_qcdq if necessary
    # )

    print(f"Exporting model to ONNX (QONNX format) at: {args.onnx_out}")

    export_qonnx(
        model,
        dummy_input,
        export_path=args.onnx_out,
        opset_version=13,
        input_names=['input'],
        output_names=['output'], # Adjust if the model returns a list
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Adjust for dynamic axes if needed
    )

    # Original torch.onnx.export (keep commented as fallback/alternative)
    # torch.onnx.export(
    #     model,
    #     dummy_input,
    #     args.onnx_out,
    #     export_params=True,
    #     opset_version=11,
    #     do_constant_folding=True,
    #     input_names=['input'],
    #     output_names=['output'], # Model returns a list, adjust output names if needed
    #     # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # May need adjustment for list output
    # )
    print(f"Exported ONNX model to {args.onnx_out}")

if __name__ == "__main__":
    main()
