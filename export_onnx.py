import torch
import argparse
import os
# Add necessary imports for model loading
import models.quant_pidinet as quant_models
from argparse import Namespace # Import Namespace

# Pass checkpoint path explicitly
def load_model(args, checkpoint_path):
    model_factory = getattr(quant_models, args.model)
    model = model_factory(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
    # Use the passed checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # Handle potential DataParallel wrapping in checkpoint
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Export quantized PiDiNet to ONNX")
    parser.add_argument('--model', type=str, required=True, help='quantized model name (e.g. quant_pidinet_micro)')
    parser.add_argument('--config', type=str, required=True, help='model config (e.g. carv4)')
    parser.add_argument('--weight-bits', type=int, default=8)
    parser.add_argument('--act-bits', type=int, default=8)
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint .pth file')
    parser.add_argument('--onnx-out', type=str, required=True, help='output ONNX filename')
    parser.add_argument('--input-shape', type=int, nargs=3, default=[3, 480, 480], help='input shape as C H W')
    # Add arguments for SA and DIL flags to match model creation needs
    parser.add_argument('--sa', action='store_true', help='use CSAM in pidinet')
    parser.add_argument('--dil', action='store_true', help='use CDCM in pidinet')
    args = parser.parse_args()

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
        checkpoint_path=args.checkpoint # Pass the checkpoint path from original args
    )

    dummy_input = torch.randn(1, *args.input_shape)

    # Ensure model is on CPU for export
    model.cpu()

    # Brevitas ONNX export requires specific setup
    from brevitas.onnx import export_finn_onnx
    export_finn_onnx(
        model,
        input_shape=(1, *args.input_shape), # Pass input shape tuple
        export_path=args.onnx_out,
        # opset_version=11 # FINN export usually defaults or uses a specific version
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
