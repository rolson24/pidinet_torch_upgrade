import torch
import argparse
import os

def load_model(args):
    import models.quant_pidinet as quant_models
    model_factory = getattr(quant_models, args.model)
    model = model_factory(args, weight_bit_width=args.weight_bits, act_bit_width=args.act_bits)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
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
    args = parser.parse_args()

    # Prepare dummy args for model factory
    class DummyArgs:
        pass
    dummy_args = DummyArgs()
    dummy_args.config = args.config
    dummy_args.sa = True
    dummy_args.dil = True

    model = load_model(argparse.Namespace(
        model=args.model,
        config=args.config,
        sa=dummy_args.sa,
        dil=dummy_args.dil,
        weight_bits=args.weight_bits,
        act_bits=args.act_bits
    ))

    dummy_input = torch.randn(1, *args.input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_out,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported ONNX model to {args.onnx_out}")

if __name__ == "__main__":
    main()
