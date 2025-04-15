import argparse
import os
import time # Import time module
import hls4ml
import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.util.cleanup as cleanup
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean

def main():
    parser = argparse.ArgumentParser(description="Import ONNX model into hls4ml")
    parser.add_argument('--onnx-model', type=str, required=True, help='Path to the input ONNX model file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the hls4ml project.')
    parser.add_argument('--project-name', type=str, default='my_hls_model', help='Name for the hls4ml project.')
    parser.add_argument('--backend', type=str, default='Vitis', help='hls4ml backend (e.g., Vitis, Vivado).')
    parser.add_argument('--board', type=str, default=None, help='Target board for the backend (e.g., pynq-z2, u250).')
    parser.add_argument('--default-precision', type=str, default='fixed<16,6>', help='Default precision for hls4ml layers.')
    parser.add_argument('--io-type', type=str, default='io_stream', choices=['io_stream', 'io_parallel'], help='IO type for hls4ml model.')

    args = parser.parse_args()

    print(f"Loading ONNX model: {args.onnx_model}")
    start_time = time.time()
    model = ModelWrapper(args.onnx_model)
    print(f"Model loading took: {time.time() - start_time:.2f} seconds")

    print("\nApplying QONNX transformations...")
    total_transform_start_time = time.time()

    # --- Initial Cleanup ---
    print("Step 1: Initial Cleanup & Shape Inference...")
    step_start_time = time.time()
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    print(f"  Step 1 took: {time.time() - step_start_time:.2f} seconds")

    # --- Convert to Channels Last ---
    print("Step 2: ConvertToChannelsLastAndClean...")
    step_start_time = time.time()
    try:
        model = model.transform(ConvertToChannelsLastAndClean())
        print(f"  Step 2 took: {time.time() - step_start_time:.2f} seconds")
    except Exception as e:
        print(f"  ERROR during ConvertToChannelsLastAndClean: {e}")
        print(f"  Time before error: {time.time() - step_start_time:.2f} seconds")
        # Optionally save the model state just before the failing transform
        pre_channels_last_path = os.path.join(os.path.dirname(args.onnx_model), "pre_channels_last_" + os.path.basename(args.onnx_model))
        model.save(pre_channels_last_path)
        print(f"  Model state before error saved to: {pre_channels_last_path}")
        return # Exit if this critical step fails

    # --- Final Cleanup ---
    print("Step 3: Final Cleanup & DataType Inference...")
    step_start_time = time.time()
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(InferDataTypes())
    print(f"  Step 3 took: {time.time() - step_start_time:.2f} seconds")

    print(f"\nTotal QONNX transformation time: {time.time() - total_transform_start_time:.2f} seconds")

    # Save the cleaned model (optional, for debugging)
    cleaned_model_path = os.path.join(os.path.dirname(args.onnx_model), "cleaned_" + os.path.basename(args.onnx_model))
    model.save(cleaned_model_path)
    print(f"Cleaned QONNX model saved to: {cleaned_model_path}")

    print("\nGenerating hls4ml configuration...")
    start_time = time.time()
    hls_config = hls4ml.utils.config_from_onnx_model(
        model,
        granularity='name', # Required for QONNX/quantized models
        backend=args.backend,
        default_precision=args.default_precision
    )
    print(f"Config generation took: {time.time() - start_time:.2f} seconds")

    # --- Optional: Modify hls_config here if needed ---
    # For example, setting specific layer precisions or reuse factors:
    # hls_config['LayerName']['Precision'] = 'ap_fixed<10,4>'
    # hls_config['LayerName']['ReuseFactor'] = 2
    # print("hls4ml Configuration:")
    # hls4ml.utils.print_dict(hls_config)
    # ----------------------------------------------------

    print(f"\nConverting model to hls4ml project: {args.project_name} in {args.output_dir}")
    start_time = time.time()
    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        output_dir=args.output_dir,
        project_name=args.project_name,
        backend=args.backend,
        board=args.board, # Pass board info if provided
        io_type=args.io_type,
        hls_config=hls_config,
    )
    print(f"hls4ml conversion took: {time.time() - start_time:.2f} seconds")

    print("\nCompiling hls4ml model...")
    start_time = time.time()
    hls_model.compile()
    print(f"hls4ml compilation took: {time.time() - start_time:.2f} seconds")

    # --- Optional: Run C Simulation or Build ---
    # print("Running C Simulation...")
    # # Need input data (e.g., from ONNX export or test set)
    # # X_input = ...
    # # y_hls, y_qonnx = hls_model.trace(X_input)
    # # print("C Simulation complete.")

    # print("Building HLS project...")
    # hls_model.build(csim=False) # Set csim=True to run C simulation during build
    # print("HLS build complete.")
    # -------------------------------------------

    print(f"\nSuccessfully created and compiled hls4ml project in: {os.path.join(args.output_dir, args.project_name)}")

if __name__ == "__main__":
    main()
