import argparse
import os
import time # Import time module
import hls4ml
# import qonnx.core.onnx_exec as oxe
# from qonnx.core.modelwrapper import ModelWrapper
# import qonnx.util.cleanup as cleanup
# from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
# from qonnx.transformation.infer_shapes import InferShapes
# from qonnx.transformation.fold_constants import FoldConstants
# from qonnx.transformation.infer_datatypes import InferDataTypes
# # Import the combined transformation again
# from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
# # Remove problematic/speculative imports
# # from qonnx.transformation.streamline import Streamline
# # from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
# # from qonnx.transformation.general import ConvertSubToAdd, ConvertDivToMul
# # from qonnx.transformation.batchnorm_to_affine import AbsorbBNIntoConv, AbsorbScalarMulAddIntoConv

from qonnx.util.cleanup import cleanup_model
# from qonnx.util.to_channels_last import to_channels_last
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.core.modelwrapper import ModelWrapper


def main():
    parser = argparse.ArgumentParser(description="Import ONNX model into hls4ml")
    parser.add_argument('--onnx-model', type=str, required=True, help='Path to the input ONNX model file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the hls4ml project.')
    parser.add_argument('--project-name', type=str, default='my_hls_model', help='Name for the hls4ml project.')
    parser.add_argument('--backend', type=str, default='Vitis', help='hls4ml backend (e.g., Vitis, Vivado).')
    parser.add_argument('--board', type=str, default=None, help='Target board for the backend (e.g., pynq-z2, u250).')
    parser.add_argument('--default-precision', type=str, default='fixed<16,6>', help='Default precision for hls4ml layers.')
    parser.add_argument('--io-type', type=str, default='io_stream', choices=['io_stream', 'io_parallel'], help='IO type for hls4ml model.')
    parser.add_argument('--skip-channels-last', action='store_true', help='Skip channels-last conversion (model already has channels-last format)')

    args = parser.parse_args()

    print(f"Loading ONNX model: {args.onnx_model}")
    start_time = time.time()
    model = ModelWrapper(args.onnx_model)
    print(f"Model loading took: {time.time() - start_time:.2f} seconds")

    print("\nApplying QONNX transformations...")
    total_transform_start_time = time.time()

    # --- Initial Cleanup & Simplification ---
    print("Cleaning up model...")
    model = cleanup_model(model, remove_unused=True, remove_redundant=True, remove_dead=True)
    print("Cleanup complete.")
    print(f"Model cleanup took: {time.time() - total_transform_start_time:.2f} seconds")
    total_transform_start_time = time.time()
    # --- Convert to Channels Last Format ---
    if not args.skip_channels_last:
        print("Converting model to channels-last format...")
        model = model.transform(ConvertToChannelsLastAndClean())
        print("Channels-last conversion complete.")
    print(f"Channels-last conversion took: {time.time() - total_transform_start_time:.2f} seconds")

    # --- Optional: Modify hls_config here if needed ---
    # For example, setting specific layer precisions or reuse factors:
    # hls_config['LayerName']['Precision'] = 'ap_fixed<10,4>'
    # hls_config['LayerName']['ReuseFactor'] = 2
    # print("hls4ml Configuration:")
    # hls4ml.utils.print_dict(hls_config)
    # ----------------------------------------------------

    print("\nGenerating hls4ml configuration...")
    start_time = time.time()
    hls_config = hls4ml.utils.config_from_onnx_model(
        model,
        granularity='name', # Required for QONNX/quantized models
        backend=args.backend,
        default_precision=args.default_precision
    )
    print(f"Config generation took: {time.time() - start_time:.2f} seconds")
    print("hls4ml configuration generated.")

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
