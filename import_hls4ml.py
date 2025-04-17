import argparse
import os
import time
import hls4ml
import onnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
import warnings


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
    parser.add_argument('--target-opset', type=int, default=9, help='Target ONNX opset version (default: 9 for compatibility)')
    parser.add_argument('--skip-cleanup', action='store_true', help='Skip QONNX cleanup (use if cleanup fails)')

    args = parser.parse_args()

    print(f"Loading ONNX model: {args.onnx_model}")
    start_time = time.time()
    
    # First, check model version and downgrade if needed
    try:
        original_model = onnx.load(args.onnx_model)
        current_opset = original_model.opset_import[0].version
        print(f"Model opset version: {current_opset}, target: {args.target_opset}")
        
        if current_opset > args.target_opset:
            print(f"Downgrading model from opset {current_opset} to {args.target_opset}")
            # Create a temporary downgraded model
            from onnx import version_converter
            try:
                converted_model = version_converter.convert_version(original_model, args.target_opset)
                temp_model_path = args.onnx_model.replace('.onnx', f'_opset{args.target_opset}.onnx')
                onnx.save(converted_model, temp_model_path)
                print(f"Saved downgraded model to {temp_model_path}")
                args.onnx_model = temp_model_path  # Use the downgraded model
            except Exception as e:
                warnings.warn(f"Could not downgrade model: {e}\nWill try to proceed with original model.")
    except Exception as e:
        print(f"Warning: Could not check/downgrade model version: {e}")
        print("Will attempt to proceed with original model.")
    
    # Load model with QONNX
    try:
        model = ModelWrapper(args.onnx_model)
        print(f"Model loading took: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\nPreprocessing ONNX model...")
    transform_start_time = time.time()
    
    # Apply cleanup if not skipped
    if not args.skip_cleanup:
        try:
            print("Cleaning up model...")
            model = cleanup_model(model)
            print("Cleanup complete.")
            # save the cleaned model if needed
            model.save(args.onnx_model.replace('.onnx', '_cleaned.onnx'))
        except Exception as e:
            print(f"Error during cleanup: {e}")
            print("Consider using --skip-cleanup if this persists.")
            return
    else:
        print("Skipping cleanup as requested.")

    # Apply channels-last conversion if not skipped
    if not args.skip_channels_last:
        try:
            print("Converting model to channels-last format...")
            model = model.transform(ConvertToChannelsLastAndClean())
            print("Channels-last conversion complete.")
        except Exception as e:
            print(f"Error during channels-last conversion: {e}")
            print("Will attempt to proceed with model as-is.")
    else:
        print("Skipping channels-last conversion as requested.")

    print(f"Preprocessing took: {time.time() - transform_start_time:.2f} seconds")

    # Generate hls4ml configuration
    print("\nGenerating hls4ml configuration...")
    start_time = time.time()
    try:
        hls_config = hls4ml.utils.config_from_onnx_model(
            model,
            granularity='name',
            backend=args.backend,
            default_precision=args.default_precision
        )
        print(f"Config generation took: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error generating hls4ml configuration: {e}")
        return

    # Convert to hls4ml
    print(f"\nConverting model to hls4ml project: {args.project_name} in {args.output_dir}")
    start_time = time.time()
    try:
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model,
            output_dir=args.output_dir,
            project_name=args.project_name,
            backend=args.backend,
            # board=args.board,
            io_type=args.io_type,
            hls_config=hls_config,
        )
        print(f"hls4ml conversion took: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error during hls4ml conversion: {e}")
        return

    # Compile the model
    print("\nCompiling hls4ml model...")
    start_time = time.time()
    try:
        hls_model.compile()
        print(f"hls4ml compilation took: {time.time() - start_time:.2f} seconds")
        print(f"\nSuccessfully created and compiled hls4ml project in: {os.path.join(args.output_dir, args.project_name)}")
    except Exception as e:
        print(f"Error during compilation: {e}")
        print("The hls4ml project was created but compilation failed.")

if __name__ == "__main__":
    main()
