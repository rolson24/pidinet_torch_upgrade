import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import hls4ml
    print(f"hls4ml imported successfully, version: {hls4ml.__version__}")
except ImportError as e:
    print(f"Failed to import hls4ml: {e}")

try:
    import qonnx
    print(f"qonnx imported successfully")
    
    # Test importing specific modules
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup_model
    from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
    print("All qonnx submodules imported successfully")
except ImportError as e:
    print(f"Failed to import qonnx: {e}")
