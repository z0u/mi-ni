import torch


def get_device(model: torch.nn.Module) -> str:
    """
    Get the device type of the model's parameters.

    Args:
        model: The PyTorch model
    Returns:
        Device type as a string (e.g., 'cuda', 'cpu')
    """
    try:
        # Attempt to get the device type from the model's parameters
        return next(model.parameters()).device.type
    except StopIteration:
        # If the model has no parameters, default to 'cpu'
        return 'cpu'


def get_dtype(dtype_str: str) -> torch.dtype:
    """
    Safely convert a string configuration value to a torch dtype.

    Supports all common PyTorch dtype names.

    Args:
        dtype_str: String representation of torch dtype (e.g., 'float16', 'bfloat16', 'half')

    Returns:
        The corresponding torch.dtype

    Raises:
        ValueError: If the provided dtype string is not a valid torch dtype

    See:
        https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
    """
    return _dtype_map[dtype_str.lower()]


# Map of all supported dtype names to their torch.dtype objects
_dtype_map = {
    # Floating point types (most relevant for AMP)
    'float16': torch.float16,
    'half': torch.float16,
    'float32': torch.float32,
    'float': torch.float32,
    'bfloat16': torch.bfloat16,
    'float64': torch.float64,
    'double': torch.float64,
    # Complex types
    'complex64': torch.complex64,
    'cfloat': torch.cfloat,
    'complex128': torch.complex128,
    'cdouble': torch.cdouble,
    # Integer types
    'uint8': torch.uint8,
    'int8': torch.int8,
    'int16': torch.int16,
    'short': torch.short,
    'int32': torch.int32,
    'int': torch.int,
    'int64': torch.int64,
    'long': torch.long,
    # Boolean type
    'bool': torch.bool,
}


def supports_bfloat16(device_type: str = 'cuda') -> bool:
    """Check if bfloat16 is supported on the given device."""
    if device_type == 'cpu':
        try:
            # Create a small tensor and attempt conversion
            x = torch.ones(1, dtype=torch.float32)
            _ = x.to(torch.bfloat16)
            return True
        except RuntimeError:
            return False

    elif device_type.startswith('cuda'):
        if not torch.cuda.is_available():
            return False
        # On CUDA devices, bfloat16 requires Ampere (SM 8.0) or higher architecture
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return props.major >= 8

    elif device_type == 'mps':
        # For Apple Silicon
        return hasattr(torch, 'mps') and torch.mps.is_available()

    return False
