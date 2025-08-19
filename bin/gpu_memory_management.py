import threading
import gc
import GPUtil

# Global variables for RMM management
_rmm_initialized = False
_rmm_lock = threading.Lock()
_gpu_available = False


def get_gpu_info_simple():
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []

        for gpu in gpus:
            gpu_data = {
                'GPU ID': gpu.id,
                'Name': gpu.name,
                'Total memory (mb)': gpu.memoryTotal,
                'Used memory (mb)': gpu.memoryUsed,
                'Free memory (mb)': gpu.memoryFree,
                'GPU load': gpu.load * 100  # Convert to percentage
            }
            gpu_info.append(gpu_data)

        return gpu_info

    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None


def initialize_rmm_once() -> bool:
    """Initialize RMM once per process to avoid multiple initialization issues"""
    global _rmm_initialized, _gpu_available
    
    with _rmm_lock:
        if _rmm_initialized:
            return _gpu_available
            
        try:
            import rmm
            import cudf
            from cuml.decomposition import TruncatedSVD
            from cuml.manifold import TSNE
            from cuml import UMAP
            from cuml.preprocessing import StandardScaler as cuStandardScaler
            
            # Initialize RMM with optimized settings
            rmm.reinitialize(
                pool_allocator=True,
                managed_memory=False,  # Better for single GPU
                initial_pool_size="2GB",  # Conservative start
                maximum_pool_size="20GB"  # Leave room for other processes
            )
            
            _rmm_initialized = True
            _gpu_available = True
            gpu_info = get_gpu_info_simple()
            print(f"\nRMM successfully  initialized for {gpu_info}\n")
            return True
            
        except Exception as e:
            print(f"Failed to initialize RMM: {e}")
            _rmm_initialized = True  # Don't try again
            _gpu_available = False
            return False

def cleanup_gpu_memory():
    """Clean up GPU memory between datasets"""
    try:
        import cudf
        import cupy as cp
        # Force garbage collection
        gc.collect()
        # Clear CuPy memory pool
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        print("GPU memory cleaned")
    except:
        pass
