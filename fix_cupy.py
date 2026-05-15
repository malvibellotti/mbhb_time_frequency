#import sys

#if 'cupy' in sys.modules:
    
#    import cupy.cuda.runtime # type:ignore

#    def patch_cupy_runtime() -> None:
#        """
#        Patches cupy.cuda.get_local_runtime_version at runtime to fix
#        compatibility issues with gpubackendtools.
#        """
#        try:
            # Check if the specific problematic internal function is missing
#            if not hasattr(cupy.cuda.runtime, '_getLocalRuntimeVersion'):
                # specific type alias for readability
#                from typing import Callable
#                # Define the replacement function using the direct runtime call
#                def get_local_runtime_version_fixed() -> int:
#                    return cupy.cuda.runtime.runtimeGetVersion()
#                # Overwrite the high-level function in the cupy.cuda module forcing it to use our fixed version
#                setattr(cupy.cuda, 'get_local_runtime_version', get_local_runtime_version_fixed)
#                print("LOG: Successfully patched cupy.cuda.get_local_runtime_version")
#        except ImportError:
#            print("WARNING: Could not import cupy to patch runtime version.")
#        except Exception as e:
#            print(f"WARNING: Failed to patch cupy: {e}")
#
#    patch_cupy_runtime()
#
#else:
#    print("LOG: CuPy not imported, skipping runtime patch.")

    
"""
Compatibility fix for CuPy with gpubackendtools.
Import this before importing bbhx or any module that uses gpubackendtools.

import cupy

# Fix for CuPy 12.x - missing get_local_runtime_version
if not hasattr(cupy.cuda, 'get_local_runtime_version'):
    def get_local_runtime_version():
        return cupy.cuda.runtime.runtimeGetVersion()
    cupy.cuda.get_local_runtime_version = get_local_runtime_version
    print("Applied CuPy 12.x compatibility patch: cupy.cuda.get_local_runtime_version")

# Fix for CuPy 13.x - missing _getLocalRuntimeVersion
if not hasattr(cupy.cuda.runtime, '_getLocalRuntimeVersion'):
    def _getLocalRuntimeVersion():
        return cupy.cuda.runtime.runtimeGetVersion()
    cupy.cuda.runtime._getLocalRuntimeVersion = _getLocalRuntimeVersion
    print("Applied CuPy 13.x compatibility patch: cupy.cuda.runtime._getLocalRuntimeVersion")
"""


import sys

def apply_patch():
    print("LOG: [fix_cupy] Attempting to patch CuPy...")
    try:
        # Force import of cupy modules
        import cupy
        import cupy.cuda
        import cupy.cuda.runtime

        # Patch 1: cupy.cuda.get_local_runtime_version (Missing in CuPy 12+)
        if not hasattr(cupy.cuda, 'get_local_runtime_version'):
            print("LOG: [fix_cupy] Patching cupy.cuda.get_local_runtime_version")
            def get_local_runtime_version():
                return cupy.cuda.runtime.runtimeGetVersion()
            cupy.cuda.get_local_runtime_version = get_local_runtime_version
        
        # Patch 2: cupy.cuda.runtime._getLocalRuntimeVersion (Missing in CuPy 12+, used by gpubackendtools)
        if not hasattr(cupy.cuda.runtime, '_getLocalRuntimeVersion'):
            print("LOG: [fix_cupy] Patching cupy.cuda.runtime._getLocalRuntimeVersion")
            def _getLocalRuntimeVersion():
                return cupy.cuda.runtime.runtimeGetVersion()
            cupy.cuda.runtime._getLocalRuntimeVersion = _getLocalRuntimeVersion
            
        print("LOG: [fix_cupy] Patch complete.")

    except ImportError:
        print("LOG: [fix_cupy] CuPy not installed. Skipping patch.")
    except Exception as e:
        print(f"WARNING: [fix_cupy] Failed to patch cupy: {e}")

# Run the patch function immediately when this module is imported
apply_patch()