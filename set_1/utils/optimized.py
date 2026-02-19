import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def wave_1d_jit(x: np.ndarray, x_prev: np.ndarray, x_next: np.ndarray, C2: float) -> None:
    """Optimized 1D wave equation solver using finite difference method with numba JIT compilation and parallelization."""
    for i in prange(1, x.shape[0] - 1):
        x_next[i] = C2 * (x[i-1] - 2*x[i] + x[i+1]) - x_prev[i] + 2*x[i]

@njit(parallel=True, fastmath=True)
def wave_2d_jit(x: np.ndarray, x_next: np.ndarray, d: float) -> None:
    """Optimized 2D wave equation solver using finite difference method with numba JIT compilation and parallelization."""
    for i in prange(1, x.shape[0] - 1):
        # boundary
        x_next[i, 0] = x_next[i, -1] = x[i, 0] + d * (x[i-1, 0] + x[i+1, 0] + x[i, 1] + x[i, -2] - 4*x[i, 0])

        # interior
        for j in range(1, x.shape[1] - 1):
            x_next[i, j] = x[i, j] + d * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1] - 4*x[i, j])

@njit(parallel=True, fastmath=True)
def jacobi_jit(x: np.ndarray, x_next: np.ndarray, obj_mask: np.ndarray) -> float:
    """Optimized Jacobi iteration using numba JIT compilation and parallelization."""
    max_diff = 0.0

    for i in prange(1, x.shape[0] - 1):
        # boundary
        x_next[i, 0] = x_next[i, -1] = 0.25 * (x[i-1, 0] + x[i+1, 0] + x[i, 1] + x[i, -2])
        max_diff = max(max_diff, abs(x[i, 0] - x_next[i, 0]))

        # interior
        for j in range(1, x.shape[1] - 1):
            if obj_mask[i, j]:
                # dont forget to index obj_mask -1, since it has shape (48, 50)
                x_next[i, j] = obj_mask[i - 1, j] * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])
                max_diff = max(max_diff, abs(x[i, j] - x_next[i, j]))

    return max_diff

@njit(parallel=True, fastmath=True)
def gauss_seidel_jit(x: np.ndarray, obj_mask: np.ndarray) -> float:
    """Optimized Gauss-Seidel iteration using numba JIT compilation and parallelization with red-black ordering."""
    rows, cols = x.shape
    max_diff = 0.0
    
    # red points
    for i in prange(1, rows - 1):
        # boundary
        if i % 2 == 0:
            next = 0.25 * (x[i-1, 0] + x[i+1, 0] + x[i, 1] + x[i, -2])
            diff = abs(next - x[i, 0])
            x[i, 0] = next
            max_diff = max(max_diff, diff)

        # interior
        start = 2 if (i % 2 == 0) else 1
        for j in range(start, cols - 1, 2):
            if obj_mask[i, j]:
                # dont forget to index obj_mask -1, since it has shape (48, 50)
                next = obj_mask[i - 1, j] * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])
                diff = abs(next - x[i, j])
                x[i, j] = next
                max_diff = max(max_diff, diff)
        
        # enforce periodicity condition for the rightmost column
        x[i, -1] = x[i, 0]

    # black points
    for i in prange(1, rows - 1):
        # boundary
        if i % 2 != 0:
            next = 0.25 * (x[i-1, 0] + x[i+1, 0] + x[i, 1] + x[i, -2])
            diff = abs(next - x[i, 0])
            x[i, 0] = next
            max_diff = max(max_diff, diff)

        # interior
        start = 1 if (i % 2 == 0) else 2
        for j in range(start, cols - 1, 2):
            if obj_mask[i, j]:
                # dont forget to index obj_mask -1, since it has shape (48, 50)
                next = obj_mask[i-1, j] * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])
                diff = abs(next - x[i, j])
                x[i, j] = next
                max_diff = max(max_diff, diff)

        # enforce periodicity condition for the rightmost column
        x[i, -1] = x[i, 0]
    
            
    return max_diff

@njit(parallel=True, fastmath=True)
def sor_jit(x: np.ndarray, omega: float, obj_mask: np.ndarray) -> float:
    """Optimized SOR iteration using numba JIT compilation and parallelization with red-black ordering."""
    rows, cols = x.shape
    max_diff = 0.0
    
    # red points
    for i in prange(1, rows - 1):
        # boundary
        if i % 2 == 0:
            neighbor_sum = 0.25 * (x[i-1, 0] + x[i+1, 0] + x[i, 1] + x[i, -2])
            next = (1 - omega) * x[i, 0] + omega * neighbor_sum
            diff = abs(next - x[i, 0])
            x[i, 0] = next
            max_diff = max(max_diff, diff)
        
        # interior
        start = 2 if (i % 2 == 0) else 1
        for j in range(start, cols - 1, 2):
            if obj_mask[i, j]:
                # dont forget to index obj_mask -1, since it has shape (48, 50)
                neighbor_sum = obj_mask[i - 1, j] * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])
                next = (1 - omega) * x[i, j] + omega * neighbor_sum
                diff = abs(next - x[i, j])
                x[i, j] = next
                max_diff = max(max_diff, diff)

        # enforce periodicity condition for the rightmost column
        x[i, -1] = x[i, 0]

    # black points
    for i in prange(1, rows - 1):
        # boundary
        if i % 2 != 0:
            neighbor_sum = 0.25 * (x[i-1, 0] + x[i+1, 0] + x[i, 1] + x[i, -2])
            next = (1 - omega) * x[i, 0] + omega * neighbor_sum
            diff = abs(next - x[i, 0])
            x[i, 0] = next
            max_diff = max(max_diff, diff)
        
        # interior
        start = 1 if (i % 2 == 0) else 2
        for j in range(start, cols - 1, 2):
            if obj_mask[i, j]:
                # dont forget to index obj_mask -1, since it has shape (48, 50)
                neighbor_sum = obj_mask[i - 1, j] * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])
                next = (1 - omega) * x[i, j] + omega * neighbor_sum
                diff = abs(next - x[i, j])
                x[i, j] = next
                max_diff = max(max_diff, diff)

        # enforce periodicity condition for the rightmost column
        x[i, -1] = x[i, 0]

    return max_diff