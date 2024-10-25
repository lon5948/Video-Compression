import cv2
import numpy as np
from typing import Tuple
import time
from tabulate import tabulate

def mean_absolute_deviation(block1: np.ndarray, block2: np.ndarray) -> float:
    return np.mean(np.abs(block1 - block2))

def full_search(cur_block: np.ndarray, ref_frame: np.ndarray, 
                block_pos: Tuple[int, int], search_range: int) -> Tuple[int, int]:
    block_size = cur_block.shape[0]
    height, width = ref_frame.shape
    min_mse = float('inf')
    motion_vector = (0, 0)
    
    y, x = block_pos
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            ref_y, ref_x = y + dy, x + dx
            
            # Check boundaries
            if (ref_y < 0 or ref_y + block_size > height or ref_x < 0 or ref_x + block_size > width):
                continue
                
            ref_block = ref_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
            mse = mean_absolute_deviation(cur_block, ref_block)
            
            if mse < min_mse:
                min_mse = mse
                motion_vector = (dy, dx)
    
    return motion_vector

def three_step_search(cur_block: np.ndarray, ref_frame: np.ndarray, 
                     block_pos: Tuple[int, int], search_range: int) -> Tuple[int, int]:
    block_size = cur_block.shape[0]
    height, width = ref_frame.shape
    step_size = search_range // 2
    min_mse = float('inf')
    motion_vector = (0, 0)
    
    center_y, center_x = block_pos
    
    while step_size >= 1:
        for dy in [-step_size, 0, step_size]:
            for dx in [-step_size, 0, step_size]:
                ref_y, ref_x = center_y + dy, center_x + dx
                
                if (ref_y < 0 or ref_y + block_size > height or 
                    ref_x < 0 or ref_x + block_size > width):
                    continue
                    
                ref_block = ref_frame[ref_y:ref_y + block_size, 
                                    ref_x:ref_x + block_size]
                mse = mean_absolute_deviation(cur_block, ref_block)
                
                if mse < min_mse:
                    min_mse = mse
                    motion_vector = (dy, dx)
        
        center_y += motion_vector[0]
        center_x += motion_vector[1]
        step_size //= 2
        
    return (center_y - block_pos[0], center_x - block_pos[1])

def motion_estimation(cur_frame: np.ndarray, ref_frame: np.ndarray, 
                     block_size: int = 8, search_range: int = 8,
                     method: str = 'full') -> Tuple[np.ndarray, np.ndarray]:
    height, width = cur_frame.shape
    mv_y = np.zeros((height // block_size, width // block_size), dtype=int)
    mv_x = np.zeros((height // block_size, width // block_size), dtype=int)
    
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            cur_block = cur_frame[i:i + block_size, j:j + block_size]
            block_pos = (i, j)
            
            if method == 'full':
                dy, dx = full_search(cur_block, ref_frame, block_pos, search_range)
            else:
                dy, dx = three_step_search(cur_block, ref_frame, block_pos, search_range)
                
            mv_y[i // block_size, j // block_size] = dy
            mv_x[i // block_size, j // block_size] = dx
            
    return mv_y, mv_x

def motion_compensation(ref_frame: np.ndarray, mv_y: np.ndarray, 
                       mv_x: np.ndarray, block_size: int = 8) -> np.ndarray:
    height, width = ref_frame.shape
    reconstructed = np.zeros_like(ref_frame)
    
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            dy = mv_y[i // block_size, j // block_size]
            dx = mv_x[i // block_size, j // block_size]
            
            ref_y = max(0, min(i + dy, height - block_size))
            ref_x = max(0, min(j + dx, width - block_size))
            
            reconstructed[i:i + block_size, j:j + block_size] = ref_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
    
    return reconstructed

def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

if __name__ == "__main__":
    # Load images
    cur_frame = cv2.imread('./inputs/one_gray.png', cv2.IMREAD_GRAYSCALE)
    ref_frame = cv2.imread('./inputs/two_gray.png', cv2.IMREAD_GRAYSCALE)
    search_ranges = [8, 16, 32]
    
    table_data = []
    
    for search_range in search_ranges:
        # Full search
        full_start_time = time.time()
        mv_y, mv_x = motion_estimation(cur_frame, ref_frame, 8, search_range, 'full')
        reconstructed_full = motion_compensation(ref_frame, mv_y, mv_x, 8)
        residual_full = cur_frame - reconstructed_full
        full_time = time.time() - full_start_time
        full_psnr = calculate_psnr(cur_frame, reconstructed_full)
        
        cv2.imwrite(f'./outputs/full_search_range_{search_range}.png', reconstructed_full)
        cv2.imwrite(f'./outputs/full_search_residual_range_{search_range}.png', residual_full)
        
        # Three-step search
        three_start_time = time.time()
        mv_y, mv_x = motion_estimation(cur_frame, ref_frame, 8, search_range, 'three_step')
        reconstructed_three = motion_compensation(ref_frame, mv_y, mv_x, 8)
        residual_three = cur_frame - reconstructed_three
        three_time = time.time() - three_start_time
        three_psnr = calculate_psnr(cur_frame, reconstructed_three)
        
        cv2.imwrite(f'./outputs/three_step_range_{search_range}.png', reconstructed_three)
        cv2.imwrite(f'./outputs/three_step_residual_range_{search_range}.png', residual_three)
        
        # Add to results table
        table_data.append([
            f"±{search_range}",
            f"{full_psnr:.2f}",
            f"{full_time:.3f}",
            f"{three_psnr:.2f}",
            f"{three_time:.3f}"
        ])
    
    headers = ["Search Range", 
              "Full Search PSNR (dB)", "Full Search Time (s)",
              "Three-Step PSNR (dB)", "Three-Step Time (s)"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    
    print("\nMotion Estimation Comparison Results:")
    print(table)