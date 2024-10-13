import numpy as np
import cv2
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm 

def visualize_dct(dct_coefficient):
    log_coefficient = np.log(np.abs(dct_coefficient) + 1)
    plt.imshow(log_coefficient, cmap='gray')
    plt.colorbar()
    plt.title('DCT Coefficients (Log Domain)')
    plt.show()

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def dct_2d(image):
    M, N = image.shape
    dct_coefficient = np.zeros(image.shape)
    x, y = np.arange(M), np.arange(N)
    
    for u in tqdm(range(M)):
        for v in range(N):
            cos_u = np.cos(((2 * x + 1) * u * np.pi) / (2 * M))
            cos_v = np.cos(((2 * y + 1) * v * np.pi) / (2 * N))
            
            C_u = 1 / np.sqrt(2) if u == 0 else 1
            C_v = 1 / np.sqrt(2) if v == 0 else 1
            
            dct_coefficient[u, v] = C_u * C_v * np.sum(image * np.outer(cos_u, cos_v))
    
    dct_coefficient = dct_coefficient * (2 / np.sqrt(M * N))
    return dct_coefficient

def idct_2d(dct_coefficient):
    M, N = dct_coefficient.shape
    reconstructed_img = np.zeros(dct_coefficient.shape)
    u, v = np.arange(M), np.arange(N)
    
    C_u = np.where(u == 0, 1 / np.sqrt(2), 1)
    C_v = np.where(v == 0, 1 / np.sqrt(2), 1)
    
    for x in tqdm(range(M)):
        for y in range(N):
            cos_x = np.cos(((2 * x + 1) * u * np.pi) / (2 * M))
            cos_y = np.cos(((2 * y + 1) * v * np.pi) / (2 * N))
            reconstructed_img[x, y] = np.sum(np.outer(C_u, C_v) * dct_coefficient * np.outer(cos_x, cos_y))
    
    reconstructed_img = reconstructed_img * (2 / np.sqrt(M * N))
    return reconstructed_img

def dct_1d(vector):
    N = len(vector)
    coff = np.zeros(N)
    x = np.arange(N)
    
    for u in range(N):
        cos_term = np.cos(np.pi * (2 * x + 1) * u / (2 * N))
        C_u = 1 / np.sqrt(2) if u == 0 else 1
        coff[u] = C_u * np.sum(vector * cos_term)
    coff = coff * np.sqrt(2 / N)
    return coff

def two_1d_dct(image):
    rows, cols = image.shape
    dct_coefficient = np.zeros(image.shape)
    
    for i in tqdm(range(rows)):
        dct_coefficient[i, :] = dct_1d(image[i, :])
    
    for j in tqdm(range(cols)):
        dct_coefficient[:, j] = dct_1d(dct_coefficient[:, j])
    
    return dct_coefficient

if __name__ == '__main__':
    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)

    # 2D-DCT
    start_time = time()
    dct_2d_coefficient = dct_2d(img)
    end_time = time()
    dct_2d_time = end_time - start_time
    
    # reconstruct
    reconstructed_img = idct_2d(dct_2d_coefficient)
    
    # Calculate PSNR
    psnr = calculate_psnr(img, reconstructed_img)
    
    # Two 1D-DCT
    start_time = time()
    dct_1d_coefficient = two_1d_dct(img)
    print(dct_1d_coefficient)
    end_time = time()
    dct_1d_time = end_time - start_time

    # Compare runtimes
    print(f"Runtime for 2D-DCT: {dct_2d_time:.6f} seconds")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Runtime for Two 1D-DCT: {dct_1d_time:.6f} seconds")

    cv2.imwrite('reconstructed_image.png', reconstructed_img)
    
    # Visualize DCT coefficients
    visualize_dct(dct_2d_coefficient)
