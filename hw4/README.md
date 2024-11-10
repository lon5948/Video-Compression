## DCT-Based Image Compression with RLE
This implementation provides a complete pipeline for image compression using Discrete Cosine Transform (DCT) combined with Run Length Encoding (RLE). The system processes grayscale images through block-based DCT transformation, quantization, and RLE compression.

### Features
- Block-based DCT transform (8x8 pixels)
- Two quantization tables for different compression ratios
- Zigzag scanning of DCT coefficients
- Run Length Encoding compression
- PSNR calculation for quality assessment


### Requirements
- Python 3.x
- NumPy
- OpenCV (cv2)

### Installation
```
pip install numpy opencv-python
```

### Usage
1. Place your input image in the same directory as the script

2. Run the script:
```
python3 main.py
```

3. The script will:
    1. Read the input image ("lena.png")
    2. Process it with two different quantization tables
    3. Save three images:
        - origin.png: Original input image
        - reconstruct_table1.png: Reconstructed image using Quantization Table 1
        - reconstruct_table2.png: Reconstructed image using Quantization Table 2

## Results
The implementation typically achieves:
- With Table 1: ~5x compression with PSNR > 37 dB
- With Table 2: ~9x compression with PSNR > 35 dB