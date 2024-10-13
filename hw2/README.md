## DCT (Discrete Cosine Transform) Implementation
This project implements both 2D-DCT and Two 1D-DCT algorithms for image processing. It includes functions for performing DCT, inverse DCT, visualizing DCT coefficients, and calculating PSNR (Peak Signal-to-Noise Ratio).

### Installation
- Python 3.11
- NumPy
- OpenCV (cv2)
- Matplotlib
- tqdm

You can install the required packages using pip:
```
pip install numpy opencv-python matplotlib tqdm
```

### Usage
1. Place the image `lena.png` in the same directory as the script

2. Run the script:
```
python3 main.py
```

3. The script will output:
- Runtime for 2D-DCT
- PSNR of the reconstructed image
- Runtime for Two 1D-DCT
- A visualization of the DCT coefficients
- A reconstructed image saved as 'reconstructed_image.png'



