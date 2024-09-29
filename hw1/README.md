## Lena Color Space Conversion
This project converts the "lena.png" image into `RGB`, `YUV`, and `YCbCr` color spaces, outputting 8 grayscale images representing `R, G, B, Y, U, V, Cb,` and `Cr` channels.

### Introduction
The script processes the classic "lena.png" image, performing the following tasks:
- Extracts `R`, `G`, and `B` channels
- Converts the image to `YUV` color space
- Converts the image to `YCbCr` color space
- Outputs 8 separate grayscale images for each channel `(R, G, B, Y, U, V, Cb, Cr)`

### Installation
- Ensure you have Python 3.11 installed on your system
- Install the required libraries:
```
pip install Pillow numpy
```

### Usage
1. Place the `lena.png` image in the same directory as the script.
2. Run the script:
```
python3 main.py
```
3. After execution, you will find 8 new grayscale images in the same directory, each representing a different channel `(R, G, B, Y, U, V, Cb, Cr)`.

### Output
The script generates the following output files:
- `r_channel.png`
- `g_channel.png`
- `b_channel.png`
- `y_channel.png`
- `u_channel.png`
- `v_channel.png`
- `cb_channel.png`
- `cr_channel.png`

Each image represents the corresponding channel in grayscale.
