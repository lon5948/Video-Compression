import numpy as np
import cv2

QUANTIZATION_TABLE_1 = np.array([
    [10, 7,  6,  10, 14, 24, 31, 37],
    [7,  7,  8,  11, 16, 35, 36, 33],
    [8,  8,  10, 14, 24, 34, 41, 34],
    [8,  10, 13, 17, 31, 52, 48, 37],
    [11, 13, 22, 34, 41, 65, 62, 46],
    [14, 21, 33, 38, 49, 62, 68, 55],
    [29, 38, 47, 52, 62, 73, 72, 61],
    [43, 55, 57, 59, 67, 60, 62, 59]
])

QUANTIZATION_TABLE_2 = np.array([
    [10, 11, 14, 28, 59, 59, 59, 59],
    [11, 13, 16, 40, 59, 59, 59, 59],
    [14, 16, 34, 59, 59, 59, 59, 59],
    [28, 40, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59]
])

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def dct_2d(block):
    M, N = block.shape
    dct_coefficient = np.zeros(block.shape)
    x, y = np.arange(M), np.arange(N)
    
    for u in range(M):
        for v in range(N):
            cos_u = np.cos(((2 * x + 1) * u * np.pi) / (2 * M))
            cos_v = np.cos(((2 * y + 1) * v * np.pi) / (2 * N))
            
            C_u = 1 / np.sqrt(2) if u == 0 else 1
            C_v = 1 / np.sqrt(2) if v == 0 else 1
            
            dct_coefficient[u, v] = C_u * C_v * np.sum(block * np.outer(cos_u, cos_v))
    
    dct_coefficient = dct_coefficient * (2 / np.sqrt(M * N))
    return dct_coefficient

def idct_2d(dct_coefficient):
    M, N = dct_coefficient.shape
    reconstructed_img = np.zeros(dct_coefficient.shape)
    u, v = np.arange(M), np.arange(N)
    
    C_u = np.where(u == 0, 1 / np.sqrt(2), 1)
    C_v = np.where(v == 0, 1 / np.sqrt(2), 1)
    
    for x in range(M):
        for y in range(N):
            cos_x = np.cos(((2 * x + 1) * u * np.pi) / (2 * M))
            cos_y = np.cos(((2 * y + 1) * v * np.pi) / (2 * N))
            reconstructed_img[x, y] = np.sum(np.outer(C_u, C_v) * dct_coefficient * np.outer(cos_x, cos_y))
    
    reconstructed_img = reconstructed_img * (2 / np.sqrt(M * N))
    return reconstructed_img

def quantize(dct_block, quantization_table):
    quantized_block = np.round(dct_block / quantization_table)
    return quantized_block

def block_idct_dequantize(quantized_block, quantization_table):
    dequantized_block = quantized_block * quantization_table
    return dequantized_block

def zigzag(input):
    h = 0
    v = 0
    
    vmin = 0
    hmin = 0
    
    vmax = input.shape[0]
    hmax = input.shape[1]

    output = np.zeros(( vmax * hmax))

    i = 0
    
    while ((v < vmax) and (h < hmax)):
        if ((h + v) % 2) == 0:
            if (v == vmin):
                output[i] = input[v, h]
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        
                i = i + 1
            elif ((h == hmax -1 ) and (v < vmax)):
                output[i] = input[v, h] 
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax -1 )):
                output[i] = input[v, h] 
                v = v - 1
                h = h + 1
                i = i + 1
        else:
            if ((v == vmax -1) and (h <= hmax -1)):
                output[i] = input[v, h] 
                h = h + 1
                i = i + 1
            elif (h == hmin):
                output[i] = input[v, h] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif ((v < vmax -1) and (h > hmin)):
                output[i] = input[v, h] 
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == vmax-1) and (h == hmax-1)):
            output[i] = input[v, h] 
            break
    return output

def RLE_encoding(img, bits=8):
    encoded = []
    count = 0
    prev = None
    
    # Zigzag scan
    zigzag_image = zigzag(img)
    
    # Run-length encoding
    for pixel in zigzag_image:
        if prev is None:
            prev = pixel
            count += 1
        else:
            if prev != pixel:
                encoded.append((count, prev))
                prev = pixel
                count = 1
            else:
                if count < (2**bits) - 1:
                    count += 1
                else:
                    encoded.append((count, prev))
                    prev = pixel
                    count = 1
    encoded.append((count, prev))
    return np.array(encoded)

def inverse_zigzag(input, shape):
	vmax, hmax = shape
	h = 0
	v = 0
	vmin = 0
	hmin = 0

	output = np.zeros((vmax, hmax))

	i = 0
	while ((v < vmax) and (h < hmax)): 	
		if ((h + v) % 2) == 0:
			if (v == vmin):
				output[v, h] = input[i]
				if (h == hmax):
					v = v + 1
				else:
					h = h + 1                        
				i = i + 1
			elif ((h == hmax -1 ) and (v < vmax)):
				output[v, h] = input[i] 
				v = v + 1
				i = i + 1
			elif ((v > vmin) and (h < hmax -1 )):
				output[v, h] = input[i] 
				v = v - 1
				h = h + 1
				i = i + 1
		else:
			if ((v == vmax -1) and (h <= hmax -1)):
				output[v, h] = input[i] 
				h = h + 1
				i = i + 1
			elif (h == hmin):
				output[v, h] = input[i] 
				if (v == vmax -1):
					h = h + 1
				else:
					v = v + 1
				i = i + 1	
			elif((v < vmax -1) and (h > hmin)):
				output[v, h] = input[i] 
				v = v + 1
				h = h - 1
				i = i + 1
		if ((v == vmax-1) and (h == hmax-1)):
			output[v, h] = input[i] 
			break
	return output

def RLE_decode(encoded, shape):
    decoded = []
    for rl in encoded:
        r, p = rl[0], rl[1]
        decoded.extend([p] * int(r))
    dimg = np.array(inverse_zigzag(decoded, shape)).reshape(shape)
    return dimg

def process_image(img, quantization_table):
    rows, cols = img.shape
    encoded_blocks = []
    encoded_sizes = []
    
    # Encoding
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            block = img[i:i+8, j:j+8]
            dct_block = dct_2d(np.float32(block))
            quantized_block = quantize(dct_block, quantization_table)
            encoded_block = RLE_encoding(quantized_block)
            encoded_blocks.append(encoded_block)
            encoded_sizes.append(len(encoded_block))
    
    # Decoding
    decoded_blocks = []
    for encoded_block in encoded_blocks:
        decoded_block = RLE_decode(encoded_block, (8,8))
        dequantized_block = block_idct_dequantize(decoded_block, quantization_table)
        idct_block = idct_2d(dequantized_block)
        decoded_blocks.append(idct_block)
    
    # Reconstruct image
    decoded_image = np.zeros_like(img)
    index = 0
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            decoded_image[i:i+8, j:j+8] = decoded_blocks[index]
            index += 1
    
    total_encoded_elements = sum(encoded_sizes)
    original_size = img.size
    compression_ratio = original_size / total_encoded_elements
    psnr = calculate_psnr(img, decoded_image)
    
    print(f"Original size: {original_size} elements")
    print(f"Compressed size: {total_encoded_elements} elements")
    print(f"Compression ratio: {compression_ratio:.2f}")
    print(f"PSNR: {psnr:.2f} dB")
    
    return np.clip(decoded_image, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
    
    print("\nProcessing with Quantization Table 1...")
    output_image1 = process_image(img, QUANTIZATION_TABLE_1)
    
    print("\nProcessing with Quantization Table 2...")
    output_image2 = process_image(img, QUANTIZATION_TABLE_2)
    
    # Save result
    cv2.imwrite("origin.png", img)
    cv2.imwrite("reconstruct_table1.png", output_image1)
    cv2.imwrite("reconstruct_table2.png", output_image2)
