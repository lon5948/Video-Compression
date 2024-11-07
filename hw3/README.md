## Motion Estimation and Motion Compensation
This project contains an implementation of two motion estimation algorithms: Full Search Block Matching (FSBM) and Three-Step Search (TSS) for video compression applications.

### Parameters

- Block Size: 8×8
- Search Ranges: ±8, ±16, and ±32
- Integer-pixel precision
- Matching Criterion: Mean Absolute Difference (MAD)

### Requirements
```
opencv-python
numpy
typing
tabulate
```

### Usage
1. Place your input frames in the `inputs/` directory

2. Run the script:
```
python3 main.py
```

3. The script will:
    - Process the frames using both algorithms
    - Generate reconstructed frames and residual images to the `outputs/` directory
    - Print comparison results table

