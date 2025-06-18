import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Config
CSV_DIR = './'  # Change to your directory
GRID_SIZE = 12
NUM_FILES = 1

# Initialize grid to count occurrences
heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Process each CSV file
for i in range(0,NUM_FILES):
    filename = os.path.join(CSV_DIR, f'attention_0_{i}.csv')
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            for item in row:
                # Convert string like "(9.0, 11.0)" to tuple of ints
                x, y = eval(item)  # Safe here since input is controlled
                x, y = int(x), int(y)
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    heatmap[y, x] += 1  # Note: [row, col] = [y, x] for plotting

# Plot heatmap
plt.figure(figsize=(6, 6))
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.title('Tile Attention Frequency (12x12 Grid)')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.colorbar(label='Occurrences')
plt.xticks(range(GRID_SIZE))
plt.yticks(range(GRID_SIZE))
plt.gca().invert_yaxis()  # Optional: (0,0) in top-left
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=300)
plt.show()
