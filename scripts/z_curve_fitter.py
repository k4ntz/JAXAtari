import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Config ---
CSV_PATH = "saved_states/cell_17_e7db3366-777b-4e61-96e9-5fbf88a77ecb.csv"
Z_RESET_VALUE = 15  # z=15 means serve or hit
POLY_DEGREE = 2     # adjust as needed

# --- Load CSV ---
def load_z_data(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        return [int(row[0]) for row in reader if row]

z_values = load_z_data(CSV_PATH)

# --- Extract arcs: split at resets to z=15 ---
def extract_aligned_arcs(z_data, reset_val):
    left_edge = True
    in_arc = False
    arcs = []
    current_arc = []
    for z in z_data:
        if z <= reset_val:
            left_edge = False
            if in_arc:
                arcs.append(current_arc)
                in_arc = False
                continue
        else:
            if not left_edge:
                in_arc = True
                current_arc.append(z)
    return arcs


def extract_arcs(z_data, reset_val):
    z_data = [z for z in z_data if z >= reset_val]

    descend = False
    #remove downward arc at beginning of data set
    for z in z_data:
        descend = z_data[0] > z_data[1]
        if z < reset_val or descend:
            z_data.pop(0)
            continue
        break
    arcs = []
    current_arc = []
    rebound = True
    for i in range(len(z_data) - 1):
        current_arc.append(z_data[i])
        if z_data[i] > 30:
            rebound = False
        # bottom of arc
        if descend and z_data[i+1] > z_data[i]:
            descend = False
            if not rebound:
                arcs.append(current_arc)
                rebound = True
            current_arc = []
        descend |= z_data[i] > z_data[i+1]
    return arcs

#arcs = extract_aligned_arcs(z_values, Z_RESET_VALUE)
#slice of first and last element
arcs = extract_arcs(z_values, Z_RESET_VALUE)

# Prepare CSV output
os.makedirs("edited_z_traces", exist_ok=True)
csv_path = os.path.join("saved_states", f"arcs_3.csv")

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for arc in arcs:
        for value in arc:
            writer.writerow([value])  # Each value in its own row
        writer.writerow([])

# --- Align all arcs to start at x=0, stack them together ---
x_all = []
y_all = []

for arc in arcs:
    for i, z in enumerate(arc):
        x_all.append(i)
        y_all.append(z)

print(y_all)

x_all = np.array(x_all)
y_all = np.array(y_all)

# --- Fit a single polynomial to all the aligned arc data ---
coeffs = np.polyfit(x_all, y_all, POLY_DEGREE)
fit_func = np.poly1d(coeffs)

# --- Plot ---
plt.figure(figsize=(10, 10))
plt.scatter(x_all, y_all, color="gray", alpha=0.6, label="All arc samples")
x_fit = np.linspace(0, max(x_all), 200)
plt.plot(x_fit, fit_func(x_fit), color="red", label=f"Fitted degree-{POLY_DEGREE} curve")

plt.title("Combined Z Trajectory and Fitted Curve")
plt.xlabel("Frame (from arc start)")
plt.ylabel("Z Position")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Optional: print coefficients
print("Fitted polynomial coefficients:", coeffs)
