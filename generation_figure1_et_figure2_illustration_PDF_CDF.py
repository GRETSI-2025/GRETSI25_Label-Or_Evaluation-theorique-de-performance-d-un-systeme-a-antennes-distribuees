"""
Copyright (c) [2025] [Orange SA]
Licensed under the MIT License. See the LICENSE file in the project root for full license information.
"""

# library importation
#############################################################################################
import numpy as np
import matplotlib.pyplot as plt

# Assuming the functions are defined in a module called useful_functions:
import useful_functions
#############################################################################################
#               FIRST PLOT
#############################################################################################

# -----------------------------
# Simulation Parameters
# -----------------------------
snr_ebno_dB = 5  # Signal-to-noise ratio in dB
ebno_base10 = 10 ** (snr_ebno_dB / 10)  # Convert dB to linear scale

# Base station power and link blockage mask:
array_BS_power = np.array([1])
# Mask indicating if the link is blocked (0) or unblocked (1)
array_mask_blocked_link = np.array([0])

# Channel variance values for unblocked (hNB) and blocked (hB) links:
array_variance_hNB = np.array([1])
array_variance_hB = np.array([0.9275625038076353])
# Compute the correlation coefficient between hB and hNB based on their variances:
array_corr_hB_hNB = np.sqrt(array_variance_hB) / np.sqrt(array_variance_hNB)

# Modulation parameters:
M = 16  # 16-QAM modulation
Power_constellation_16QAM = (
    10  # Constant for the 16-QAM constellation (not used directly here)
)

# -----------------------------
# Define 16-QAM Symbols
# -----------------------------
# List of normalized 16-QAM symbols (each symbol is scaled by sqrt(1/10))
symbols = [
    (3 + 3j) * np.sqrt(1 / 10),
    (3 + 1j) * np.sqrt(1 / 10),
    (-3 + 3j) * np.sqrt(1 / 10),
    (-3 + 1j) * np.sqrt(1 / 10),
    (3 - 3j) * np.sqrt(1 / 10),
    (3 - 1j) * np.sqrt(1 / 10),
    (-3 - 3j) * np.sqrt(1 / 10),
    (-3 - 1j) * np.sqrt(1 / 10),
    (1 + 3j) * np.sqrt(1 / 10),
    (1 + 1j) * np.sqrt(1 / 10),
    (-1 + 3j) * np.sqrt(1 / 10),
    (-1 + 1j) * np.sqrt(1 / 10),
    (1 - 3j) * np.sqrt(1 / 10),
    (1 - 1j) * np.sqrt(1 / 10),
    (-1 - 3j) * np.sqrt(1 / 10),
    (-1 - 1j) * np.sqrt(1 / 10),
]

# Convert list of symbols into a NumPy array.
symboles_array = np.array(symbols)
# Create an unnormalized version by multiplying by sqrt(10)
symboles_array_non_norm = symboles_array * np.sqrt(10)

# -----------------------------
# Preallocate arrays and define grid for joint PDF evaluation
# -----------------------------
corr_array = []  # Will store correlation ratio for each symbol
sigma_r_i = []  # Will store the "sigma_r" value for each symbol

# Define the span for the real and imaginary axes for the contour plots
a = 1.5  # Range parameter for the real axis
span_real = np.linspace(-a, a, 1000)
# For the imaginary part, choose a range (here from -0.7 to a)
span_complex = np.linspace(-0.7, a, 1000)

# -----------------------------
# Plot Setup
# -----------------------------
plt.figure(figsize=(6.6, 5), dpi=100)

# Loop over each symbol in the 16-QAM constellation
for i, symbol in enumerate(symboles_array):
    # Compute the correlation ratio for the current symbol using the unnormalized symbol.
    corr_value = useful_functions.corr_num_denom(
        symboles_array_non_norm[i],
        array_BS_power,
        array_variance_hB,
        array_variance_hNB,
        M,
        ebno_base10,
    )
    corr_array.append(corr_value)

    # Compute power contributions for blocked and unblocked links:
    mu_link_unblocked = array_mask_blocked_link * array_BS_power
    mu_link_blocked = ((array_mask_blocked_link == 0).astype(int)) * array_BS_power

    # Denominator components:
    denom1 = (
        mu_link_blocked * array_variance_hB
    )  # Blocked links variance weighted by power
    denom2 = (
        mu_link_unblocked * array_variance_hNB
    )  # Unblocked links variance weighted by power
    denom3 = (
        array_BS_power * array_variance_hNB
    )  # Common denominator part for all symbols

    # sigma_e is common to all symbols (sqrt of the total weighted variance for unblocked links)
    sigma_e = np.sqrt(np.sum(denom3))

    # For the current symbol, compute its noise-related term:
    a_d = np.real(symboles_array_non_norm[i])
    b_d = np.imag(symboles_array_non_norm[i])
    N0_over_mod_x_squared_d = (
        (ebno_base10**-1) * (2 * (M - 1)) / (3 * ((a_d) ** 2 + (b_d) ** 2) * np.log2(M))
    )

    # Compute sigma_r_i for the current symbol.
    # This value scales with the symbol's energy and the overall noise and channel variance.
    sigma_r_current = np.sqrt(
        np.abs(symboles_array[i]) ** 2
        * (np.sum(denom1) + np.sum(denom2) + N0_over_mod_x_squared_d)
    )
    sigma_r_i.append(sigma_r_current)

    # Set mean values for the joint PDF function (assuming zero means)
    m_r = 0
    m_e = 0

    # Compute the joint PDF of the ratio using the current parameters.
    # The joint PDF is evaluated over the grid defined by span_real and span_complex.
    z_real, z_imag, f_z = useful_functions.joint_pdf_ratio_complex_gaussian(
        m_r, m_e, sigma_r_i[i], sigma_e, corr_array[i], span_real, span_complex
    )

    # For a few selected symbols (here indices 0, 13, and 3), plot the contour of the joint PDF.
    if i in [0, 13, 3]:
        plt.contour(z_real, z_imag, f_z, cmap="turbo")

# -----------------------------
# Add Annotations with Arrows
# -----------------------------
# Annotation for a symbol with correlation ρ = -0.88 + j0.29
plt.annotate(
    r"$\mathbf{\rho=-0.88+j0.29}$",
    xy=(-0.8, 0.1),
    xytext=(-1.45, -0.5),
    arrowprops=dict(arrowstyle="->", linewidth=2),
    weight="bold",
    fontsize=14,
)

plt.annotate(
    r"$\mathbf{\sigma_r=1.0}$",
    xy=(-1, 0.6),
    xytext=(-1.45, -0.65),
    weight="bold",
    fontsize=14,
)

# Annotation for a symbol with correlation ρ = 0.57 - j0.57
plt.annotate(
    r"$\mathbf{\rho=0.57-j0.57}$",
    xy=(0.4, -0.4),
    xytext=(0.25, -0.8),
    arrowprops=dict(arrowstyle="->", linewidth=2),
    weight="bold",
    fontsize=14,
)

plt.annotate(
    r"$\mathbf{\sigma_r=0.51}$",
    xy=(0.4, -0.5),
    xytext=(0.25, -0.95),
    weight="bold",
    fontsize=14,
)

# Annotation for a symbol with correlation ρ = 0.66 + j0.66
plt.annotate(
    r"$\mathbf{\rho=0.66+j0.66}$",
    xy=(1, 0.6),
    xytext=(-1.25, 1.5),
    weight="bold",
    fontsize=14,
)

plt.annotate(
    r"$\mathbf{\sigma_r=1.32}$",
    xy=(0.6, 1),
    xytext=(-1.25, 1.35),
    arrowprops=dict(arrowstyle="->", linewidth=2),
    weight="bold",
    fontsize=14,
)

# -----------------------------
# Finalize the Plot
# -----------------------------
# Draw horizontal and vertical axes through zero
plt.axhline(0, color="black", linestyle="-", linewidth=1)
plt.axvline(0, color="black", linestyle="-", linewidth=1)

# Add a colorbar for the contour plot
plt.colorbar()

# Set equal scaling for both axes
plt.axis("equal")

# Label the axes
plt.xlabel(r"$z_I$", fontsize=15)
plt.ylabel(r"$z_Q$", fontsize=15)

# Enable grid for better readability
plt.grid()

# Display the plot
plt.show()

#############################################################################################
#               SECOND PLOT
#############################################################################################
# Set figure size and DPI
plt.figure(figsize=(6.6, 5), dpi=100)

# Generate x values for plotting
x = np.linspace(-3, 3, 1000)

# Draw horizontal and vertical axes with specified properties
x_axis = plt.axhline(0, xmin=0, xmax=1, color="black", linestyle="-", linewidth=1)
y_axis = plt.axvline(0, ymin=0, ymax=1, color="red", linestyle="--", linewidth=2)

# Calculate PDF values for two different complex Gaussian distributions
pdf_real = useful_functions.marginal_pdf_component_ratio_centered_complex_gaussian(
    x, corr_array[0], sigma_r_i[0], sigma_y=1
)
pdf_real2 = useful_functions.marginal_pdf_component_ratio_centered_complex_gaussian(
    x, (0.5 + 0.5j), sigma_r_i[0], sigma_y=1
)

# Plot the PDFs of the two distributions
plt.plot(x, pdf_real, label=r"$\mathbf{f_{Z_I}(z_I,\rho_1,\sigma_r=1.32)}$")
plt.plot(x, pdf_real2, label=r"$\mathbf{f_{Z_I}(z_I,\rho_2,\sigma_r=1.32)}$")

# Integration region (highlight area under the curve for pdf_real and pdf_real2)
x_min = 0
plt.fill_between(
    x, pdf_real, where=(x <= x_min), color="green", alpha=0.3, hatch="/", linewidth=4
)
plt.fill_between(
    x, pdf_real2, where=(x <= x_min), color="yellow", alpha=0.3, hatch="/", linewidth=4
)

# Annotate the plot with arrows pointing to specific points on the curves
plt.annotate(
    r"$\mathbf{F_{Z_I}(z_I=0,\rho_2)}$",
    xy=(-0.1, 0.01),
    xytext=(-3, 0.2),
    arrowprops=dict(arrowstyle="->", linewidth=2),
    weight="bold",
    fontsize=15,
)
plt.annotate(
    r"$\mathbf{F_{Z_I}(z_I=0,\rho_1)}$",
    xy=(-0.1, 0.18),
    xytext=(-3, 0.36),
    arrowprops=dict(arrowstyle="->", linewidth=2),
    weight="bold",
    fontsize=15,
)

# Display grid, legend, and labels
plt.grid()
plt.legend(prop={"weight": "bold", "size": 15})
plt.xlabel(r"$z_I$", fontsize=15)
plt.ylabel(r"$f_{Z_I}(z_I)$", fontsize=14)

# Show the plot
plt.show()
