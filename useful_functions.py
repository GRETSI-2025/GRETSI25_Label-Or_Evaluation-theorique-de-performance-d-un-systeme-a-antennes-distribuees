import tensorflow as tf
import numpy as np
###############################################################################
#                              Useful functions                               #
###############################################################################
def generate_blocked_CIR(unblocked_cir, blockage_type, num_clusters_to_block, custom_mask, cluster_powers, batch_size, num_ofdm_symbols):
    """
    Simulate blockage in a channel impulse response (CIR) by modifying its amplitude coefficients.
    
    Parameters:
        unblocked_cir (list): A two-element list [amplitudes, delays] representing the unblocked CIR.
        blockage_type (str): Type of blockage to simulate. Options are:
            - "strongest_cluster_blocked": Block clusters with highest power.
            - "weakest_cluster_blocked": Block clusters with lowest power.
            - "custom": Use a provided custom mask.
        num_clusters_to_block (int): Number of clusters to block.
        custom_mask (tf.Tensor): Custom blockage mask to use if blockage_type is "custom".
        cluster_powers (tf.Tensor): A tensor representing the powers of clusters.
        batch_size (int): Batch size for tiling tensors.
        num_ofdm_symbols (int): Number of OFDM symbols, used to tile dimensions.
    
    Returns:
        list: A two-element list [blocked_amplitudes, delays] representing the blocked CIR.
    """
    # Extract the amplitude coefficients and delays from the unblocked CIR.
    unblocked_amplitudes = unblocked_cir[0]
    delays = unblocked_cir[1]
    
    # Expand the cluster powers tensor to match the shape of the amplitude coefficients.
    # We perform multiple expand_dims to add extra dimensions, then transpose and tile 
    # so that the resulting tensor matches the shape required for element-wise multiplication.
    expanded_powers = tf.expand_dims(cluster_powers, 0)
    expanded_powers = tf.expand_dims(expanded_powers, 0)
    expanded_powers = tf.expand_dims(expanded_powers, 0)
    expanded_powers = tf.expand_dims(expanded_powers, 0)
    expanded_powers = tf.expand_dims(expanded_powers, 0)
    expanded_powers = tf.expand_dims(expanded_powers, 0)
    # Rearrange dimensions so that the cluster dimension aligns (assumed to be axis 5).
    expanded_powers = tf.transpose(expanded_powers, [0, 1, 2, 3, 4, 6, 5])
    tiled_powers = tf.tile(expanded_powers, [batch_size, 1, 1, 1, 1, 1, num_ofdm_symbols])
    
    # Start with a mask of ones (no blockage) having the same shape as the amplitude coefficients.
    blockage_mask = tf.ones_like(unblocked_amplitudes)
    
    if blockage_type == "strongest_cluster_blocked":
        # Identify the clusters with highest power:
        # 1. Sort the powers in descending order along the cluster axis (axis 5).
        sorted_indices = tf.argsort(tiled_powers, axis=5, direction='DESCENDING')
        # 2. Obtain the ranking (order) for each cluster.
        cluster_ranking = tf.argsort(sorted_indices, axis=5)
        # 3. Create a mask that zeros out the clusters with ranking less than num_clusters_to_block.
        blockage_mask = tf.where(cluster_ranking < num_clusters_to_block,
                                 tf.constant(0, dtype=tf.complex64),
                                 tf.constant(1, dtype=tf.complex64))
        
    elif blockage_type == "weakest_cluster_blocked":
        # Identify the clusters with lowest power:
        # 1. Sort the powers in ascending order along the cluster axis.
        sorted_indices = tf.argsort(tiled_powers, axis=5, direction='ASCENDING')
        # 2. Obtain the ranking for each cluster.
        cluster_ranking = tf.argsort(sorted_indices, axis=5, direction='ASCENDING')
        # 3. Create a mask that zeros out the clusters with ranking less than num_clusters_to_block.
        blockage_mask = tf.where(cluster_ranking < num_clusters_to_block,
                                 tf.constant(0, dtype=tf.complex64),
                                 tf.constant(1, dtype=tf.complex64))
        
    elif blockage_type == "custom":
        # Expand the custom mask to match the shape of the amplitude coefficients.
        blockage_mask = tf.expand_dims(custom_mask, 0)
        blockage_mask = tf.expand_dims(blockage_mask, 0)
        blockage_mask = tf.expand_dims(blockage_mask, 0)
        blockage_mask = tf.expand_dims(blockage_mask, 0)
        blockage_mask = tf.expand_dims(blockage_mask, 0)
        blockage_mask = tf.expand_dims(blockage_mask, 0)
        blockage_mask = tf.transpose(blockage_mask, [0, 1, 2, 3, 4, 6, 5])
        blockage_mask = tf.tile(blockage_mask, [batch_size, 1, 1, 1, 1, 1, num_ofdm_symbols])
        
    else:
        raise ValueError("Invalid blockage type. Choose from 'strongest_cluster_blocked', 'weakest_cluster_blocked', or 'custom'.")
        
    # Apply the blockage mask to the amplitude coefficients.
    blocked_amplitudes = tf.multiply(unblocked_amplitudes, blockage_mask)
    
    # Return the blocked CIR as a list containing the modified amplitudes and the unchanged delays.
    blocked_cir = [blocked_amplitudes, delays]
    return blocked_cir


@tf.function(jit_compile=True)
def rotation_matrix_xla(angle_z, angle_y, angle_x):
    """
    Compute the composite rotation matrix given Euler angles for rotations about the z, y, and x axes.

    The rotation matrices are defined as:
      - Rz: Rotation about the z-axis by 'angle_z'
      - Ry: Rotation about the y-axis by 'angle_y'
      - Rx: Rotation about the x-axis by 'angle_x'
    
    The composite rotation is computed as:
      R = Rx * (Ry * Rz)
    
    This means a vector is first rotated by Rz, then by Ry, and finally by Rx.

    Parameters:
        angle_z (tf.Tensor): Scalar tensor (in radians) for the rotation angle about the z-axis.
        angle_y (tf.Tensor): Scalar tensor (in radians) for the rotation angle about the y-axis.
        angle_x (tf.Tensor): Scalar tensor (in radians) for the rotation angle about the x-axis.

    Returns:
        tf.Tensor: A 3x3 tensor representing the composite rotation matrix.
    """
    # Define rotation matrix about the z-axis
    Rz = tf.stack([
        tf.stack([tf.cos(angle_z), -tf.sin(angle_z), 0.0], axis=0),
        tf.stack([tf.sin(angle_z),  tf.cos(angle_z), 0.0], axis=0),
        tf.stack([0.0,              0.0,             1.0], axis=0)
    ], axis=0)

    # Define rotation matrix about the y-axis
    Ry = tf.stack([
        tf.stack([ tf.cos(angle_y), 0.0, tf.sin(angle_y)], axis=0),
        tf.stack([ 0.0,             1.0, 0.0],             axis=0),
        tf.stack([-tf.sin(angle_y), 0.0, tf.cos(angle_y)], axis=0)
    ], axis=0)

    # Define rotation matrix about the x-axis
    Rx = tf.stack([
        tf.stack([1.0,             0.0,              0.0],            axis=0),
        tf.stack([0.0, tf.cos(angle_x), -tf.sin(angle_x)], axis=0),
        tf.stack([0.0, tf.sin(angle_x),  tf.cos(angle_x)], axis=0)
    ], axis=0)

    # Combine the rotations:
    # The multiplication order ensures that a vector is first rotated by Rz, then Ry, then Rx.
    composite_rotation_matrix = tf.linalg.matmul(Rx, tf.linalg.matmul(Ry, Rz))
    
    return composite_rotation_matrix


def plot_coordinate_system(ax, origin, R, color, label):
    """
    Plot a coordinate system with a given rotation matrix R and origin
    """
    # Basis vectors
    i = R[:, 0]
    j = R[:, 1]
    k = R[:, 2]

    ax.quiver(*origin, *i, color=color, length=1.0, label=f'{label}-x', normalize=True)
    ax.quiver(*origin, *j, color=color, length=1.0, label=f'{label}-y', normalize=True)
    ax.quiver(*origin, *k, color=color, length=1.0, label=f'{label}-z', normalize=True)
    
    # Annotate the axes
    ax.text(origin[0] + i[0], origin[1] + i[1], origin[2] + i[2], f'{label}-x', color=color, fontsize=12, ha='center', va='center')
    ax.text(origin[0] + j[0], origin[1] + j[1], origin[2] + j[2], f'{label}-y', color=color, fontsize=12, ha='center', va='center')
    ax.text(origin[0] + k[0], origin[1] + k[1], origin[2] + k[2], f'{label}-z', color=color, fontsize=12, ha='center', va='center')
 

def compute_marginal_cdf(val_threshold, corr_coeff, std_numerator, std_denominator, for_real_part=True):
    """
    Computes the marginal CDF for the real or imaginary part of the ratio of two correlated complex Gaussian variables having zero means.
    
    Parameters:
    - val_threshold (float): The maximum value for the CDF computation.
    - corr_coeff (complex): The complex correlation coefficient.
    - std_numerator (float): Standard deviation of the numerator (X).
    - std_denominator (float): Standard deviation of the denominator (Y).
    - for_real_part (bool): If True, computes for the real part; otherwise, for the imaginary part.
    
    Returns:
    - float: The calculated marginal CDF value.
    """
    real_part_corr = np.real(corr_coeff)
    imag_part_corr = np.imag(corr_coeff)

    if for_real_part:
        relative_value = (val_threshold / std_numerator - real_part_corr / std_denominator)
    else:
        relative_value = (val_threshold / std_numerator - imag_part_corr / std_denominator)

    denominator_value = np.sqrt((1 - np.abs(corr_coeff)**2) / std_denominator**2 + relative_value**2)

    marginal_cdf = 0.5 * (relative_value / denominator_value + 1)
    
    return marginal_cdf

def corr_num_denom(symbol_d, bs_power, var_hB, var_hNB, M, ebno_base10):
    """
    Compute the correlation ratio based on channel parameters and modulation settings.
    
    This function calculates a correlation ratio that is used in some communication system
    performance metrics. The ratio is computed by weighting the variance of one channel state (var_hB)
    by the base station (BS) power and combining it with the variance of another channel state (var_hNB)
    along with a noise term derived from Eb/N0 and the modulation order.
    
    The correlation ratio is given by:
    
        corr_ratio = phase_factor * ( sum(bs_power * var_hB) /
                     sqrt( (sum(bs_power * var_hB) + noise_term) * sum(bs_power * var_hNB) ) )
    
    where:
      - phase_factor is the unit-magnitude complex number preserving the phase of symbol_d.
      - noise_term is computed as: (1/ebno_base10) * (2*(M-1)) / (3*|symbol_d|^2*log2(M)).
    
    Parameters
    ----------
    symbol_d : complex
        The complex symbol for which the correlation is computed.
    bs_power : array_like
        An array of power values for each base station or link.
    var_hB : array_like
        An array of variance values for the channel state (e.g., blocked channel variance).
    var_hNB : array_like
        An array of variance values for the alternative channel state (e.g., non-blocked channel variance).
    M : int
        The modulation order (e.g., M=4 for QPSK). Used in the noise computation.
    ebno_base10 : float
        The Eb/N0 value in linear (base-10) scale.
    
    Returns
    -------
    corr_ratio : complex
        The computed correlation ratio, with the phase of symbol_d preserved.
    """
    
    # Extract the real and imaginary parts of symbol_d to compute its energy.
    a = np.real(symbol_d)
    b = np.imag(symbol_d)
    
    # Compute the noise term.
    # Formula: noise_term = (1/ebno_base10) * (2*(M-1)) / (3*|symbol_d|^2 * log2(M))
    N0_over_mod_x_squared = (ebno_base10 ** -1) * (2 * (M - 1)) / (3 * (a**2 + b**2) * np.log2(M))
    
    # Weight the variance for the 'hB' channel state by the BS power.
    weighted_var_link = var_hB * bs_power
    
    # Compute the numerator as the sum of the weighted variance.
    numerator = np.sum(weighted_var_link)
    
    # Compute the denominator:
    #   - The first term is the sum of the weighted variance plus the noise term.
    #   - The second term is the sum of (bs_power * var_hNB) for the alternative channel state.
    denominator = np.sqrt((np.sum(weighted_var_link) + N0_over_mod_x_squared) * np.sum(var_hNB * bs_power))
    
    # Extract the phase factor from symbol_d.
    # This ensures that the final ratio preserves the phase of the input symbol.
    phase_factor = symbol_d / np.abs(symbol_d)
    
    # Compute the final correlation ratio.
    corr_ratio = phase_factor * (numerator / denominator)
    
    return corr_ratio

def joint_pdf_ratio_complex_gaussian(m_x, m_y, sigma_x, sigma_y, rho, span_real, span_complex):
    """
    Compute the joint probability density function (PDF) of the ratio Z = X/Y,
    where X and Y are two correlated complex Gaussian random variables.
    
    The variables X and Y are assumed to be circularly symmetric complex Gaussian (CN)
    with means m_x and m_y, variances sigma_x**2 and sigma_y**2, respectively. The
    correlation between X and Y is given by the complex number rho (rho = rho_r + i*rho_i).
    
    The formula implemented here is based on the paper:
    "On the ratio of two correlated complex Gaussian random variables" by Yang Li.
    
    Parameters
    ----------
    m_x : complex or float
        Mean of the complex random variable X.
    m_y : complex or float
        Mean of the complex random variable Y.
    sigma_x : float
        Standard deviation of X.
    sigma_y : float
        Standard deviation of Y.
    rho : complex
        Complex correlation coefficient between X and Y.
    span_real : array_like
        1D array of values for the real part of Z where the PDF is evaluated.
    span_complex : array_like
        1D array of values for the imaginary part of Z where the PDF is evaluated.
    
    Returns
    -------
    z_real : ndarray
        2D array (meshgrid) of the real parts corresponding to the grid of Z values.
    z_imag : ndarray
        2D array (meshgrid) of the imaginary parts corresponding to the grid of Z values.
    f_z : ndarray
        2D array of the joint PDF values of Z = X/Y evaluated over the grid.
    """
    
    # Create a meshgrid for the real and imaginary parts of Z.
    z_real, z_imag = np.meshgrid(span_real, span_complex)
    # Combine the real and imaginary parts to form the complex variable Z.
    z = z_real + 1j * z_imag

    # Intermediate parameter a: scales the effect of correlation on sigma_x relative to sigma_y.
    a = rho * sigma_x / sigma_y
    
    # Intermediate parameter b: difference between m_x and a scaled version of m_y.
    b = m_x - (rho * sigma_x * m_y) / sigma_y
    
    # Parameter c: accounts for the variance reduction due to correlation.
    c = sigma_x**2 * (1 - np.abs(rho)**2)
    
    # Compute alpha, a parameter combining the offsets and variances.
    alpha = (np.abs(b)**2) / c + (np.abs(m_y)**2) / (sigma_y**2)
    
    # Compute g_z, which depends on the distance of z from a, scaled by c and sigma_y.
    g_z = (np.abs(z - a)**2) / c + 1 / (sigma_y**2)
    
    # Compute lambda_z, which combines the effect of b and m_y on z.
    lambda_z = np.abs(np.conjugate(z - a) * b / c + m_y / (sigma_y**2))
    
    # Finally, compute the joint PDF f_z using the derived parameters.
    f_z = (np.exp(-alpha) / (c * np.pi * sigma_y**2 * g_z**2) *
           np.exp((lambda_z**2) / g_z) *
           (1 + (lambda_z**2) / g_z))
    
    return z_real, z_imag, f_z

def marginal_pdf_component_ratio_centered_complex_gaussian(x, rho, sigma_x, sigma_y, real=True):
    """
    Compute the marginal probability density function (PDF) of the real or imaginary 
    component of the ratio Z = X/Y, where X and Y are centered complex Gaussian random variables.
    
    X and Y are assumed to be circularly symmetric complex Gaussian random variables with 
    zero mean, and standard deviations sigma_x and sigma_y, respectively. The complex 
    correlation coefficient between X and Y is given by 'rho'. This function returns the marginal PDF 
    of either the real or the imaginary part of Z based on the 'real' flag.
    
    The marginal PDF is given by:
    
        f_Z(x) = (1 - |rho|^2) / (2 * sigma_x * sigma_y^2) *
                 { [ (x/sigma_x) - (component/sigma_y) ]^2 + (1 - |rho|^2)/(sigma_y^2) }^(-3/2)
    
    where 'component' is Re(rho) if real=True, and Im(rho) if real=False.
    
    Parameters
    ----------
    x : float or array_like
        The value(s) at which to evaluate the marginal PDF.
    rho : complex
        The complex correlation coefficient between X and Y.
    sigma_x : float
        The standard deviation of X.
    sigma_y : float
        The standard deviation of Y.
    real : bool, optional
        If True (default), compute the marginal PDF for the real part of Z.
        If False, compute the marginal PDF for the imaginary part of Z.
    
    Returns
    -------
    f_Z : float or array_like
        The evaluated marginal PDF of the chosen component of Z at x.
    """
    
    # Choose the appropriate component of rho: real part if 'real' is True; otherwise, imaginary part.
    component = np.real(rho) if real else np.imag(rho)
    
    # Compute the constant numerator factor in the PDF expression.
    numerator = (1 - np.abs(rho)**2) / (2 * sigma_x * sigma_y**2)
    
    # Compute the term inside the parentheses of the PDF:
    # It includes the deviation of the scaled variable x from the scaled chosen component of rho.
    term = ((x / sigma_x) - (component / sigma_y))**2 + (1 - np.abs(rho)**2) / (sigma_y**2)
    
    # The marginal PDF is given by the numerator multiplied by the term raised to the power (-3/2).
    f_Z = numerator * term**(-3/2)
    
    return f_Z
