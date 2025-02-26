# library importation
#############################################################################################
import matplotlib.pyplot as plt
import numpy as np
import sionna   # numerical communication library
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
# own module importation
import useful_functions

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)


#############################################################################################
# PARAMETERS/SCENARIO FOR SIMULATION
#############################################################################################
N_BS = 2
SIMULATION_PARAM = {
    "num_ofdm_symbols":14,
    "subcarrier_spacing" :15e3, # in Hz
    "carrier_frequency": 3.5e9,  # in Hz
    "fft_size": 102,
    "num_guard_carrier": [0, 0],
    "num_bits_per_symbol":2,
    "BS_power" : 1/N_BS*np.ones(N_BS), # power equally split accross BSs. It reflects BS to UE distances.
    "num_ut_ant":1,
    "num_bs_ant":1,
    "cdl_model" : ["C"]*N_BS, # A,B,C => NLoS & D,E => LoS channel model from 3GPP TR38.901
    "delay_spread" : 100e-9, # in second
    "normalize_channel" : False,
    "show_cluster" : False,
    "channel_blocked_equalizer" : False, # To model cluster loss power
    "nb_clusters_blocked" : 0 , # ignored if custom mode
    "mask_clusters_blocked_per_link" : [ tf.constant([1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],dtype=tf.complex64), # MASK used must have same shape as powers (3GPP TABLES) : # CDL a => 23 ; CDL b => 23; CDL c=> 24
                                         tf.constant([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],dtype=tf.complex64)
                                       ], 
    # number of elements has to be equal to the number of clusters : 23 : [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] 13: [1,1,1,1,1,1,1,1,1,1,1,1,1]
    "DMIMO_blockage_mode" : np.zeros([N_BS]), # 0 => blocked link     
    "precoding" : ["steering_angles"],#["MRT"], #None,"SVD", ZF,"MRT" , "steering_angles"
    "graph_mode" : None,  # None,"xla","graph"
    "ebno_db" : list(np.arange( -30, 31, 2)), 
    "batch_size" : 2048,
    "num_target_bit_errors":None,
    "target_ber" : 10e-6,
    "max_mc_iter" : 10,
}

# for figure display purposes
blocked_indices = [tf.where(tf.equal(mask, tf.constant(0, dtype=mask.dtype))).numpy().flatten().tolist() for mask in SIMULATION_PARAM["mask_clusters_blocked_per_link"]]
# Convert to 1-based indexing by adding 1 to each index
adjusted_blocked_indices = [[idx + 1 for idx in indices] for indices in blocked_indices]
blocked_counts = [len(indices) for indices in adjusted_blocked_indices]
info_text = "\n".join(
    f"Base Station {i+1} : {count} blocked cluster(s) at index(s) {indices}"
    for i, (count, indices) in enumerate(zip(blocked_counts, adjusted_blocked_indices))
)

#############################################################################################
# Model definition
#############################################################################################
class DistributedMIMOModel(tf.keras.Model):
    """
    DistributedMIMOModel simulates OFDM MIMO transmissions over 3GPP CDL channel models,
    capturing multipoint-to-point communications between user equipment (UE) and a distributed
    cluster of base stations (BSs). The model includes dynamic channel conditions such as
    partial blockage of specific propagation clusters, enabling accurate evaluation of Bit 
    Error Rate (BER) under various blockage scenarios.

    Assumptions:
      - A single transmit antenna per BS is assumed (extension to antenna arrays with beamforming 
        is possible).
      - The simulation process involves transmitting data, receiving estimated bits, and using
        Sionna functions to compute the BER.
    """
    def __init__(self,
                 num_ofdm_symbols,
                 subcarrier_spacing,
                 carrier_frequency,
                 fft_size,
                 num_guard_carriers,
                 bits_per_symbol,
                 bs_power,
                 num_ue_antennas,
                 num_bs_antennas,
                 cdl_models,
                 delay_spread,
                 normalize_channel,
                 channel_blocked_equalizer,
                 blocked_clusters_count,
                 mask_clusters_blocked_per_link,
                 distributed_mimo_blockage_mode,
                 precoding,
                 graph_mode):
        """
        Initialize the Distributed MIMO Model with simulation, channel, and blockage parameters.
        
        Parameters:
            num_ofdm_symbols (int): Number of OFDM symbols.
            subcarrier_spacing (float): Subcarrier spacing (Hz).
            carrier_frequency (float): Carrier frequency (Hz).
            fft_size (int): Size of the FFT.
            num_guard_carriers (int): Number of guard carriers.
            bits_per_symbol (int): Bits per modulation symbol.
            bs_power (array-like): Transmit power for each base station.
            num_ue_antennas (int): Number of antennas at the UE.
            num_bs_antennas (int): Number of antennas per BS.
            cdl_models (list): List of CDL channel models (one per BS).
            delay_spread (float): Delay spread parameter for the channel model.
            normalize_channel (bool): Flag to normalize channel gains.
            channel_blocked_equalizer: Equalizer instance for blocked channel scenarios.
            blocked_clusters_count (int): Number of clusters to block.
            mask_clusters_blocked_per_link: Custom mask for blocking clusters per link.
            distributed_mimo_blockage_mode: Mode for DMIMO blockage simulation.
            precoding (bool): Flag indicating whether to use precoding.
            graph_mode (bool): Flag to enable graphical output (e.g., plotting coordinate systems).
        """
        super().__init__()

        # Store provided parameters
        self.bs_power = bs_power
        self.num_bs = len(bs_power)  # Number of base stations
        self.cdl_models = cdl_models
        self.delay_spread = delay_spread
        self.normalize_channel = normalize_channel
        self.channel_blocked_equalizer = channel_blocked_equalizer
        self.blocked_clusters_count = blocked_clusters_count
        self.mask_clusters_blocked_per_link = mask_clusters_blocked_per_link
        self.distributed_mimo_blockage_mode = distributed_mimo_blockage_mode
        self.precoding = precoding
        self.graph_mode = graph_mode
        self.carrier_frequency = carrier_frequency 
        self.subcarrier_spacing = subcarrier_spacing
        self.fft_size = fft_size 
        self.num_ofdm_symbols = num_ofdm_symbols 
        self.num_ue_antennas = num_ue_antennas 
        self.num_bs_antennas = num_bs_antennas 
        self.bits_per_symbol = bits_per_symbol
        
        # Determine the number of streams per transmission: limited by the minimum number of antennas
        self.num_streams_per_tx = np.minimum(self.num_ue_antennas, self.num_bs_antennas)
        self.num_guard_carriers = num_guard_carriers
        
        # Hardcoded parameters for simulation (these may be adjusted as needed)
        self.dc_null = False
        self.pilot_pattern = None  # Options: "kronecker", "empty", or None
        self.pilot_ofdm_symbol_indices = [0, 0]  # Pilot OFDM symbol indices
        self.direction = "downlink"
        self.num_tx_antennas = self.num_bs_antennas
        self.num_rx_antennas = self.num_ue_antennas
        self.cyclic_prefix_length = 0
        self.coderate = 1
        self.blockage = False
        
        # Power allocation across streams:
        # This scales the power by taking the square root and dividing by the square root of the number
        # of streams per transmission. The result is reshaped and cast to a complex tensor.
        self.power_allocation = tf.sqrt(self.bs_power) / tf.sqrt(tf.cast(self.num_streams_per_tx, tf.float64))
        self.power_allocation = tf.cast(tf.reshape(self.power_allocation,tf.shape(tf.transpose(self.bs_power))),tf.complex64)
        
        #######################################################################
        #                   Define Antenna Arrays for UE and BS               #
        #######################################################################
        # Create the UE (User Equipment) antenna array using an omnidirectional pattern.
        self.ue_array = sionna.channel.tr38901.AntennaArray(
            num_rows=1,
            num_cols=self.num_ue_antennas,
            polarization="single",
            polarization_type="H",  # "V", "H", or "cross"
            antenna_pattern="omni",   # Options: "38.901", "omni"
            carrier_frequency=self.carrier_frequency)
        self.ue_array.show()
        plt.title("User Equipment Antenna Array")
        
        # UE orientation (in radians): rotations about Z, Y, and X axes respectively.
        self.ue_orientation = tf.constant([np.deg2rad(0.0), np.deg2rad(0.0), 0.0], tf.float32)
        
        # Create the BS (Base Station) antenna array.
        # Here, a single antenna per BS is assumed.
        self.bs_array = sionna.channel.tr38901.AntennaArray(
            num_rows=1,
            num_cols=1,  # Single antenna per BS; adjust if using multiple antennas.
            polarization="single",
            polarization_type="H",
            antenna_pattern="omni",
            carrier_frequency=self.carrier_frequency)
        self.bs_array.show()
        plt.title("Base Station Antenna Array")
        
        # Default BS orientation: no rotation (aligned with the Global Coordinate System)
        self.bs_orientation_array = tf.broadcast_to(tf.constant([0.0, 0.0, 0.0], tf.float32), [self.num_bs, 3])
        self.global_coord_system_orientation = tf.constant([0.0, 0.0, 0.0], tf.float32)
        
        #######################################################################
        #               Instantiate CDL Channel Models for Each BS            #
        #######################################################################
        self.cdl_channel_models = []
        for bs_index, bs_model in enumerate(self.cdl_models):
            cdl_channel = sionna.channel.tr38901.CDL(
                model=bs_model,
                delay_spread=self.delay_spread,
                carrier_frequency=self.carrier_frequency,
                ut_array=self.ue_array,
                bs_array=self.bs_array,
                direction=self.direction,
                ut_orientation=self.ue_orientation,
                bs_orientation=self.bs_orientation_array[bs_index],
                min_speed=0)
            self.cdl_channel_models.append(cdl_channel)
        
        #######################################################################
        #           Initialize Sionna Components for the Simulation           #
        #######################################################################
        self.stream_manager = sionna.mimo.StreamManagement(np.array([[1]]), self.num_streams_per_tx)
        self.resource_grid = sionna.ofdm.ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=1,
            num_streams_per_tx=self.num_streams_per_tx,
            cyclic_prefix_length=self.cyclic_prefix_length,
            num_guard_carriers=self.num_guard_carriers,
            dc_null=self.dc_null,
            pilot_pattern=self.pilot_pattern,
            pilot_ofdm_symbol_indices=self.pilot_ofdm_symbol_indices)
        
        self.resource_grid_demapper = sionna.ofdm.ResourceGridDemapper(self.resource_grid, self.stream_manager)
        self.frequencies = sionna.channel.subcarrier_frequencies(self.resource_grid.fft_size,
                                                                  self.resource_grid.subcarrier_spacing)
        self.channel_frequency = sionna.channel.ApplyOFDMChannel(add_awgn=True)
        self.modulator = sionna.ofdm.OFDMModulator(self.cyclic_prefix_length)
        self.binary_source = sionna.utils.BinarySource()
        
        # Total number of data bits transmitted in one resource grid
        self.num_data_bits = int(self.resource_grid.num_data_symbols * self.bits_per_symbol)
        
        # Choose mapping and demapping based on modulation order:
        # For a single bit per symbol, PAM is used; otherwise, QAM is used.
        if self.bits_per_symbol == 1:
            self.mapper = sionna.mapping.Mapper("pam", self.bits_per_symbol)
            self.demapper = sionna.mapping.Demapper("app", "pam", self.bits_per_symbol, hard_out=True)
        else:
            self.mapper = sionna.mapping.Mapper("qam", self.bits_per_symbol)
            self.demapper = sionna.mapping.Demapper("app", "qam", self.bits_per_symbol, hard_out=True)
        
        self.resource_grid_mapper = sionna.ofdm.ResourceGridMapper(self.resource_grid)
        self.equalizer = sionna.ofdm.ZFEqualizer(self.resource_grid, self.stream_manager)
        self.remove_nulled_subcarriers = sionna.ofdm.RemoveNulledSubcarriers(self.resource_grid)
        
        #######################################################################
        #                   Plot Coordinate Systems (if enabled)              #
        #######################################################################
        if self.graph_mode:
            self.plot_coordinate_systems()
     
    #@tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        """
        Execute one Monte Carlo simulation iteration for the distributed MIMO model.

        This method performs the following steps:
        1. Initializes lists to store per-base station channel impulse responses (CIRs) 
            (with and without blockage), OFDM channel responses, and received signals.
        2. Generates transmit bits, maps them to symbols, and places them on the resource grid.
        3. Applies power allocation and optional beamforming (precoding) per base station.
        4. Simulates the channel for each base station, applying blockage effects based on 
            the CDL model type.
        5. Converts CIRs to frequency-domain channel responses.
        6. Combines signals from all base stations and performs channel estimation.
        7. Equalizes the combined signal and performs demapping to obtain bit estimates.
        8. Optionally, a custom demapping procedure is executed for specific modulation orders.
        
        Parameters:
        batch_size (int): Number of samples to simulate.
        ebno_db (float): Eb/N0 value in dB for the current simulation.
        
        Returns:
        tuple: (tx_bits, estimated_bits) where tx_bits are the transmitted bits and 
                estimated_bits are the receiver's bit estimates.
        """

        ###########################################################################
        # Initialize lists to collect simulation data per base station
        ###########################################################################
        cir_unblocked_list = []         # Original CIRs (without blockage)
        cir_blocked_list = []           # Blocked CIRs (after applying blockage)
        channel_freq_list = []          # Frequency-domain channel responses (unblocked)
        channel_freq_blocked_list = []  # Frequency-domain channel responses (blocked)
        received_signals_list = []      # Received signals per BS
        estimated_channel_list = []     # Channel estimates for equalization
        blocked_channel_list = []       # Blocked channel estimates (if used)

        # Combined received signal (for D-MIMO processing)
        total_received_signal = tf.zeros([batch_size, 1, self.num_rx_antennas,
                                            self.num_ofdm_symbols, self.fft_size],
                                        dtype=tf.complex64)
        
        ###########################################################################
        # Initialize channel estimation accumulators
        ###########################################################################
        total_estimated_channel = tf.zeros([batch_size, 1, self.num_rx_antennas, 1,
                                            self.num_streams_per_tx, self.num_ofdm_symbols,
                                            self.fft_size],
                                            dtype=tf.complex64)
        ###########################################################################
        # PHY Layer: Transmitter operations
        ###########################################################################
        # Convert Eb/N0 in dB to noise variance using Sionna helper function.
        noise = sionna.utils.ebnodb2no(ebno_db, self.bits_per_symbol, self.coderate, self.resource_grid)

        # Generate transmit bits and map them to modulation symbols.
        tx_bits = self.binary_source([batch_size, 1, self.num_streams_per_tx, self.num_data_bits])
        mapped_symbols = self.mapper(tx_bits)

        # Map symbols onto the OFDM resource grid.
        resource_grid_symbols = self.resource_grid_mapper(mapped_symbols)

        ###########################################################################
        # Apply power allocation and (optional) precoding per Base Station (BS)
        ###########################################################################
        tx_power_allocated_rg = []  # Resource grid symbols after power allocation per BS
        tx_precoded_rg = []         # Precoded resource grid symbols per BS

        for bs_idx, bs_power_value in enumerate(self.bs_power):
            # Allocate power for this BS
            allocated_symbols = tf.math.multiply(resource_grid_symbols, self.power_allocation[bs_idx])
            tx_power_allocated_rg.append(allocated_symbols)

            ###############################################################
            # Optional Precoding (Beamforming)
            ###############################################################
            # Use a simple normalized beamforming vector (all ones)
            bf_weights = tf.cast(tf.ones(self.num_bs_antennas), tf.complex64)
            bf_weights = bf_weights / tf.norm(bf_weights)
            bf_weights = tf.reshape(bf_weights, [1, 1, self.num_bs_antennas, 1, 1])
            allocated_symbols_repeated = tf.tile(allocated_symbols, [1, 1, self.num_bs_antennas, 1, 1])
            precoded_symbols = allocated_symbols_repeated * bf_weights
            tx_precoded_rg.append(precoded_symbols)

            ###################################################################
            # Simulate Channel Impulse Responses (CIR) for each BS
            ###################################################################
            # Obtain the unblocked CIR from the CDL channel model for this BS.
            cir_unblocked = self.cdl_channel_models[bs_idx](batch_size,
                                                            self.resource_grid.num_ofdm_symbols,
                                                            1 / self.resource_grid.ofdm_symbol_duration)
            cir_unblocked_list.append(cir_unblocked)

            ###############################################################
            # Apply blockage to the CIR based on the CDL model type
            ###############################################################
            # Predefined delay values for each CDL model type are used to sort clusters.
            delay_values = self.cdl_channel_models[bs_idx].delays
            # Sort delays to determine cluster order (coefficients may not be stored in delay order)
            sorted_delay_indices = tf.argsort(delay_values)
            #sorted_indices_tf = tf.constant(sorted_delay_indices, dtype=tf.int64)
            # Gather the corresponding blockage mask based on sorted delays
            blockage_mask = tf.gather(self.mask_clusters_blocked_per_link[bs_idx], sorted_delay_indices)

            # Generate the blocked CIR using a helper function.
            cir_blocked = useful_functions.generate_blocked_CIR(
                cir_unblocked,
                "custom",
                self.blocked_clusters_count,
                blockage_mask,
                self.cdl_channel_models[bs_idx].powers,
                batch_size,
                self.resource_grid.num_ofdm_symbols)
            cir_blocked_list.append(cir_blocked)

            ###################################################################
            # Convert CIRs to Frequency-Domain Channel Responses
            ###################################################################
            # The "normalize" parameter determines whether channel gains are normalized.
            channel_freq = sionna.channel.cir_to_ofdm_channel(
                self.frequencies, cir_unblocked_list[bs_idx][0], cir_unblocked_list[bs_idx][1],
                normalize=self.normalize_channel)
            channel_freq_blocked = sionna.channel.cir_to_ofdm_channel(
                self.frequencies, cir_blocked_list[bs_idx][0], cir_blocked_list[bs_idx][1],
                normalize=self.normalize_channel)
            channel_freq_list.append(channel_freq)
            channel_freq_blocked_list.append(channel_freq_blocked)

            ###################################################################
            # Apply Channel and Noise to Generate Received Signals
            ###################################################################
            # For downlink, noise is added only for the first BS; subsequent BS signals use zero noise.
            if self.direction == "downlink":
                if bs_idx == 0:
                    rx_signal = self.channel_frequency([tx_precoded_rg[bs_idx], channel_freq_blocked, noise])
                else:
                    rx_signal = self.channel_frequency([tx_precoded_rg[bs_idx], channel_freq_blocked, 0])
                received_signals_list.append(rx_signal)

            ###################################################################
            # Channel Estimation
            ###################################################################
            # Depending on whether blockage is enabled and the equalizer configuration,
            # select either the blocked or unblocked channel estimate.
            if self.blockage:
                if self.channel_blocked_equalizer:  # Not representative of a real-life scenario
                    est_channel = self.remove_nulled_subcarriers(channel_freq_blocked)
                else:
                    est_channel = self.remove_nulled_subcarriers(channel_freq)
                    blocked_channel = self.remove_nulled_subcarriers(channel_freq_blocked)
                    blocked_channel_list.append(blocked_channel)
            else:
                blocked_channel = self.remove_nulled_subcarriers(channel_freq_blocked)
                blocked_channel_list.append(blocked_channel)
                est_channel = self.remove_nulled_subcarriers(channel_freq)
            estimated_channel_list.append(est_channel)

            ###################################################################
            # Combine Received Signals and Accumulate Channel Estimates
            ###################################################################
            total_received_signal = tf.math.add(total_received_signal, received_signals_list[bs_idx])
            # Weight the channel estimate by the square root of the BS power.
            total_estimated_channel = tf.math.add(total_estimated_channel,
                                                tf.sqrt(tf.cast(bs_power_value,dtype=tf.complex64)) * est_channel)

        # End loop over base stations

        ###########################################################################
        # Equalization: Combine signals from all BSs using beamforming weights
        ###########################################################################
        # Reshape beamforming weights from the last iteration to match equalizer dimensions.
        bf_weights_eq = tf.cast(tf.reshape(bf_weights, [1, 1, 1, 1, self.num_bs_antennas, 1, 1]), tf.complex64)
        # Equalize by dividing the total received signal by the weighted channel estimate.
        equalized_symbols = tf.expand_dims(total_received_signal, axis=1) / tf.reduce_sum(total_estimated_channel * bf_weights_eq, axis=4)
        # Reshape equalized symbols to a 2D grid: [batch_size, total_num_symbols]
        equalized_symbols = tf.reshape(equalized_symbols,
                                    [batch_size, 1, 1, 1, self.num_ofdm_symbols * self.fft_size])
        equalized_symbols = tf.squeeze(equalized_symbols, axis=1)

        # Demap equalized symbols to generate soft bit estimates using the configured demapper.
        demapped_llr = self.demapper([equalized_symbols, noise])

        ###########################################################################
        # Optional: Custom Demapping Implementation for Specific Modulation Orders
        ###########################################################################
        # For example, for QAM-4 (2 bits per symbol) or QAM-16 (4 bits per symbol), a custom mapping is applied.
        if self.bits_per_symbol == 2:
            energy = 2
            num_bits_per_symbol = 2
            qam4_points = {
                (1, 1): -1 - 1j, (1, 0): -1 + 1j,
                (0, 1):  1 - 1j, (0, 0):  1 + 1j,
            }
            constellation_points = qam4_points
        elif self.bits_per_symbol == 4:
            energy = 10
            num_bits_per_symbol = 4
            qam16_points = {
                (1, 1, 1, 1): -3 - 3j, (1, 1, 1, 0): -3 - 1j,
                (1, 1, 0, 1): -1 - 3j, (1, 1, 0, 0): -1 - 1j,
                (1, 0, 1, 1): -3 + 3j, (1, 0, 1, 0): -3 + 1j,
                (1, 0, 0, 1): -1 + 3j, (1, 0, 0, 0): -1 + 1j,
                (0, 1, 1, 1):  3 - 3j, (0, 1, 1, 0):  3 - 1j,
                (0, 1, 0, 1):  1 - 3j, (0, 1, 0, 0):  1 - 1j,
                (0, 0, 1, 1):  3 + 3j, (0, 0, 1, 0):  3 + 1j,
                (0, 0, 0, 1):  1 + 3j, (0, 0, 0, 0):  1 + 1j,
            }
            constellation_points = qam16_points

        # Convert constellation points and bit labels into tensors.
        constellation_points_array = np.array(list(constellation_points.values()))
        constellation_bits_array = tf.constant(list(constellation_points.keys()), dtype=tf.int32)
        # Normalize the constellation points.
        constellation_points_normalized = constellation_points_array / tf.sqrt(tf.constant(energy, dtype=tf.complex64))
        constellation_points_normalized = tf.reshape(constellation_points_normalized,
                                                    [1, 1, 1, constellation_points_normalized.shape[0]])
        # Compute distances between each equalized symbol and every constellation point.
        distances = tf.abs(tf.expand_dims(equalized_symbols, axis=-1) - constellation_points_normalized)
        # Find the index of the closest constellation point.
        closest_point_indices = tf.argmin(distances, axis=-1)
        # Map indices to corresponding bit labels.
        demapped_symbols = tf.gather(constellation_bits_array, closest_point_indices)
        # Use the shape of the originally mapped symbols to infer the total symbol count.
        demapped_symbols_reshaped = tf.reshape(demapped_symbols,
                                            (batch_size, 1, 1, tf.shape(mapped_symbols)[-1] * num_bits_per_symbol))
        custom_demapped_llr = tf.cast(demapped_symbols_reshaped, dtype=tf.float32)

        # Final bit estimate: Here we use the LLR output from the configured demapper.
        estimated_bits = custom_demapped_llr #demapped_llr

        return tx_bits, estimated_bits

    def plot_coordinate_systems(self):
        """
        Plot the coordinate systems for the UE and each BS relative to the Global Coordinate System (GCS).
        """
        # Compute the UE rotation matrix using the provided orientation angles.
        ue_rotation_matrix = useful_functions.rotation_matrix_xla(
            self.ue_orientation[0], self.ue_orientation[1], self.ue_orientation[2])
        
        # Create a 3D figure for plotting the coordinate systems.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the UE local coordinate system (LCS)
        useful_functions.plot_coordinate_system(ax, origin=[0, 0, 0],
                                                  R=ue_rotation_matrix,
                                                  color='r',
                                                  label='UE LCS')
        
        # Plot the Global Coordinate System (GCS) for reference
        useful_functions.plot_coordinate_system(ax, origin=[-2, -2, -2],
                                                  R=np.eye(3),
                                                  color='black',
                                                  label='GCS')
        
        # Place BSs uniformly on a circle around the origin
        angles = 2 * np.pi * np.arange(self.num_bs) / self.num_bs
        radius = 3.0  # Circle radius
        bs_x_positions = radius * np.cos(angles)
        bs_y_positions = radius * np.sin(angles)
        
        # Plot the coordinate system for each BS
        for bs_index in range(self.num_bs):
            bs_rotation_matrix = useful_functions.rotation_matrix_xla(
                self.bs_orientation_array[bs_index][0],
                self.bs_orientation_array[bs_index][1],
                self.bs_orientation_array[bs_index][2])
            useful_functions.plot_coordinate_system(ax,
                                                      origin=[bs_x_positions[bs_index],
                                                              bs_y_positions[bs_index],
                                                              0],
                                                      R=bs_rotation_matrix,
                                                      color='b',
                                                      label='BS LCS')
        
        # Set plot limits and axis labels
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title("Geometrical Layout of the Scenario")
        plt.show()


#############################################################################################
# INITIALISATION OF THE MODEL
#############################################################################################
model_distributed_MIMO = DistributedMIMOModel(num_ofdm_symbols = SIMULATION_PARAM["num_ofdm_symbols"],
                                                subcarrier_spacing = SIMULATION_PARAM["subcarrier_spacing"],
                                                carrier_frequency = SIMULATION_PARAM["carrier_frequency"],
                                                fft_size = SIMULATION_PARAM["fft_size"],
                                                num_guard_carriers = SIMULATION_PARAM["num_guard_carrier"],
                                                bits_per_symbol = SIMULATION_PARAM["num_bits_per_symbol"],
                                                bs_power = SIMULATION_PARAM["BS_power"],
                                                num_ue_antennas = SIMULATION_PARAM["num_ut_ant"],
                                                num_bs_antennas = SIMULATION_PARAM["num_bs_ant"],
                                                cdl_models = SIMULATION_PARAM["cdl_model"],
                                                delay_spread = SIMULATION_PARAM["delay_spread"],
                                                normalize_channel = SIMULATION_PARAM["normalize_channel"],
                                                channel_blocked_equalizer = SIMULATION_PARAM["channel_blocked_equalizer"],
                                                blocked_clusters_count = SIMULATION_PARAM["nb_clusters_blocked"],
                                                mask_clusters_blocked_per_link = SIMULATION_PARAM["mask_clusters_blocked_per_link"],
                                                distributed_mimo_blockage_mode = SIMULATION_PARAM["DMIMO_blockage_mode"],
                                                precoding = SIMULATION_PARAM["precoding"],
                                                graph_mode = SIMULATION_PARAM["graph_mode"]
                                                )  

#############################################################################################
# CALL THE MODEL with selected execution
############################################################################################# 
ber_plots = sionna.utils.PlotBER("BER Vs Eb/N0") # info_text
t_start = time.perf_counter()
ber_MC, bler_MC = ber_plots.simulate(model_distributed_MIMO, 
                    ebno_dbs=SIMULATION_PARAM["ebno_db"], 
                    batch_size=SIMULATION_PARAM["batch_size"],
                    graph_mode = SIMULATION_PARAM["graph_mode"], 
                    num_target_block_errors = None, 
                    num_target_bit_errors = SIMULATION_PARAM["num_target_bit_errors"],
                    target_ber  = SIMULATION_PARAM["target_ber"],
                    soft_estimates = True,
                    max_mc_iter = SIMULATION_PARAM["max_mc_iter"], # run 1000 Monte-Carlo simulations maximum (each with batch_size samples)
                    show_fig=False);
t_stop = time.perf_counter()
duration = (t_stop - t_start)
tf.print("duration: ",duration, "seconds")

#############################################################################################
# 
#
#                           THEORETICAL WORK
#
#
############################################################################################# 
# deduced parameters
ebno_base10 = np.power(10,np.array(SIMULATION_PARAM["ebno_db"])/10)
M = 2**SIMULATION_PARAM["num_bits_per_symbol"] 
# parameters for Peb calculus
constellation_symbol = 1+1j
cdl_instance = model_distributed_MIMO.cdl_channel_models[0]
cluster_number_per_link = cdl_instance._num_clusters
power_base10_normalized_per_cluster = cdl_instance._powers[0,0,0,:] # Shared powers for all BS as same channel type used
#############################################################################################
# EQUATION 6 OF PAPER
#############################################################################################
# Sum of powers for clusters where mask is non-zero
var_hB = [ np.sum(power_base10_normalized_per_cluster[mask != 0])
    for mask in SIMULATION_PARAM["mask_clusters_blocked_per_link"] ]
#############################################################################################
# EQUATION 5 OF PAPER
#############################################################################################
mean_per_BS = np.zeros([N_BS]) # Because only NLoS cannel considered in this article
#############################################################################################
# EQUATION 20 OF PAPER
#############################################################################################
var_hNB = np.ones(N_BS)
corr_hB_hNB = np.sqrt(var_hB/var_hNB) # no division by variance of unblocked link as normalisation in done => value is 1
#############################################################################################
# EQUATION 18 OF PAPER
#############################################################################################
symbol_real = np.real(constellation_symbol)
symbol_imag = np.imag(constellation_symbol)
# Compute noise power based on Eb/N0 and modulation order
N0_over_mod_x_squared = ebno_base10**-1 * (2 * (M - 1)) / (3 * (symbol_real**2 + symbol_imag**2) * np.log2(M))
weighted_var_link = var_hB * SIMULATION_PARAM["BS_power"]
# Final correlation calculation
correlation = constellation_symbol / np.abs(constellation_symbol) * np.sum(weighted_var_link) / np.sqrt( (np.sum(weighted_var_link) + N0_over_mod_x_squared) * np.sum(var_hNB * SIMULATION_PARAM["BS_power"]))
#############################################################################################
# EQUATION 17 OF PAPER
#############################################################################################
sigma_r = np.sqrt( np.abs(constellation_symbol)**2 *(np.sum(weighted_var_link) + N0_over_mod_x_squared) ) 
sigma_e = np.sqrt(np.sum(var_hNB * SIMULATION_PARAM["BS_power"])) 
#############################################################################################
# EQUATION 28 OF PAPER
#############################################################################################
# Python function for the marginal CDF of real or imaginary parts of the ratio of centered complex Gaussian variables
val_threshold=0
Peb_QPSK = useful_functions.compute_marginal_cdf(val_threshold, correlation, sigma_r, sigma_e, for_real_part=True)

###############################################################################
#       PLOT Monte-Carlo and Theoretical curves                                    #
###############################################################################
# Génération de la figure et définition des axes
fig, ax = plt.subplots(figsize=(6, 5))  # figsize=(10, 6)
ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=14)
ax.set_ylabel("BER", fontsize=14)
ax.set_yscale("log")
ax.grid(which="both")
ax.set_ylim([1e-3, 0.6])
colors=['#1f77b4', '#ff7f0e','#17becf', '#2ca02c', '#d62728', '#9467bd', '#8c564b','#7f7f7f', ]
ax.semilogy(SIMULATION_PARAM["ebno_db"], ber_MC,'*', markersize=8, color=colors[1],label="BER Monte-Carlo")
ax.semilogy(SIMULATION_PARAM["ebno_db"], Peb_QPSK, linewidth=2, linestyle='-', color=colors[1], label="Peb Theoretical") 
ax.legend( loc="best", frameon=True, title="Link Clusters Information", fontsize=10)
# add text box
dummy_handle = Line2D([], [], linestyle="", marker="")
handles, labels = ax.get_legend_handles_labels()
handles.append(dummy_handle)
labels.append(info_text)
ax.legend(handles, labels, loc="best", frameon=True, title="Legend & Link Clusters Information", fontsize=10)
plt.show()
