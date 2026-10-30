#!/usr/bin/env python3
"""
Core Module:
1. 19-Channel Spatial Harmonizer & Real Electrophysiological Feature Extractor
   (Periodic Residual Bandpowers + Aperiodic Exponent chi + Real Hilbert PLI).
2. True Orthogonal Minimal Spanning Trees (OMST) with Global Cost Efficiency (GCE)
   Optimization (Dimitriadis et al., 2017, Front. Neuroinform.), alongside explicit
   MST + Proportional Thresholding.
3. Fold-Nested Healthy/Baseline Normative Deviation Z-Score Calibration
   (Rutherford et al., 2022, Nature Protocols) and Pure PyTorch GCN.
"""

import numpy as np
import scipy.signal as signal
import scipy.sparse.csgraph as csgraph
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS_19 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
    'Fz', 'Cz', 'Pz'
]

# Legacy 10-20 to Modern 10-20 mapping (for EDF files using T3/T4/T5/T6)
LEGACY_1020_MAP = {
    'T3': 'T7',
    'T4': 'T8',
    'T5': 'P7',
    'T6': 'P8'
}


class TopologicalGraphBuilder:
    """
    Constructs Functional Connectivity Brain Graphs using either:
      1. mode='omst': True Orthogonal Minimal Spanning Trees (OMST) with automatic
         Global Cost Efficiency (GCE = E_global - Cost) optimization (Dimitriadis et al., 2017).
      2. mode='mst_proportional': Single MST backbone + top-k proportional edges
         up to fixed target_density (van Wijk et al., 2010).
    """
    def __init__(self, mode: str = 'omst', target_density: float = 0.20, max_msts: int = 8):
        if mode not in ('omst', 'mst_proportional'):
            raise ValueError(f"Unsupported mode: {mode}. Choose 'omst' or 'mst_proportional'.")
        self.mode = mode
        self.target_density = target_density
        self.max_msts = max_msts

    @staticmethod
    def _global_efficiency(dist_mat: np.ndarray) -> float:
        """Computes global efficiency E_global from all-pairs shortest path distances."""
        n = dist_mat.shape[0]
        with np.errstate(divide='ignore'):
            inv_d = 1.0 / dist_mat
        np.fill_diagonal(inv_d, 0.0)
        inv_d[np.isinf(inv_d) | np.isnan(inv_d)] = 0.0
        return float(np.sum(inv_d) / (n * (n - 1)))

    def build_omst_graph(self, fc_matrix: np.ndarray):
        """
        True iterative OMST algorithm (Dimitriadis et al., 2017):
        Iteratively extracts edge-disjoint (orthogonal) MSTs and selects the optimal
        number of trees m* that maximizes Global Cost Efficiency:
            GCE(m) = E_global(G_m) / E_global(G_full) - Cost(G_m)
        Returns: (binary_adj, weighted_adj, optimal_m, max_gce)
        """
        n = fc_matrix.shape[0]
        w_full = np.clip(fc_matrix.copy(), 1e-6, 1.0)
        np.fill_diagonal(w_full, 0.0)
        total_weight = np.sum( np.triu(w_full, k=1) )
        if total_weight <= 0:
            raise ValueError("Functional connectivity matrix has zero total weight.")

        # Shortest-path distance matrix on the full weighted graph (stronger coupling = shorter path)
        full_len_mat = 1.0 / (w_full + 1e-6)
        np.fill_diagonal(full_len_mat, 0.0)
        full_sp = csgraph.floyd_warshall(full_len_mat, directed=False)
        e_glob_full = max(self._global_efficiency(full_sp), 1e-8)

        # Iterative extraction of orthogonal MSTs
        # To maximize tree weight with minimum_spanning_tree, we invert weights: cost = 1.0 - w
        # Add an epsilon (+ 1e-4) so that all available valid edges have cost > 0.
        # This prevents SciPy's sparse minimum_spanning_tree from treating 0.0-cost edges (w=1.0) as empty/absent.
        inv_cost_available = (1.0 - w_full) + 1e-4
        np.fill_diagonal(inv_cost_available, 0.0)
        available_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(available_mask, False)

        cumulative_adj = np.zeros((n, n), dtype=np.float32)
        best_adj = None
        best_m = 1
        best_gce = -np.inf

        max_possible_trees = min(self.max_msts, n // 2)
        for m in range(1, max_possible_trees + 1):
            # Masked distance matrix for current MST extraction
            step_cost = np.where(available_mask, inv_cost_available, 0.0)
            mst_csr = csgraph.minimum_spanning_tree(step_cost)
            mst_arr = mst_csr.toarray()
            tree_edges = (mst_arr > 0) | (mst_arr.T > 0)

            if np.sum(tree_edges) == 0:
                break

            # Add m-th orthogonal MST edges to cumulative network
            cumulative_adj[tree_edges] = 1.0
            # Remove used edges from pool for strict orthogonality
            available_mask[tree_edges] = False

            # Compute weighted subgraph and its Global Cost Efficiency (GCE)
            sub_w = w_full * cumulative_adj
            cost_m = np.sum(np.triu(sub_w, k=1)) / total_weight

            sub_len = np.where(cumulative_adj > 0, 1.0 / (sub_w + 1e-6), np.inf)
            np.fill_diagonal(sub_len, 0.0)
            sub_sp = csgraph.floyd_warshall(sub_len, directed=False)
            e_glob_m = self._global_efficiency(sub_sp) / e_glob_full

            gce_m = e_glob_m - cost_m
            if gce_m > best_gce:
                best_gce = gce_m
                best_m = m
                best_adj = cumulative_adj.copy()

        if best_adj is None:
            best_adj = cumulative_adj

        weighted_adj = best_adj * w_full
        return best_adj, weighted_adj, best_m, float(best_gce)

    def build_mst_proportional_graph(self, fc_matrix: np.ndarray) -> np.ndarray:
        """Single MST backbone + top-k proportional edges up to target_density."""
        n = fc_matrix.shape[0]
        dist_matrix = 1.0 - np.clip(fc_matrix, 0.0, 1.0)
        np.fill_diagonal(dist_matrix, 0.0)

        mst = csgraph.minimum_spanning_tree(dist_matrix).toarray()
        mst_adj = ((mst + mst.T) > 0).astype(np.float32)

        n_possible_edges = n * (n - 1) // 2
        k_target_edges = int(np.round(self.target_density * n_possible_edges))
        k_needed = max(0, k_target_edges - (n - 1))

        if k_needed > 0:
            upper_tri = np.triu(fc_matrix, k=1).copy()
            upper_tri[mst_adj > 0] = -1.0
            flat_indices = np.argsort(upper_tri.ravel())[::-1]
            for idx in flat_indices[:k_needed]:
                i, j = np.unravel_index(idx, upper_tri.shape)
                if upper_tri[i, j] > 0:
                    mst_adj[i, j] = 1.0
                    mst_adj[j, i] = 1.0
        return mst_adj

    def build_graph(self, fc_matrix: np.ndarray) -> np.ndarray:
        if self.mode == 'omst':
            binary_adj, _, _, _ = self.build_omst_graph(fc_matrix)
            return binary_adj
        return self.build_mst_proportional_graph(fc_matrix)


def extract_real_19ch_features_and_pli(epochs_data: np.ndarray, sfreq: float):
    """
    Computes 100% REAL electrophysiological node features and Phase Lag Index (PLI)
    from a 3D EEG array of shape [n_epochs, 19, n_times].
    Returns:
      - x_mat: [19, 4] float32 array per channel:
          col 0: Periodic (flattened) Theta power (4-8 Hz)
          col 1: Periodic (flattened) Delta power (1-4 Hz)
          col 2: Periodic (flattened) Alpha power (8-13 Hz)
          col 3: Aperiodic 1/f exponent chi (fit across 1-40 Hz excluding 7-14 Hz alpha bump)
      - pli_matrix: [19, 19] float32 Phase Lag Index connectivity matrix in 4-13 Hz
    """
    n_epochs, n_ch, n_times = epochs_data.shape
    assert n_ch == 19, f"Expected 19 channels, got {n_ch}"

    # 1. Compute Welch PSD averaged across epochs for each of the 19 channels
    nperseg = min(int(sfreq * 2.0), n_times)
    freqs, psd = signal.welch(epochs_data, fs=sfreq, nperseg=nperseg, axis=-1)
    mean_psd = np.mean(psd, axis=0)  # [19, n_freqs]

    valid_mask = (freqs >= 1.0) & (freqs <= 40.0)
    f_valid = freqs[valid_mask]
    log_f = np.log10(f_valid)

    # Fit aperiodic 1/f background excluding canonical alpha peak [7.0, 14.0] Hz
    aperiodic_fit_mask = (f_valid < 7.0) | (f_valid > 14.0)
    log_f_fit = log_f[aperiodic_fit_mask]

    theta_mask = (f_valid >= 4.0) & (f_valid < 8.0)
    delta_mask = (f_valid >= 1.0) & (f_valid < 4.0)
    alpha_mask = (f_valid >= 8.0) & (f_valid <= 13.0)

    x_mat = np.zeros((19, 4), dtype=np.float32)
    for c in range(19):
        log_p = np.log10(np.maximum(mean_psd[c, valid_mask], 1e-15))
        # Linear regression in log-log space: log10(P) = offset - chi * log10(f)
        slope, intercept = np.polyfit(log_f_fit, log_p[aperiodic_fit_mask], 1)
        chi = -slope  # Positive aperiodic exponent
        aperiodic_line = intercept + slope * log_f
        flattened_spec = log_p - aperiodic_line

        x_mat[c, 0] = float(np.mean(flattened_spec[theta_mask]))
        x_mat[c, 1] = float(np.mean(flattened_spec[delta_mask]))
        x_mat[c, 2] = float(np.mean(flattened_spec[alpha_mask]))
        x_mat[c, 3] = float(chi)

    # 2. Compute Real Phase Lag Index (PLI) in 4-13 Hz across epochs (use up to 30 clean epochs for speed)
    max_ep = min(n_epochs, 30)
    sub_epochs = epochs_data[:max_ep]
    nyq = sfreq / 2.0
    b, a = signal.butter(4, [4.0 / nyq, 13.0 / nyq], btype='band')
    filtered = signal.filtfilt(b, a, sub_epochs, axis=-1)
    analytic = signal.hilbert(filtered, axis=-1)
    phases = np.angle(analytic)  # [max_ep, 19, n_times]

    # Concatenate epochs along time axis -> [19, max_ep * n_times]
    phases_concat = phases.transpose(1, 0, 2).reshape(19, -1)
    pli_matrix = np.zeros((19, 19), dtype=np.float32)
    for i in range(19):
        for j in range(i + 1, 19):
            phase_diff = phases_concat[i] - phases_concat[j]
            pli_val = np.abs(np.mean(np.sign(np.sin(phase_diff))))
            pli_matrix[i, j] = pli_val
            pli_matrix[j, i] = pli_val
    np.fill_diagonal(pli_matrix, 1.0)

    return x_mat, pli_matrix


def compute_fold_normative_z(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor):
    """
    Fold-Nested Normative Modeling Calibration (Rutherford et al., 2022, Nature Protocols).
    Estimates vertex-wise normative reference mean and standard deviation EXCLUSIVELY
    from the reference control/baseline group (y == 0) inside the training fold.
    Transforms both training and evaluation node features into Normative Deviation Z-scores:
        Z_{i, c, f} = (X_{i, c, f} - mu_{y=0, c, f}) / sigma_{y=0, c, f}
    """
    ref_mask = (train_y == 0)
    if torch.sum(ref_mask) < 2:
        raise ValueError("Training fold must contain at least 2 reference (y=0) samples for normative calibration.")
    ref_x = train_x[ref_mask]  # [N_ref, 19, 4]
    mu_ref = torch.mean(ref_x, dim=0, keepdim=True)         # [1, 19, 4]
    sigma_ref = torch.std(ref_x, dim=0, keepdim=True) + 1e-6 # [1, 19, 4]

    train_z = (train_x - mu_ref) / sigma_ref
    eval_z = (eval_x - mu_ref) / sigma_ref
    return train_z, eval_z, mu_ref, sigma_ref


class PurePyTorchGCNLayer(nn.Module):
    """
    Spectral Graph Convolution (Kipf & Welling, ICLR 2017) implemented in pure PyTorch.
    Supports batched graph adjacency matrices [B, N, N] and node features [B, N, Fin].
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, n_nodes, _ = x.shape
        identity = torch.eye(n_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        a_hat = adj + identity

        degrees = torch.sum(a_hat, dim=2)
        deg_inv_sqrt = torch.pow(degrees, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        d_mat = torch.diag_embed(deg_inv_sqrt)

        norm_adj = torch.bmm(torch.bmm(d_mat, a_hat), d_mat)
        xw = torch.matmul(x, self.weight)
        out = torch.bmm(norm_adj, xw) + self.bias
        return out


class NormativeGCN(nn.Module):
    """
    2-Layer Graph Convolutional Network operating on Fold-Nested Normative Deviation Z-Maps.
    Combines node-level normative deviation scores Z_{c,f} with OMST topological message passing.
    """
    def __init__(self, in_node_features: int = 4, hidden_dim: int = 32, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.gcn1 = PurePyTorchGCNLayer(in_node_features, hidden_dim)
        self.gcn2 = PurePyTorchGCNLayer(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z_norm: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.gcn1(z_norm, adj))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.relu(self.gcn2(h1, adj))

        h_mean = torch.mean(h2, dim=1)
        h_max, _ = torch.max(h2, dim=1)
        graph_embed = torch.cat([h_mean, h_max], dim=-1)
        return self.classifier(graph_embed)
