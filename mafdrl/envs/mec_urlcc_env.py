# mafdrl/envs/mec_urlcc_env.py
import numpy as np
from scipy.stats import norm  # for Q^{-1}(ε)

class MECURLLCEnv:
    """
    Multi-agent uplink MEC with URLLC (finite blocklength).
    One step ~ one scheduling epoch with new channels and task arrivals.

    State per user (aligned to paper):
      s_u(t) = [ q_u(t), h_u(t), f_u^loc(t), rho_u(t-1) ]
        - q_u(t): queue bits
        - h_u(t): CSI proxy = ||H_u||_F
        - f_u^loc(t): local CPU frequency used in current step (last chosen)
        - rho_u(t-1): previous offloading ratio

    Action per user (aligned to paper):
      a_u(t) = [ p_u(t), w_u(t), rho_u(t), f_u^loc(t) ]
        - p_u ∈ [0, p_max], scalar power
        - w_u ∈ R^{M_t} (proxy for complex unit-norm beam), normalized inside
        - rho_u ∈ [0,1], offload ratio
        - f_u^loc ∈ [0, f_loc_max], local CPU frequency

    Core equations implemented:
      - SINR:     Eq. (sinr)   γ_u
      - FBL rate: Eq. (fb_rate) R_u ≈ log2(1+γ_u) - sqrt(V(γ_u)/D)*Q^{-1}(ε)
      - E2E lat:  Eq. (e2e_latency) T_u^{E2E} = T_loc + T_tx + T_Q + T_cpu
      - Reward:   R_u - α p_u - β T_u^{E2E} - λ·1{T_u^{E2E} > T_deadline}
    """

    def __init__(
        self,
        U=3, Mt=2, Nr=4, bandwidth=1e6, noise_pow=1e-13,
        D=200, eps=1e-5, alpha_path=3.5, c0=1.0, shadow_sigma_db=6.0,
        p_max=0.5, f_loc_max=2.5e9, f_bs_max=20e9, f_bs=15e9,
        T_deadline=0.01,   # 10 ms
        alpha_energy=1e-3, beta_latency=1e-4, lambda_pen=0.01,
        seed=0
    ):
        # basic params
        self.rng = np.random.default_rng(seed)
        self.U, self.Mt, self.Nr = U, Mt, Nr
        self.B = bandwidth
        self.sigma2 = noise_pow
        self.D = D
        self.eps = eps
        self.alpha_path = alpha_path
        self.c0 = c0
        self.shadow_sigma = shadow_sigma_db
        self.p_max = p_max
        self.f_loc_max = f_loc_max
        self.f_bs_max = f_bs_max
        self.f_bs = min(f_bs, f_bs_max)
        self.T_deadline = T_deadline
        self.alpha_energy = alpha_energy  # α
        self.beta_latency = beta_latency  # β
        self.lambda_pen = lambda_pen      # λ

        # compute density (cycles/bit) & mean task size (bits)
        self.C_u = np.full(self.U, 10.0)
        self.L_mean = np.full(self.U, 2e5)

        # queues and memory of last decisions
        self.queue_bits = np.zeros(self.U, dtype=float)  # q_u
        self.prev_rho = np.zeros(self.U, dtype=float)    # rho_u(t-1)
        self.last_f_loc = np.zeros(self.U, dtype=float)  # f_u^loc(t)

        # user–BS distances (fixed geometry, 15–60m)
        self.d_u = self.rng.uniform(15.0, 60.0, size=self.U)

    # ---------- channel models ----------
    def _shadowing(self):
        s_db = self.rng.normal(0.0, self.shadow_sigma, size=self.U)
        return 10**(s_db / 10.0)

    def _small_fading(self):
        Re = self.rng.normal(size=(self.U, self.Nr, self.Mt))
        Im = self.rng.normal(size=(self.U, self.Nr, self.Mt))
        return (Re + 1j * Im) / np.sqrt(2.0)

    def _channels(self):
        chi = self._shadowing()
        ell = self.c0 * (self.d_u ** (-self.alpha_path)) * chi
        G = self._small_fading()
        H = np.zeros_like(G, dtype=complex)
        for u in range(self.U):
            H[u] = np.sqrt(ell[u]) * G[u]
        return H  # (U, Nr, Mt)

    # ---------- finite-blocklength URLLC rate ----------
    def _finite_blocklength_rate(self, sinr):
        """
        Eq. (fb_rate): R ≈ log2(1+γ) - sqrt(V(γ)/D) * Q^{-1}(ε),  V(γ) = 1 - (1+γ)^(-2).
        """
        sinr = np.asarray(sinr, dtype=float)
        V = 1.0 - 1.0 / (1.0 + sinr)**2
        qinv = float(norm.isf(self.eps))  # true Q^{-1}(ε)
        backoff = np.sqrt(np.maximum(V, 1e-12) / max(self.D, 1e-9)) * qinv
        R = np.log2(1.0 + np.maximum(sinr, 0.0)) - backoff
        return np.maximum(R, 0.0)

    def _mg1_queue_upper(self, lam_tot_bits_s, Ctot_cycles_per_bit):
        # T_Q <= (λ C^2) / (2 (f_bs - λ C))
        denom = self.f_bs - lam_tot_bits_s * Ctot_cycles_per_bit
        if denom <= 1e-9:
            return 1e3  # heavy delay if unstable
        return (lam_tot_bits_s * (Ctot_cycles_per_bit**2)) / (2.0 * denom)

    # ---------- interaction ----------
    def reset(self):
        self.queue_bits[:] = 0.0
        self.prev_rho[:] = 0.0
        self.last_f_loc[:] = 0.0
        # small random warmup arrivals
        self.queue_bits += self.rng.poisson(1.0, size=self.U) * self.L_mean * 0.2
        return self._obs()

    def _obs(self):
        """
        State per user: [ q_u, h_u, f_u^loc, prev_rho ],
        where h_u = ||H_u||_F (scalar CSI proxy to keep obs_dim=4).
        """
        H = self._channels()
        Hnorm = np.linalg.norm(H.reshape(self.U, -1), axis=1)  # ||H_u||_F
        obs = np.stack([
            self.queue_bits,
            Hnorm,
            self.last_f_loc,
            self.prev_rho
        ], axis=1).astype(np.float32)
        self._last_H = H
        return obs

    def step(self, actions):
        """
        actions: array (U, 2+Mt+1) ordered as:
          [ p, rho, w1...w_Mt, f_loc ]
        """
        if isinstance(actions, dict):
            A = np.stack([actions[u] for u in range(self.U)], axis=0)
        else:
            A = np.asarray(actions)

        if A.shape[1] < (2 + self.Mt + 1):
            raise ValueError(f"Expected action dim >= {2 + self.Mt + 1} (p, rho, w(Mt), f_loc) but got {A.shape[1]}")

        # unpack and clip
        p = np.clip(A[:, 0], 0.0, self.p_max)               # p_u
        rho = np.clip(A[:, 1], 0.0, 1.0)                    # rho_u
        w = A[:, 2:2+self.Mt]                               # w_u
        f_loc = np.clip(A[:, 2+self.Mt], 0.0, self.f_loc_max)  # f_u^loc

        # unit-norm beam (real proxy for complex)
        w = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)

        # channels from last obs
        H = self._last_H  # (U, Nr, Mt)

        # Linear combiner: matched filter (Eq. reception/combining piece)
        h_eff = np.zeros((self.U, self.Nr), dtype=complex)
        for u in range(self.U):
            h_eff[u] = (H[u] @ w[u][:, None]).squeeze(-1)
        g = h_eff / np.maximum(np.linalg.norm(h_eff, axis=1, keepdims=True), 1e-12)

        # Tx beam per user
        f = np.sqrt(p)[:, None] * w  # (U, Mt)

        # ---------- SINR per user (Eq. (sinr)) ----------
        sinr = np.zeros(self.U)
        for u in range(self.U):
            num = np.abs(np.vdot(g[u], (H[u] @ f[u][:, None]).squeeze(-1)))**2
            inter = 0.0
            for j in range(self.U):
                if j == u:
                    continue
                inter += np.abs(np.vdot(g[u], (H[j] @ f[j][:, None]).squeeze(-1)))**2
            den = inter + self.sigma2 * np.linalg.norm(g[u])**2
            sinr[u] = float(np.real(num) / max(den, 1e-20))

        # ---------- Finite-blocklength rate & throughput ----------
        R = self._finite_blocklength_rate(sinr)  # bits/s/Hz
        thr = self.B * R                         # bits/s

        # ---------- Arrivals and service ----------
        new_bits = self.rng.poisson(lam=1.0, size=self.U) * self.L_mean * 0.5
        self.queue_bits += new_bits

        L_take = np.minimum(self.queue_bits, self.L_mean)
        L_off = rho * L_take
        L_loc = (1.0 - rho) * L_take

        # ---------- Latencies (Eq. (e2e_latency)) ----------
        # local compute
        T_loc = (self.C_u * L_loc) / np.maximum(f_loc, 1e-9)

        # uplink transmission (safe)
        thr_safe = np.maximum(thr, 1e-3)                 # avoid div-by-zero
        T_tx = np.where(L_off > 1e-12, L_off / thr_safe, 0.0)
        T_tx = np.minimum(T_tx, 1.0)                     # cap at 1s

        # queueing upper bound (M/G/1)
        lam_tot = float(np.sum(new_bits))
        C_tot = float(np.mean(self.C_u))
        T_Q = self._mg1_queue_upper(lam_tot, C_tot) * np.ones(self.U)

        # BS compute
        T_cpu = (self.C_u * L_off) / np.maximum(self.f_bs, 1e-9)

        # end-to-end
        T_e2e = T_loc + T_tx + T_Q + T_cpu

        # ---------- Reward (aligned to paper) ----------
        # R_u - α p_u - β T_u^{E2E} - λ · 1{T_u^{E2E} > T_deadline}
        reward = R - self.alpha_energy * p - self.beta_latency * T_e2e
        indicator_viol = (T_e2e > self.T_deadline).astype(float)
        reward -= self.lambda_pen * indicator_viol

        # ---------- State bookkeeping ----------
        self.queue_bits -= L_take
        self.queue_bits = np.maximum(self.queue_bits, 0.0)
        self.prev_rho = rho.copy()
        self.last_f_loc = f_loc.copy()

        # next observation
        obs = self._obs()

        # simple episodic termination when too many deadline violations
        done = (indicator_viol.mean() > 0.1).item()
        term_flags = np.array([done] * self.U)

        info = {
            "sinr": sinr, "R": R, "thr": thr,
            "T_e2e": T_e2e, "T_loc": T_loc, "T_tx": T_tx,
            "T_Q": T_Q, "T_cpu": T_cpu,
        }
        return obs, reward.astype(np.float32), term_flags, False, info
