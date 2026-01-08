# # # MEC–URLLC environment
# # # mafdrl/envs/mec_urlcc_env.py
# # import numpy as np
# # from scipy.stats import norm  # for Q^{-1}(ε)


# # class MECURLLCEnv:
# #     """
# #     Multi-agent uplink MEC with URLLC (finite blocklength).
# #     One step ~ one scheduling epoch with new channels and task arrivals.

# #     State per user:
# #       s_u(t) = [ q_u(t), h_u(t), f_u^loc(t), rho_u(t-1) ]
# #         - q_u(t): queue bits
# #         - h_u(t): CSI proxy = ||H_u||_F
# #         - f_u^loc(t): local CPU frequency used in current step (last chosen)
# #         - rho_u(t-1): previous offloading ratio

# #     Action per user:
# #       a_u(t) = [ p_u(t), w_u(t), rho_u(t), f_u^loc(t) ]
# #         - p_u ∈ [0, p_max], scalar power
# #         - w_u ∈ R^{M_t} (proxy beam), we still accept it for dimension
# #         - rho_u ∈ [0,1], offload ratio
# #         - f_u^loc ∈ [0, f_loc_max], local CPU frequency
# #     """

# #     def __init__(
# #         self,
# #         U=3,
# #         Mt=2,
# #         Nr=4,
# #         bandwidth=1e6,
# #         noise_pow=1e-13,
# #         D=200,
# #         eps=1e-5,
# #         alpha_path=3.5,
# #         c0=1.0,
# #         shadow_sigma_db=6.0,
# #         p_max=0.5,
# #         f_loc_max=2.5e9,
# #         f_bs_max=20e9,
# #         f_bs=15e9,
# #         T_deadline=0.01,          # 10 ms
# #         alpha_energy=5e-4,        # α (smaller energy penalty)
# #         beta_latency=1e-2,        # β (acts on normalized latency)
# #         lambda_pen=2e-2,          # λ (deadline violation penalty)
# #         queue_penalty=1e-3,       # η (queue/backlog penalty)
# #         seed=0,
# #     ):
# #         # basic params
# #         self.rng = np.random.default_rng(seed)
# #         self.U, self.Mt, self.Nr = U, Mt, Nr
# #         self.B = bandwidth
# #         self.sigma2 = noise_pow
# #         self.D = D
# #         self.eps = eps
# #         self.alpha_path = alpha_path
# #         self.c0 = c0
# #         self.shadow_sigma = shadow_sigma_db
# #         self.p_max = p_max
# #         self.f_loc_max = f_loc_max
# #         self.f_bs_max = f_bs_max
# #         self.f_bs = min(f_bs, f_bs_max)
# #         self.T_deadline = T_deadline
# #         self.alpha_energy = alpha_energy
# #         self.beta_latency = beta_latency
# #         self.lambda_pen = lambda_pen
# #         self.queue_penalty = queue_penalty

# #         # compute density (cycles/bit) & mean task size (bits)
# #         self.C_u = np.full(self.U, 10.0)
# #         self.L_mean = np.full(self.U, 2e5)

# #         # queues and memory of last decisions
# #         self.queue_bits = np.zeros(self.U, dtype=float)  # q_u
# #         self.prev_rho = np.zeros(self.U, dtype=float)    # rho_u(t-1)
# #         self.last_f_loc = np.zeros(self.U, dtype=float)  # f_u^loc(t)

# #         # user–BS distances (fixed geometry, 15–60m)
# #         self.d_u = self.rng.uniform(15.0, 60.0, size=self.U)

# #     # ---------- channel models ----------
# #     def _shadowing(self):
# #         s_db = self.rng.normal(0.0, self.shadow_sigma, size=self.U)
# #         return 10.0 ** (s_db / 10.0)

# #     def _small_fading(self):
# #         Re = self.rng.normal(size=(self.U, self.Nr, self.Mt))
# #         Im = self.rng.normal(size=(self.U, self.Nr, self.Mt))
# #         return (Re + 1j * Im) / np.sqrt(2.0)

# #     def _channels(self):
# #         chi = self._shadowing()
# #         ell = self.c0 * (self.d_u ** (-self.alpha_path)) * chi
# #         G = self._small_fading()
# #         H = np.zeros_like(G, dtype=complex)
# #         for u in range(self.U):
# #             H[u] = np.sqrt(ell[u]) * G[u]
# #         return H  # (U, Nr, Mt)

# #     # ---------- finite-blocklength URLLC rate ----------
# #     def _finite_blocklength_rate(self, sinr):
# #         """
# #         R ≈ log2(1+γ) - sqrt(V(γ)/D) * Q^{-1}(ε),
# #         with V(γ) = 1 - (1+γ)^(-2).
# #         """
# #         sinr = np.asarray(sinr, dtype=float)
# #         sinr = np.maximum(sinr, 0.0)

# #         V = 1.0 - 1.0 / (1.0 + sinr) ** 2
# #         qinv = float(norm.isf(self.eps))  # Q^{-1}(ε)
# #         backoff = np.sqrt(np.maximum(V, 1e-12) / max(self.D, 1e-9)) * qinv
# #         R = np.log2(1.0 + sinr) - backoff
# #         return np.maximum(R, 0.0)

# #     def _mg1_queue_upper(self, lam_tot_bits_s, Ctot_cycles_per_bit):
# #         # T_Q <= (λ C^2) / (2 (f_bs - λ C))
# #         denom = self.f_bs - lam_tot_bits_s * Ctot_cycles_per_bit
# #         if denom <= 1e-9:
# #             # heavy delay if unstable, but we will clip later
# #             return 1.0
# #         return (lam_tot_bits_s * (Ctot_cycles_per_bit ** 2)) / (2.0 * denom)

# #     # ---------- interaction ----------
# #     def reset(self):
# #         self.queue_bits[:] = 0.0
# #         self.prev_rho[:] = 0.0
# #         self.last_f_loc[:] = 0.0
# #         # small random warmup arrivals (reduced so queues don't explode)
# #         self.queue_bits += (
# #             self.rng.poisson(1.0, size=self.U) * self.L_mean * 0.05
# #         )
# #         return self._obs()

# #     def _obs(self):
# #         """
# #         State per user: [ q_u, h_u, f_u^loc, prev_rho ],
# #         where h_u = ||H_u||_F (scalar CSI proxy to keep obs_dim=4).
# #         """
# #         H = self._channels()
# #         Hnorm = np.linalg.norm(H.reshape(self.U, -1), axis=1)  # ||H_u||_F
# #         obs = np.stack(
# #             [
# #                 self.queue_bits,
# #                 Hnorm,
# #                 self.last_f_loc,
# #                 self.prev_rho,
# #             ],
# #             axis=1,
# #         ).astype(np.float32)
# #         self._last_H = H
# #         self._last_Hnorm2 = Hnorm ** 2  # used in SINR
# #         return obs

# #     def step(self, actions):
# #         """
# #         actions: array (U, 2+Mt+1) ordered as:
# #           [ p, rho, w1...w_Mt, f_loc ]
# #         """
# #         if isinstance(actions, dict):
# #             A = np.stack([actions[u] for u in range(self.U)], axis=0)
# #         else:
# #             A = np.asarray(actions, dtype=float)

# #         if A.shape[1] < (2 + self.Mt + 1):
# #             raise ValueError(
# #                 f"Expected action dim >= {2 + self.Mt + 1} "
# #                 f"(p, rho, w(Mt), f_loc) but got {A.shape[1]}"
# #             )

# #         # unpack and clip
# #         p = np.clip(A[:, 0], 0.0, self.p_max)  # p_u
# #         rho = np.clip(A[:, 1], 0.0, 1.0)       # rho_u
# #         # w is kept only for dimensional consistency (not used in SINR now)
# #         w = A[:, 2: 2 + self.Mt]
# #         f_loc = np.clip(
# #             A[:, 2 + self.Mt], 0.0, self.f_loc_max
# #         )  # f_u^loc

# #         # avoid fully zero beam (for completeness)
# #         w_norm = np.linalg.norm(w, axis=1, keepdims=True)
# #         w = w / np.maximum(w_norm, 1e-12)

# #         # channels from last obs
# #         H = self._last_H                     # (U, Nr, Mt)
# #         gain = self._last_Hnorm2.copy()      # ||H_u||_F^2, shape (U,)

# #         # ---------- SINR per user (stable scalar model) ----------
# #         sinr = np.zeros(self.U, dtype=float)
# #         total_interf = float(np.dot(p, gain))  # Σ_j p_j ||H_j||^2

# #         for u in range(self.U):
# #             signal = p[u] * gain[u]
# #             interference = total_interf - signal
# #             den = interference + self.sigma2
# #             den = max(den, 1e-12)
# #             sinr[u] = signal / den

# #         sinr = np.maximum(sinr, 0.0)

# #         # ---------- Finite-blocklength rate & throughput ----------
# #         R = self._finite_blocklength_rate(sinr)  # bits/s/Hz
# #         thr = self.B * R                         # bits/s

# #         # ---------- Arrivals and service ----------
# #         # slightly smaller arrivals so queues remain stable
# #         new_bits = (
# #             self.rng.poisson(lam=1.0, size=self.U) * self.L_mean * 0.1
# #         )
# #         self.queue_bits += new_bits

# #         L_take = np.minimum(self.queue_bits, self.L_mean)
# #         L_off = rho * L_take
# #         L_loc = (1.0 - rho) * L_take

# #         # ---------- Latencies ----------
# #         # local compute
# #         T_loc = (self.C_u * L_loc) / np.maximum(f_loc, 1e-9)

# #         # uplink transmission
# #         thr_safe = np.maximum(thr, 1e-6)  # avoid div-by-zero
# #         T_tx = np.where(L_off > 1e-12, L_off / thr_safe, 0.0)
# #         # cap TX delay to a few deadlines
# #         T_tx = np.minimum(T_tx, 5.0 * self.T_deadline)

# #         # queueing upper bound (M/G/1), clip to avoid huge values
# #         lam_tot = float(np.sum(new_bits))
# #         C_tot = float(np.mean(self.C_u))
# #         T_Q_scalar = self._mg1_queue_upper(lam_tot, C_tot)
# #         T_Q_scalar = min(T_Q_scalar, 5.0 * self.T_deadline)
# #         T_Q = T_Q_scalar * np.ones(self.U)

# #         # BS compute
# #         T_cpu = (self.C_u * L_off) / np.maximum(self.f_bs, 1e-9)

# #         # end-to-end
# #         T_e2e = T_loc + T_tx + T_Q + T_cpu

# #         # ---------- Reward ----------
# #         # Normalize latency w.r.t. deadline and clip
# #         T_norm = T_e2e / max(self.T_deadline, 1e-6)
# #         T_norm = np.minimum(T_norm, 10.0)

# #         # Queue backlog penalty
# #         q_norm = self.queue_bits / (self.L_mean * 3.0)
# #         q_norm = np.minimum(q_norm, 5.0)

# #         # Base reward:
# #         #   R - α p - β T_norm - η q_norm - λ 1{T_u^{E2E} > T_deadline}
# #         reward = (
# #             R
# #             - self.alpha_energy * p
# #             - self.beta_latency * T_norm
# #             - self.queue_penalty * q_norm
# #         )

# #         # Strong penalty when URLLC deadline is violated
# #         indicator_viol = (T_e2e > self.T_deadline).astype(float)
# #         reward -= self.lambda_pen * indicator_viol

# #         # Clip reward to a reasonable range for critic stability
# #         reward = np.clip(reward, -10.0, 5.0)

# #         # ---------- State bookkeeping ----------
# #         self.queue_bits -= L_take
# #         self.queue_bits = np.maximum(self.queue_bits, 0.0)
# #         self.prev_rho = rho.copy()
# #         self.last_f_loc = f_loc.copy()

# #         # next observation
# #         obs = self._obs()

# #         # simple episodic termination when too many deadline violations
# #         done = (indicator_viol.mean() > 0.1).item()
# #         term_flags = np.array([done] * self.U)

# #         info = {
# #             "sinr": sinr,
# #             "R": R,
# #             "thr": thr,
# #             "T_e2e": T_e2e,
# #             "T_loc": T_loc,
# #             "T_tx": T_tx,
# #             "T_Q": T_Q,
# #             "T_cpu": T_cpu,
# #             "rho": rho,
# #         }
# #         return obs, reward.astype(np.float32), term_flags, False, info

# # MEC–URLLC environment
# # mafdrl/envs/mec_urlcc_env.py

# import numpy as np
# from scipy.stats import norm  # for Q^{-1}(ε)


# class MECURLLCEnv:
#     """
#     Multi-agent uplink MEC with URLLC (finite blocklength).
#     One step ~ one scheduling epoch with new channels and task arrivals.

#     State per user:
#       s_u(t) = [ q_u(t), h_u(t), f_u^loc(t), rho_u(t-1) ]
#         - q_u(t): queue bits
#         - h_u(t): CSI proxy = ||H_u||_F
#         - f_u^loc(t): local CPU frequency used in current step (last chosen)
#         - rho_u(t-1): previous offloading ratio

#     Action per user:
#       a_u(t) = [ p_u(t), rho_u(t), w_u(t), f_u^loc(t) ]
#         - p_u ∈ [0, p_max]
#         - rho_u ∈ [0, 1]
#         - w_u ∈ R^{M_t} (proxy for beam), normalized to unit norm inside
#         - f_u^loc ∈ [0, f_loc_max]

#     Implemented:
#       - SINR (stable MRC-style)
#       - Finite blocklength rate
#       - E2E latency: T_loc + T_tx + T_Q + T_cpu
#       - Reward: R - α p - β (T_e2e/T_deadline) - η queue_norm - λ deadline_violation
#     """

#     def __init__(
#         self,
#         U=3,
#         Mt=2,
#         Nr=4,
#         bandwidth=1e6,
#         noise_pow=1e-13,
#         D=200,
#         eps=1e-5,
#         alpha_path=3.5,
#         c0=1.0,
#         shadow_sigma_db=6.0,
#         p_max=0.5,
#         f_loc_max=2.5e9,
#         f_bs_max=20e9,
#         f_bs=15e9,
#         T_deadline=0.01,          # 10 ms
#         alpha_energy=1e-3,        # α
#         beta_latency=5e-2,        # β
#         lambda_pen=5e-2,          # λ
#         queue_penalty=2e-2,       # η
#         seed=0,
#     ):
#         self.rng = np.random.default_rng(seed)
#         self.U, self.Mt, self.Nr = U, Mt, Nr
#         self.B = bandwidth
#         self.sigma2 = float(noise_pow)
#         self.D = int(D)
#         self.eps = float(eps)

#         self.alpha_path = float(alpha_path)
#         self.c0 = float(c0)
#         self.shadow_sigma = float(shadow_sigma_db)

#         self.p_max = float(p_max)
#         self.f_loc_max = float(f_loc_max)

#         self.f_bs_max = float(f_bs_max)
#         self.f_bs = float(min(f_bs, f_bs_max))

#         self.T_deadline = float(T_deadline)
#         self.alpha_energy = float(alpha_energy)
#         self.beta_latency = float(beta_latency)
#         self.lambda_pen = float(lambda_pen)
#         self.queue_penalty = float(queue_penalty)

#         # cycles/bit and mean task size (bits)
#         self.C_u = np.full(self.U, 10.0, dtype=float)
#         self.L_mean = np.full(self.U, 2e5, dtype=float)

#         # queues + last actions
#         self.queue_bits = np.zeros(self.U, dtype=float)
#         self.prev_rho = np.zeros(self.U, dtype=float)
#         self.last_f_loc = np.zeros(self.U, dtype=float)

#         # fixed distances (meters)
#         self.d_u = self.rng.uniform(15.0, 60.0, size=self.U)

#         # cache
#         self._last_H = None

#     # ---------- channel models ----------
#     def _shadowing(self):
#         s_db = self.rng.normal(0.0, self.shadow_sigma, size=self.U)
#         return 10 ** (s_db / 10.0)

#     def _small_fading(self):
#         Re = self.rng.normal(size=(self.U, self.Nr, self.Mt))
#         Im = self.rng.normal(size=(self.U, self.Nr, self.Mt))
#         return (Re + 1j * Im) / np.sqrt(2.0)

#     def _channels(self):
#         chi = self._shadowing()
#         ell = self.c0 * (self.d_u ** (-self.alpha_path)) * chi
#         G = self._small_fading()
#         H = np.zeros_like(G, dtype=complex)
#         for u in range(self.U):
#             H[u] = np.sqrt(ell[u]) * G[u]
#         return H  # (U, Nr, Mt)

#     # ---------- finite-blocklength URLLC rate ----------
#     def _finite_blocklength_rate(self, sinr_lin):
#         """
#         R ≈ log2(1+γ) - sqrt(V(γ)/D)*Q^{-1}(ε),
#         V(γ) = 1 - (1+γ)^(-2)
#         """
#         sinr = np.asarray(sinr_lin, dtype=np.float64)
#         sinr = np.maximum(sinr, 0.0)

#         V = 1.0 - 1.0 / (1.0 + sinr) ** 2
#         qinv = float(norm.isf(self.eps))  # Q^{-1}(ε)

#         backoff = np.sqrt(np.maximum(V, 1e-12) / max(self.D, 1)) * qinv
#         R = np.log2(1.0 + sinr) - backoff
#         return np.maximum(R, 0.0)

#     def _mg1_queue_upper(self, lam_tot_bits_s, Ctot_cycles_per_bit):
#         # T_Q <= (λ C^2) / (2 (f_bs - λ C))
#         denom = self.f_bs - lam_tot_bits_s * Ctot_cycles_per_bit
#         if denom <= 1e-9:
#             return 1e3
#         return (lam_tot_bits_s * (Ctot_cycles_per_bit ** 2)) / (2.0 * denom)

#     # ---------- interaction ----------
#     def reset(self):
#         self.queue_bits[:] = 0.0
#         self.prev_rho[:] = 0.0
#         self.last_f_loc[:] = 0.0

#         # warmup
#         self.queue_bits += self.rng.poisson(1.0, size=self.U) * self.L_mean * 0.2
#         return self._obs()

#     def _obs(self):
#         H = self._channels()
#         Hnorm = np.linalg.norm(H.reshape(self.U, -1), axis=1)
#         obs = np.stack(
#             [self.queue_bits, Hnorm, self.last_f_loc, self.prev_rho],
#             axis=1,
#         ).astype(np.float32)
#         self._last_H = H
#         return obs

#     def step(self, actions):
#         """
#         actions: (U, 2+Mt+1) ordered as:
#           [ p, rho, w1..w_Mt, f_loc ]
#         """
#         if isinstance(actions, dict):
#             A = np.stack([actions[u] for u in range(self.U)], axis=0)
#         else:
#             A = np.asarray(actions)

#         req_dim = 2 + self.Mt + 1
#         if A.ndim != 2 or A.shape[0] != self.U or A.shape[1] < req_dim:
#             raise ValueError(
#                 f"Expected actions shape (U,{req_dim}) with U={self.U}, got {A.shape}"
#             )

#         # unpack
#         p = np.clip(A[:, 0], 0.0, self.p_max)
#         rho = np.clip(A[:, 1], 0.0, 1.0)
#         w = A[:, 2:2 + self.Mt]
#         f_loc = np.clip(A[:, 2 + self.Mt], 0.0, self.f_loc_max)

#         # normalize beam (real proxy)
#         w_norm = np.linalg.norm(w, axis=1, keepdims=True)
#         w = w / np.maximum(w_norm, 1e-12)

#         # channels
#         H = self._last_H
#         if H is None:
#             H = self._channels()
#             self._last_H = H

#         # ======== Stable SINR computation (MRC-style) ========
#         # f_u = sqrt(p_u) * w_u
#         f_tx = (np.sqrt(p)[:, None]) * w  # (U, Mt)

#         # compute received vectors y_{j->BS} = H_j f_j  (U, Nr)
#         y = np.zeros((self.U, self.Nr), dtype=complex)
#         for j in range(self.U):
#             y[j] = (H[j] @ f_tx[j][:, None]).squeeze(-1)

#         # combiner g_u = y_u / ||y_u||  (MRC on effective desired signal direction)
#         g = np.zeros((self.U, self.Nr), dtype=complex)
#         for u in range(self.U):
#             denom = np.linalg.norm(y[u])
#             if denom < 1e-12:
#                 # fallback: use channel direction from H_u w_u (still valid even if p≈0)
#                 h_eff = (H[u] @ w[u][:, None]).squeeze(-1)
#                 denom2 = np.linalg.norm(h_eff)
#                 g[u] = h_eff / max(denom2, 1e-12)
#             else:
#                 g[u] = y[u] / denom

#         # SINR_u = |g_u^H y_u|^2 / (sum_{j!=u}|g_u^H y_j|^2 + sigma2*||g_u||^2)
#         sinr = np.zeros(self.U, dtype=np.float64)
#         for u in range(self.U):
#             gu = g[u]
#             # desired
#             num = np.abs(np.vdot(gu, y[u])) ** 2

#             # interference
#             inter = 0.0
#             for j in range(self.U):
#                 if j == u:
#                     continue
#                 inter += np.abs(np.vdot(gu, y[j])) ** 2

#             noise = self.sigma2 * (np.linalg.norm(gu) ** 2)
#             den = inter + noise
#             sinr[u] = float(np.real(num) / max(den, 1e-20))

#         # rates
#         R = self._finite_blocklength_rate(sinr)  # bits/s/Hz
#         thr = self.B * R                         # bits/s

#         # arrivals
#         new_bits = self.rng.poisson(lam=1.0, size=self.U) * self.L_mean * 0.5
#         self.queue_bits += new_bits

#         L_take = np.minimum(self.queue_bits, self.L_mean)
#         L_off = rho * L_take
#         L_loc = (1.0 - rho) * L_take

#         # latencies
#         T_loc = (self.C_u * L_loc) / np.maximum(f_loc, 1e-9)

#         thr_safe = np.maximum(thr, 1e-3)
#         T_tx = np.where(L_off > 1e-12, L_off / thr_safe, 0.0)
#         T_tx = np.minimum(T_tx, 1.0)

#         lam_tot = float(np.sum(new_bits))
#         C_tot = float(np.mean(self.C_u))
#         T_Q = self._mg1_queue_upper(lam_tot, C_tot) * np.ones(self.U)

#         T_cpu = (self.C_u * L_off) / np.maximum(self.f_bs, 1e-9)

#         T_e2e = T_loc + T_tx + T_Q + T_cpu

#         # reward
#         T_norm = T_e2e / max(self.T_deadline, 1e-6)
#         q_norm = self.queue_bits / (self.L_mean * 3.0)

#         reward = (
#             R
#             - self.alpha_energy * p
#             - self.beta_latency * T_norm
#             - self.queue_penalty * q_norm
#         )

#         indicator_viol = (T_e2e > self.T_deadline).astype(float)
#         reward -= self.lambda_pen * indicator_viol

#         # bookkeeping
#         self.queue_bits -= L_take
#         self.queue_bits = np.maximum(self.queue_bits, 0.0)
#         self.prev_rho = rho.copy()
#         self.last_f_loc = f_loc.copy()

#         obs = self._obs()

#         done = (indicator_viol.mean() > 0.1).item()
#         term_flags = np.array([done] * self.U)

#         info = {
#             "sinr": sinr,      # linear
#             "R": R,
#             "thr": thr,
#             "T_e2e": T_e2e,
#             "T_loc": T_loc,
#             "T_tx": T_tx,
#             "T_Q": T_Q,
#             "T_cpu": T_cpu,
#             "rho": rho,
#             "p": p,
#         }
#         return obs, reward.astype(np.float32), term_flags, False, info


# mafdrl/envs/mec_urlcc_env.py
import numpy as np
from scipy.stats import norm  # for Q^{-1}(ε)


class MECURLLCEnv:
    """
    Multi-agent uplink MEC with URLLC (finite blocklength).
    One step ~ one scheduling epoch with new channels and task arrivals.

    State per user (obs_dim=4):
      [ queue_bits, ||H_u||_F, last_f_loc, prev_rho ]

    Action per user (act_dim = 2 + Mt + 1):
      [ p, rho, w( Mt dims ), f_loc ]

    Notes:
    - This version fixes the biggest practical issue in your previous runs:
      queueing delay used an M/G/1 bound with WRONG units (bits instead of bits/sec).
      We introduce an epoch duration T_epoch and convert arrivals to rate.
    - We also bound T_Q to keep URLLC feasible and avoid exploding delays.
    """

    def __init__(
        self,
        U=3,
        Mt=2,
        Nr=4,
        bandwidth=1e6,
        noise_pow=1e-13,
        D=200,
        eps=1e-5,
        alpha_path=3.5,
        c0=1.0,
        shadow_sigma_db=6.0,
        p_max=0.5,
        f_loc_max=2.5e9,
        f_bs_max=20e9,
        f_bs=15e9,
        T_deadline=0.01,          # 10 ms
        T_epoch=1e-3,             # 1 ms scheduling epoch (IMPORTANT for units)
        # reward weights
        alpha_energy=1e-3,
        beta_latency=5e-2,
        lambda_pen=5e-2,
        queue_penalty=2e-2,
        # arrivals control (IMPORTANT for URLLC feasibility)
        arrival_scale=0.08,       # <<< reduce/raise if needed (0.05~0.15 typical)
        seed=0,
    ):
        self.rng = np.random.default_rng(seed)
        self.U, self.Mt, self.Nr = int(U), int(Mt), int(Nr)
        self.B = float(bandwidth)
        self.sigma2 = float(noise_pow)
        self.D = int(D)
        self.eps = float(eps)

        self.alpha_path = float(alpha_path)
        self.c0 = float(c0)
        self.shadow_sigma = float(shadow_sigma_db)

        self.p_max = float(p_max)
        self.f_loc_max = float(f_loc_max)

        self.f_bs_max = float(f_bs_max)
        self.f_bs = float(min(f_bs, f_bs_max))

        self.T_deadline = float(T_deadline)
        self.T_epoch = float(T_epoch)

        self.alpha_energy = float(alpha_energy)
        self.beta_latency = float(beta_latency)
        self.lambda_pen = float(lambda_pen)
        self.queue_penalty = float(queue_penalty)

        self.arrival_scale = float(arrival_scale)

        # cycles/bit and mean task size (bits)
        self.C_u = np.full(self.U, 10.0, dtype=float)     # cycles per bit
        self.L_mean = np.full(self.U, 2e5, dtype=float)   # bits

        # queues + last actions
        self.queue_bits = np.zeros(self.U, dtype=float)
        self.prev_rho = np.zeros(self.U, dtype=float)
        self.last_f_loc = np.zeros(self.U, dtype=float)

        # fixed distances
        self.d_u = self.rng.uniform(15.0, 60.0, size=self.U)

        # caches
        self._last_H = None

    # ---------- channel models ----------
    def _shadowing(self):
        s_db = self.rng.normal(0.0, self.shadow_sigma, size=self.U)
        return 10 ** (s_db / 10.0)

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
    def _finite_blocklength_rate(self, sinr_lin):
        """
        R ≈ log2(1+γ) - sqrt(V(γ)/D) * Q^{-1}(ε),
        V(γ) = 1 - (1+γ)^(-2)
        """
        sinr = np.asarray(sinr_lin, dtype=np.float64)
        sinr = np.maximum(sinr, 0.0)

        V = 1.0 - 1.0 / (1.0 + sinr) ** 2
        qinv = float(norm.isf(self.eps))  # Q^{-1}(ε)

        backoff = np.sqrt(np.maximum(V, 1e-12) / max(self.D, 1)) * qinv
        R = np.log2(1.0 + sinr) - backoff
        return np.maximum(R, 0.0)

    def _mg1_queue_upper_seconds(self, lam_bits_per_s, C_cycles_per_bit):
        """
        Unit-consistent M/G/1-ish upper bound in seconds:
        Use f_bs in cycles/s, arrival rate in bits/s, service demand C in cycles/bit.
        """
        denom = self.f_bs - lam_bits_per_s * C_cycles_per_bit
        if denom <= 1e-9:
            return 1e3  # unstable system => huge delay
        return (lam_bits_per_s * (C_cycles_per_bit ** 2)) / (2.0 * denom)

    # ---------- interaction ----------
    def reset(self):
        self.queue_bits[:] = 0.0
        self.prev_rho[:] = 0.0
        self.last_f_loc[:] = 0.0

        # warmup: small queue
        self.queue_bits += self.rng.poisson(1.0, size=self.U) * self.L_mean * 0.05
        return self._obs()

    def _obs(self):
        H = self._channels()
        Hnorm = np.linalg.norm(H.reshape(self.U, -1), axis=1)
        obs = np.stack(
            [self.queue_bits, Hnorm, self.last_f_loc, self.prev_rho],
            axis=1,
        ).astype(np.float32)
        self._last_H = H
        return obs

    def step(self, actions):
        """
        actions: (U, 2+Mt+1) ordered as:
          [ p, rho, w1..w_Mt, f_loc ]
        """
        if isinstance(actions, dict):
            A = np.stack([actions[u] for u in range(self.U)], axis=0)
        else:
            A = np.asarray(actions)

        req_dim = 2 + self.Mt + 1
        if A.ndim != 2 or A.shape[0] != self.U or A.shape[1] < req_dim:
            raise ValueError(f"Expected actions shape (U,{req_dim}), got {A.shape}")

        p = np.clip(A[:, 0], 0.0, self.p_max)
        rho = np.clip(A[:, 1], 0.0, 1.0)
        w = A[:, 2:2 + self.Mt]
        f_loc = np.clip(A[:, 2 + self.Mt], 0.0, self.f_loc_max)

        # normalize beam proxy
        w_norm = np.linalg.norm(w, axis=1, keepdims=True)
        w = w / np.maximum(w_norm, 1e-12)

        # channels
        H = self._last_H
        if H is None:
            H = self._channels()
            self._last_H = H

        # ======== Stable SINR computation (MRC-like) ========
        f_tx = (np.sqrt(p)[:, None]) * w  # (U, Mt)

        # received vectors y_j = H_j f_j  in C^{Nr}
        y = np.zeros((self.U, self.Nr), dtype=complex)
        for j in range(self.U):
            y[j] = (H[j] @ f_tx[j][:, None]).squeeze(-1)

        # combiner g_u = y_u / ||y_u|| (fallback if ||y_u|| small)
        g = np.zeros((self.U, self.Nr), dtype=complex)
        for u in range(self.U):
            denom = np.linalg.norm(y[u])
            if denom < 1e-12:
                h_eff = (H[u] @ w[u][:, None]).squeeze(-1)
                denom2 = np.linalg.norm(h_eff)
                g[u] = h_eff / max(denom2, 1e-12)
            else:
                g[u] = y[u] / denom

        sinr = np.zeros(self.U, dtype=np.float64)
        for u in range(self.U):
            gu = g[u]
            num = np.abs(np.vdot(gu, y[u])) ** 2
            inter = 0.0
            for j in range(self.U):
                if j == u:
                    continue
                inter += np.abs(np.vdot(gu, y[j])) ** 2
            noise = self.sigma2 * (np.linalg.norm(gu) ** 2)
            sinr[u] = float(np.real(num) / max(inter + noise, 1e-20))

        # rates
        R = self._finite_blocklength_rate(sinr)  # bits/s/Hz
        thr = self.B * R                         # bits/s

        # arrivals per epoch (bits)
        new_bits = self.rng.poisson(lam=1.0, size=self.U) * self.L_mean * self.arrival_scale
        self.queue_bits += new_bits

        # serve from queue
        L_take = np.minimum(self.queue_bits, self.L_mean)
        L_off = rho * L_take
        L_loc = (1.0 - rho) * L_take

        # local compute latency
        T_loc = (self.C_u * L_loc) / np.maximum(f_loc, 1e-9)

        # uplink tx latency
        thr_safe = np.maximum(thr, 1e-3)
        T_tx = np.where(L_off > 1e-12, L_off / thr_safe, 0.0)
        T_tx = np.minimum(T_tx, 5.0 * self.T_deadline)

        # ---------- FIXED QUEUEING (UNIT CONSISTENT) ----------
        # Convert arrivals (bits/epoch) to arrival rate (bits/s)
        lam_bits_per_s = float(np.sum(new_bits)) / max(self.T_epoch, 1e-12)
        C_eff = float(np.mean(self.C_u))
        T_Q_scalar = self._mg1_queue_upper_seconds(lam_bits_per_s, C_eff)

        # bound queueing (keeps URLLC feasible; otherwise tails explode)
        T_Q_scalar = float(np.clip(T_Q_scalar, 0.0, 5.0 * self.T_deadline))
        T_Q = T_Q_scalar * np.ones(self.U, dtype=float)

        # BS compute latency
        T_cpu = (self.C_u * L_off) / np.maximum(self.f_bs, 1e-9)

        # end-to-end latency
        T_e2e = T_loc + T_tx + T_Q + T_cpu

        # reward
        T_norm = np.minimum(T_e2e / max(self.T_deadline, 1e-6), 10.0)
        q_norm = np.minimum(self.queue_bits / (self.L_mean * 3.0), 5.0)

        reward = (
            R
            - self.alpha_energy * p
            - self.beta_latency * T_norm
            - self.queue_penalty * q_norm
        )

        indicator_viol = (T_e2e > self.T_deadline).astype(float)
        reward -= self.lambda_pen * indicator_viol
        reward = np.clip(reward, -10.0, 5.0)

        # bookkeeping
        self.queue_bits -= L_take
        self.queue_bits = np.maximum(self.queue_bits, 0.0)
        self.prev_rho = rho.copy()
        self.last_f_loc = f_loc.copy()

        obs = self._obs()

        done = bool(indicator_viol.mean() > 0.25)  # allow some violations before terminating
        term_flags = np.array([done] * self.U)

        info = {
            "sinr": sinr,      # linear
            "R": R,
            "thr": thr,
            "T_e2e": T_e2e,
            "T_loc": T_loc,
            "T_tx": T_tx,
            "T_Q": T_Q,
            "T_cpu": T_cpu,
            "rho": rho,
            "p": p,
        }
        return obs, reward.astype(np.float32), term_flags, False, info
