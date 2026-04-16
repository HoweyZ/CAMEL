import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Building blocks (borrowed & adapted from reference)
# ==============================================================

class AdaptiveLayerNorm(nn.Module):
    """LayerNorm across feature dim, then modulate with external signal."""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x, scale=None, shift=None):
        x = self.norm(x)
        if scale is not None:
            x = x * (1.0 + scale)
        if shift is not None:
            x = x + shift
        return x


class FreNetwork(nn.Module):
    """Frequency-domain filter (from reference)."""
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.complex_weight = nn.Parameter(
            torch.randn(1, 1, time_dim // 2 + 1, 2) * 0.02)
        nn.init.xavier_uniform_(self.complex_weight)

    def forward(self, x):
        # x: [B, N, T]
        x_fft = torch.fft.rfft(x, dim=2, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight
        return torch.fft.irfft(x_fft, n=self.time_dim, dim=2, norm='ortho')


class FConv(nn.Module):
    """Frequency-enhanced linear layer (from reference)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.freq = FreNetwork(out_features)

    def forward(self, x):
        return self.freq(self.linear(x))


# ==============================================================
# Temporal Encoder (enhanced with frequency filtering)
# ==============================================================

class TemporalEncoder(nn.Module):
    """
    Encodes [B, T, N] → [B, N, d_model].
    Uses frequency-enhanced convolution for richer temporal features.
    """
    def __init__(self, d_model, n_nodes, seq_len, dropout=0.1):
        super().__init__()
        self.n_nodes = n_nodes
        self.fconv = FConv(seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, N]
        x = x.permute(0, 2, 1)  # [B, N, T]
        x = self.dropout(F.gelu(self.fconv(x)))  # [B, N, d_model]
        return self.norm(x)


# ==============================================================
# Dual-path Memory Attention (inspired by DLGA)
# ==============================================================

class DualPathMemoryAttention(nn.Module):
    """
    Two paths of linear attention:
      Path 1: query-key-value from input (standard self-attention)
      Path 2: query from input, key from pattern memory, value from input
              (memory-guided attention)
    
    Output = Path1 + Path2, then gated.
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x, b, n):
        # x: [B, N, d_model] → [B, heads, N, d_k]
        return x.view(b, n, self.n_heads, self.d_k).permute(0, 2, 1, 3)

    def forward(self, x, memory_key=None):
        """
        x: [B, N, d_model] — input features
        memory_key: [B, N, d_model] — pattern memory (optional second key path)
        """
        b, n, _ = x.shape
        q = self._reshape_heads(self.q_proj(x), b, n)   # [B, H, N, dk]
        k = self._reshape_heads(self.k_proj(x), b, n)
        v = self._reshape_heads(self.v_proj(x), b, n)

        # Linear attention (softmax on q and k separately → O(N) complexity)
        q_soft = torch.softmax(q / math.sqrt(self.d_k), dim=-1)
        k_soft = torch.softmax(k / math.sqrt(self.d_k), dim=-1)

        # Path 1: standard Q-K-V
        kv = torch.einsum('bhnd,bhne->bhde', k_soft, v)      # [B,H,dk,dk]
        out1 = torch.einsum('bhnd,bhde->bhne', q_soft, kv)   # [B,H,N,dk]

        # Path 2: memory-guided (if pattern memory provided)
        if memory_key is not None:
            m = self._reshape_heads(memory_key, b, n)
            m_soft = torch.softmax(m / math.sqrt(self.d_k), dim=-1)
            mv = torch.einsum('bhnd,bhne->bhde', m_soft, v)
            out2 = torch.einsum('bhnd,bhde->bhne', q_soft, mv)
            out = out1 + out2
        else:
            out = out1

        # Reshape back
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, self.d_model)
        return self.dropout(self.out_proj(out))


# ==============================================================
# Cross-Year Episodic Memory (CEM)
# ==============================================================

class CrossYearEpisodicMemory(nn.Module):
    """
    Two-level memory architecture:
    
    Level 1 — Pattern Bank (learnable, nn.Parameter):
      Per-node learnable patterns that capture recurring temporal archetypes.
      Split into (gate, scale, memory) for adaptive modulation.
      Updated via backpropagation — learns "what patterns matter" globally.
      
    Level 2 — Episodic Bank (buffer, stratified storage):
      Stores actual encoded samples from training, indexed by (season, year).
      Retrieved via cosine similarity at inference.
      Updated via write-on-forward — captures specific historical instances.
      
    The two levels interact:
      - Pattern bank modulates the input via AdaLN (global knowledge)
      - Episodic bank provides retrieved neighbors (specific instances)
      - Dual-path attention fuses both
    """

    def __init__(self, d_model, n_nodes, seq_len,
                 n_seasons=4, n_year_buckets=6, slots_per_bin=32,
                 k_retrieve=8, tau_contrast=0.07, min_year_gap=1.0,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.k = k_retrieve
        self.tau_contrast = tau_contrast
        self.min_year_gap = float(min_year_gap)
        self.d_model = d_model
        self.n_nodes = n_nodes
        self.n_seasons = n_seasons
        self.n_year_buckets = n_year_buckets
        self.slots_per_bin = slots_per_bin
        self.n_bins = n_seasons * n_year_buckets
        self.memory_size = self.n_bins * slots_per_bin

        # --- Encoder ---
        self.encoder = TemporalEncoder(d_model, n_nodes, seq_len, dropout)

        # --- Level 1: Learnable Pattern Bank ---
        # Each node gets a learnable pattern vector, split into 3 roles
        self.pattern_bank = nn.Parameter(
            torch.empty(n_nodes, d_model * 3).uniform_(-0.1, 0.1))

        # --- Modulated attention block ---
        self.norm1 = AdaptiveLayerNorm(d_model)
        self.norm2 = AdaptiveLayerNorm(d_model)
        self.dual_attn = DualPathMemoryAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))

        # --- Episodic retrieval cross-attention ---
        self.episodic_attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads, batch_first=True, dropout=dropout)

        # --- Output projection ---
        self.gate_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.out_proj = nn.Linear(d_model, d_model)

        # --- Level 2: Episodic Memory Bank (stratified buffer) ---
        self.register_buffer('memory_bank',
                             torch.zeros(self.memory_size, n_nodes, d_model))
        self.register_buffer('memory_seasons',
                             torch.full((self.memory_size,), -1, dtype=torch.long))
        self.register_buffer('memory_years',
                             torch.full((self.memory_size,), -1.0))
        self.register_buffer('memory_valid',
                             torch.zeros(self.memory_size, dtype=torch.bool))
        self.register_buffer('bin_ptrs',
                             torch.zeros(self.n_bins, dtype=torch.long))
        self.register_buffer('year_min', torch.tensor(float('inf')))
        self.register_buffer('year_max', torch.tensor(float('-inf')))

    # ------------------------------------------------------------------
    # Year range & bin mapping
    # ------------------------------------------------------------------
    @torch.no_grad()
    def print_memory_distribution(self):
        valid_idx = self.memory_valid.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            print("[MEM DIST] memory is empty")
            return

        mem_years = self.memory_years[valid_idx]
        mem_seasons = self.memory_seasons[valid_idx]

        print("\n[MEM DIST] =================================")
        print(f"total valid: {valid_idx.numel()} / {self.memory_size}")
        print(f"year range : {mem_years.min().item():.2f} ~ {mem_years.max().item():.2f}")

        uniq_years, year_counts = torch.unique(mem_years.round().long(), return_counts=True)
        print("\nYear counts:")
        for y, c in zip(uniq_years.tolist(), year_counts.tolist()):
            print(f"  year {y}: {c}")

        print("\nSeason counts:")
        for s in range(self.n_seasons):
            cnt = (mem_seasons == s).sum().item()
            print(f"  season {s}: {cnt}")

        print("===========================================\n")

    @torch.no_grad()
    def _update_year_range(self, year_labels):
        batch_min = year_labels.min()
        batch_max = year_labels.max()
        if batch_min < self.year_min:
            self.year_min.fill_(batch_min.item())
        if batch_max > self.year_max:
            self.year_max.fill_(batch_max.item())

    def _year_to_bucket(self, year):
        y_min = self.year_min.item()
        y_max = self.year_max.item()
        span = max(y_max - y_min, 1.0)
        if isinstance(year, torch.Tensor):
            return ((year - y_min) / span * (self.n_year_buckets - 1)
                    ).round().long().clamp(0, self.n_year_buckets - 1)
        bucket = int(round((year - y_min) / span * (self.n_year_buckets - 1)))
        return max(0, min(bucket, self.n_year_buckets - 1))

    def _get_bin_id(self, season, year):
        yb = self._year_to_bucket(year)
        if isinstance(season, torch.Tensor):
            return season.long() * self.n_year_buckets + yb
        return int(season) * self.n_year_buckets + int(yb)

    def _bin_slice(self, bin_id):
        start = bin_id * self.slots_per_bin
        return start, start + self.slots_per_bin

    # ------------------------------------------------------------------
    # Episodic storage
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_memory(self, encoded_batch, season_labels, year_labels):
        self._update_year_range(year_labels)
        b = encoded_batch.shape[0]
        for i in range(b):
            s = max(0, min(int(season_labels[i].item()), self.n_seasons - 1))
            y = float(year_labels[i].item())
            bin_id = self._get_bin_id(s, y)
            if isinstance(bin_id, torch.Tensor):
                bin_id = int(bin_id.item())
            bin_id = max(0, min(bin_id, self.n_bins - 1))
            start, _ = self._bin_slice(bin_id)
            local_ptr = int(self.bin_ptrs[bin_id].item())
            write_idx = start + (local_ptr % self.slots_per_bin)
            self.memory_bank[write_idx] = encoded_batch[i]
            self.memory_seasons[write_idx] = s
            self.memory_years[write_idx] = y
            self.memory_valid[write_idx] = True
            self.bin_ptrs[bin_id] = local_ptr + 1

    # ------------------------------------------------------------------
    # Episodic retrieval
    # ------------------------------------------------------------------
    def retrieve(self, q, season_q=None, year_q=None, debug=False):
        b = q.shape[0]
        valid_idx = self.memory_valid.nonzero(as_tuple=True)[0]
        m = valid_idx.numel()
        if m <= 0:
            return q.unsqueeze(1).expand(-1, self.k, -1, -1)

        mem_bank = self.memory_bank[valid_idx]
        q_flat = F.normalize(q.reshape(b, -1), dim=-1)
        mem_flat = F.normalize(mem_bank.reshape(m, -1), dim=-1)
        sim = q_flat @ mem_flat.T

        k_actual = min(self.k, m)
        _, topk_local = sim.topk(k_actual, dim=-1)
        retrieved = mem_bank[topk_local]

        if k_actual < self.k:
            pad = q.unsqueeze(1).expand(-1, self.k - k_actual, -1, -1)
            retrieved = torch.cat([retrieved, pad], dim=1)
        
        if debug:
            mem_years = self.memory_years[valid_idx]
            mem_seasons = self.memory_seasons[valid_idx]

            print("\n[MEM READ DEBUG] =================================")
            for i in range(b):
                if year_q is not None and season_q is not None:
                    print(f"\nQuery {i}: year={year_q[i].item():.2f}, season={int(season_q[i].item())}")
                else:
                    print(f"\nQuery {i}")

                for j in range(k_actual):
                    local_idx = topk_local[i, j].item()
                    global_idx = valid_idx[local_idx].item()
                    mem_year = mem_years[local_idx].item()
                    mem_season = int(mem_seasons[local_idx].item())
                    mem_bucket = self._year_to_bucket(mem_year)

                    print(
                        f"  top{j}: slot={global_idx:4d} | "
                        f"sim={sim[i, local_idx].item():.4f} | "
                        f"season={mem_season} | "
                        f"year={mem_year:.2f} | "
                        f"yb={mem_bucket}"
                    )
            print("==================================================\n")
        return retrieved

    # ------------------------------------------------------------------
    # Forward: two-level memory interaction
    # ------------------------------------------------------------------
    def forward(self, x_scalar, season_q, year_q, gap_years=0.0):
        """
        x_scalar: [B, T, N]
        
        Flow:
          1. Encode input → q [B, N, d_model]
          2. Pattern bank → gate, scale, memory [B, N, d_model] each
          3. AdaLN modulation + dual-path attention (pattern-guided)
          4. Episodic retrieval → cross-attention (instance-guided)
          5. Gated fusion of pattern path and episodic path
        """
        b = x_scalar.shape[0]

        # Step 1: Encode
        q = self.encoder(x_scalar)  # [B, N, d_model]

        # Step 2: Pattern bank → three roles
        # pattern_bank: [N, d_model*3] → expand to [B, N, d_model*3]
        pb = self.pattern_bank.unsqueeze(0).expand(b, -1, -1)
        gate, scale, memory = pb.chunk(3, dim=-1)  # each [B, N, d_model]

        # Step 3: Pattern-modulated dual-path attention
        # AdaLN: normalize then modulate with scale from pattern bank
        h = q + gate * self.dual_attn(
            self.norm1(q, scale=scale), memory_key=memory)
        h = h + gate * self.ffn(self.norm2(h, scale=scale))

        # Step 4: Episodic retrieval + cross-attention
        retrieved = self.retrieve(q)  # [B, K, N, d_model]
        k_ret = retrieved.shape[1]

        # Per-node cross-attention: query=h, kv=retrieved
        h_flat = h.reshape(b * self.n_nodes, 1, self.d_model)
        ret_flat = retrieved.permute(0, 2, 1, 3).reshape(
            b * self.n_nodes, k_ret, self.d_model)
        h_episodic, _ = self.episodic_attn(h_flat, ret_flat, ret_flat)
        h_episodic = h_episodic.reshape(b, self.n_nodes, self.d_model)

        # Step 5: Gated fusion
        fuse_gate = self.gate_fuse(torch.cat([h, h_episodic], dim=-1))
        h_out = fuse_gate * h + (1 - fuse_gate) * h_episodic

        return self.out_proj(h_out), q

    # ------------------------------------------------------------------
    # Contrastive loss
    # ------------------------------------------------------------------
    def contrastive_loss(self, q, season_q, year_q, gap_years=0.0):
        b = q.shape[0]
        valid_idx = self.memory_valid.nonzero(as_tuple=True)[0]
        m = valid_idx.numel()
        if m <= 0:
            return q.new_tensor(0.0)

        mem_bank = self.memory_bank[valid_idx]
        mem_seasons = self.memory_seasons[valid_idx]
        mem_years = self.memory_years[valid_idx]

        q_flat = F.normalize(q.reshape(b, -1), dim=-1)
        mem_flat = F.normalize(mem_bank.reshape(m, -1), dim=-1)
        logits = (q_flat @ mem_flat.T) / self.tau_contrast

        season_match = (season_q.unsqueeze(1) == mem_seasons.unsqueeze(0))
        delta_year = (year_q.unsqueeze(1) - mem_years.unsqueeze(0)).abs()
        pos_mask = season_match & (delta_year >= self.min_year_gap)

        loss_nce = q.new_tensor(0.0)
        valid_count = 0
        for i in range(b):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            if pos_idx.numel() == 0:
                fallback = (delta_year[i] >= 0.5).nonzero(as_tuple=True)[0]
                if fallback.numel() == 0:
                    continue
                pos_idx = fallback
            log_sum_pos = torch.logsumexp(logits[i, pos_idx], dim=0)
            log_sum_all = torch.logsumexp(logits[i], dim=0)
            loss_nce = loss_nce + (log_sum_all - log_sum_pos)
            valid_count += 1
        loss_nce = loss_nce / max(valid_count, 1)

        # Uniformity (anti-collapse)
        if b > 1:
            pw = q_flat @ q_flat.T
            mask_diag = ~torch.eye(b, dtype=torch.bool, device=q.device)
            loss_uniform = pw[mask_diag].mean()
        else:
            loss_uniform = q.new_tensor(0.0)

        return loss_nce + 0.5 * loss_uniform

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def get_memory_stats(self):
        season_names = ['spring', 'summer', 'autumn', 'winter']
        stats = {
            'total_valid': int(self.memory_valid.sum().item()),
            'capacity': self.memory_size,
            'year_range': f"{self.year_min.item():.1f}-{self.year_max.item():.1f}",
            'bins': {}
        }
        for s in range(self.n_seasons):
            for yb in range(self.n_year_buckets):
                bid = s * self.n_year_buckets + yb
                st, en = self._bin_slice(bid)
                v = self.memory_valid[st:en]
                c = int(v.sum().item())
                if c > 0:
                    yrs = self.memory_years[st:en][v]
                    stats['bins'][f"{season_names[s]}/yb{yb}"] = {
                        'count': c,
                        'years': f"{yrs.min():.0f}-{yrs.max():.0f}"}
        return stats


# ==============================================================
# Remaining modules
# ==============================================================

class TimeEncoding(nn.Module):
    def __init__(self, d_model, steps_per_day=288):
        super().__init__()
        self.steps_per_day = steps_per_day
        day_period = steps_per_day
        week_period = 7 * day_period
        year_period = 365 * day_period
        self.periods = [1, day_period, week_period, year_period]
        self.proj = nn.Linear(len(self.periods) * 2, d_model)

    def forward(self, t_scalar):
        feats = []
        for p in self.periods:
            feats.append(torch.cos(2 * torch.pi * t_scalar / p))
            feats.append(torch.sin(2 * torch.pi * t_scalar / p))
        return self.proj(torch.stack(feats, dim=-1))

#ode0
class GraphODEFunc(nn.Module):
    def __init__(self, d_latent, n_nodes, d_embed=32, dropout=0.1,
                 steps_per_day=12):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_emb1 = nn.Embedding(n_nodes, d_embed)
        self.node_emb2 = nn.Embedding(n_nodes, d_embed)
        self.time_mod = nn.Sequential(
            nn.Linear(d_embed, d_embed), nn.GELU(),
            nn.Linear(d_embed, n_nodes * n_nodes))
        self.graph_proj = nn.Linear(d_latent, d_latent)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_latent + d_embed, d_latent * 2), nn.GELU(),
            nn.Linear(d_latent * 2, d_latent))
        self.time_enc = TimeEncoding(d_embed, steps_per_day=steps_per_day)
        self.norm = nn.LayerNorm(d_latent)
        self.dropout = nn.Dropout(dropout)

    def adaptive_adj(self, t_feat):
        idx = torch.arange(self.n_nodes, device=t_feat.device)
        e1 = self.node_emb1(idx)
        e2 = self.node_emb2(idx)
        base_adj = e1 @ e2.T
        mod = self.time_mod(t_feat).reshape(-1, self.n_nodes, self.n_nodes)
        return F.softmax(base_adj.unsqueeze(0) * mod, dim=-1)

    def forward(self, t, z):
        b = z.shape[0]
        t_tensor = torch.full((b,), float(t), device=z.device, dtype=z.dtype)
        t_feat = self.time_enc(t_tensor)
        adj = self.adaptive_adj(t_feat)
        z_spatial = self.dropout(F.gelu(self.graph_proj(torch.bmm(adj, z))))
        t_expand = t_feat.unsqueeze(1).expand(-1, self.n_nodes, -1)
        z_time = self.dropout(self.time_mlp(torch.cat([z, t_expand], dim=-1)))
        return self.norm(z_spatial + z_time)


class LatentDynamicsExtrapolator(nn.Module):
    def __init__(self, d_model, n_nodes, d_latent=32, n_euler_steps=12,
                 dropout=0.1, steps_per_day=12):
        super().__init__()
        self.n_euler_steps = n_euler_steps
        self.steps_per_day = steps_per_day
        self.encoder = nn.Sequential(
            nn.Linear(1, d_latent * 2), nn.GELU(),
            nn.Linear(d_latent * 2, d_latent))
        self.ode_func = GraphODEFunc(d_latent, n_nodes, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model))

    def euler_integrate(self, z0, t_start, t_end):
        dt = (t_end - t_start) / max(self.n_euler_steps, 1)
        z = z0
        t = float(t_start)
        for _ in range(self.n_euler_steps):
            z = z + dt * self.ode_func(t, z)
            t += dt
        return z

    def forward(self, x_scalar, gap_years=0.0):
        z0 = self.encoder(x_scalar[:, -1, :].unsqueeze(-1))
        gap_steps = float(gap_years) * 365 * self.steps_per_day
        z_gap = self.euler_integrate(z0, 0, gap_steps) if gap_steps > 0 else z0
        return self.decoder(z_gap)

    def ode_reconstruction_loss(self, x_scalar):
        _, t, _ = x_scalar.shape
        half = max(1, t // 2)
        z0 = self.encoder(x_scalar[:, half - 1, :].unsqueeze(-1))
        z_hat = self.euler_integrate(z0, 0, half)
        z_true = self.encoder(x_scalar[:, -1, :].unsqueeze(-1).detach())
        loss_ode = F.mse_loss(z_hat, z_true)
        z_mid = self.euler_integrate(z0, 0, max(1, half // 2))
        loss_smooth = (z_hat - 2 * z_mid + z0).pow(2).mean()
        return loss_ode, loss_smooth


class UncertaintyHead(nn.Module):
    def __init__(self, d_model, horizon):
        super().__init__()
        hid = max(4, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hid), nn.GELU(),
            nn.Linear(hid, horizon), nn.Softplus())

    def forward(self, z):
        return self.net(z) + 1e-6


class AnchorTemporalFusion(nn.Module):
    def __init__(self, d_model, horizon, n_gap_types=4, dropout=0.1):
        super().__init__()
        self.gap_embed = nn.Embedding(n_gap_types, d_model)
        self.x_proj = nn.Linear(1, d_model)
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, 3))
        self.out_proj = nn.Linear(d_model, d_model)
        self.delta_head = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.uncertainty = UncertaintyHead(d_model, horizon)
        self.dropout = nn.Dropout(dropout)

    def gap_to_idx(self, gap_years):
        return {0.0: 0, 1.0: 1, 1.5: 2, 2.0: 3}.get(float(gap_years), 0)

    def forward(self, h_cem, h_lde, x_scalar, gap_years=0.0):
        b, t, n = x_scalar.shape
        x_enc_embed = self.x_proj(x_scalar[:, -1, :].unsqueeze(-1))
        gap_ids = torch.full((b,), self.gap_to_idx(gap_years),
                             dtype=torch.long, device=x_scalar.device)
        e_gap = self.gap_embed(gap_ids).unsqueeze(1).expand(-1, n, -1)
        gates = F.softmax(self.gate_net(
            torch.cat([h_cem, h_lde, x_enc_embed, e_gap], dim=-1)), dim=-1)
        z = (gates[..., 0:1] * h_cem +
             gates[..., 1:2] * h_lde +
             gates[..., 2:3] * x_enc_embed)
        z = self.dropout(self.norm(self.out_proj(z)))
        sigma = self.uncertainty(z)
        delta = self.delta_head(z).squeeze(-1)
        z_out = x_scalar + delta.unsqueeze(1).expand(-1, t, -1)
        return z_out, sigma


class NLLLoss(nn.Module):
    def forward(self, pred, target, sigma):
        return (((pred - target) ** 2) / (2 * sigma ** 2)
                + torch.log(sigma)).mean()


class CAMELCore(nn.Module):
    def __init__(self, d_model, n_nodes, seq_len,
                 n_seasons=4, n_year_buckets=6, slots_per_bin=32,
                 k_retrieve=8, d_latent=32, horizon=12, min_year_gap=1.0,
                 n_heads=4, dropout=0.1, steps_per_day=288,use_which_ode=0, ablate_memory=False, ablate_ode=False, ablate_atf=False):
        super().__init__()
        self.cem = CrossYearEpisodicMemory(
            d_model=d_model, n_nodes=n_nodes, seq_len=seq_len,
            n_seasons=n_seasons, n_year_buckets=n_year_buckets,
            slots_per_bin=slots_per_bin,
            k_retrieve=k_retrieve, min_year_gap=min_year_gap,
            n_heads=n_heads, dropout=dropout)
        self.lde = LatentDynamicsExtrapolator(
            d_model=d_model, n_nodes=n_nodes, d_latent=d_latent,
            dropout=dropout, steps_per_day=steps_per_day)
        self.atf = AnchorTemporalFusion(
            d_model=d_model, horizon=horizon, dropout=dropout)
        self.ablate_memory = ablate_memory
        self.ablate_ode = ablate_ode
        self.ablate_atf = ablate_atf

    def forward(self, x_scalar, season_q, year_q, gap_years=0.0,
                update_memory=True):
        if not self.ablate_memory:
            h_cem, q = self.cem(x_scalar, season_q, year_q, gap_years=gap_years)
            loss_mem = self.cem.contrastive_loss(q, season_q, year_q, gap_years=gap_years)
        else:
            b, _, n = x_scalar.shape
            q = torch.zeros(b, n, self.cem.d_model, device=x_scalar.device, dtype=x_scalar.dtype)
            h_cem = torch.zeros_like(q)
            loss_mem = x_scalar.new_tensor(0.0)

        if not self.ablate_ode:
            h_lde = self.lde(x_scalar, gap_years)
            loss_ode, loss_smooth = self.lde.ode_reconstruction_loss(x_scalar)
        else:
            h_lde = torch.zeros_like(h_cem)
            loss_ode = x_scalar.new_tensor(0.0)
            loss_smooth = x_scalar.new_tensor(0.0)

        if not self.ablate_atf:
            z_out, sigma = self.atf(h_cem, h_lde, x_scalar, gap_years)
        else:
            # 只取当前输入最后一个时间步做 embedding
            x_enc_embed = self.atf.x_proj(x_scalar[:, -1, :].unsqueeze(-1))   # [B, N, d_model]

            # 纯平均消融：只平均，不加任何额外层
            z = (h_cem + h_lde + x_enc_embed) / 3.0

            # 直接把平均结果映射回时间维度输出
            z_out = z.mean(dim=-1, keepdim=False)          # [B, N]
            z_out = z_out.unsqueeze(1).expand(-1, x_scalar.size(1), -1)   # [B, T, N]

            # 消融分支不做不确定性建模
            sigma = None

        if update_memory and not self.ablate_memory and self.training:
            with torch.no_grad():
                    self.cem.update_memory(q.detach(), season_q.detach(), year_q.detach())
        aux = {'mem': loss_mem, 'ode': loss_ode, 'smooth': loss_smooth}
        return z_out, sigma, aux


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = (configs.pred_len
                         if self.task_name in ['long_term_forecast',
                                               'short_term_forecast']
                         else configs.seq_len)
        self.output_attention = getattr(configs, 'output_attention', False)
        self.dropout_rate = getattr(configs, 'dropout', 0.1)
        self.steps_per_day = getattr(configs, 'steps_per_day', 288)

        self.core = CAMELCore(
            d_model=configs.camel_d_model,
            n_nodes=configs.enc_in,
            seq_len=self.seq_len,
            n_seasons=getattr(configs, 'camel_n_seasons', 4),
            n_year_buckets=getattr(configs, 'camel_n_year_buckets', 6),
            slots_per_bin=getattr(configs, 'camel_slots_per_bin', 32),
            k_retrieve=configs.camel_k_retrieve,
            d_latent=configs.camel_latent_dim,
            horizon=self.pred_len,
            min_year_gap=getattr(configs, 'camel_min_year_gap', 1.0),
            n_heads=getattr(configs, 'camel_n_heads', 4),
            dropout=self.dropout_rate,
            steps_per_day=self.steps_per_day,
            use_which_ode=getattr(configs, 'use_which_ode', 0),
            ablate_memory=getattr(configs, 'ablate_memory', False),
            ablate_ode=getattr(configs, 'ablate_ode', False),
            ablate_atf=getattr(configs, 'ablate_atf', False),
            )
        self.temporal_proj = nn.Linear(self.seq_len, self.pred_len)
        self.camel_gap_years = getattr(configs, 'camel_gap_years', 0.0)

    def _extract_meta(self, x_mark_enc, x_enc):
        b = x_enc.shape[0]
        device = x_enc.device
        if x_mark_enc is None or x_mark_enc.numel() == 0:
            return (torch.zeros(b, dtype=torch.long, device=device),
                    torch.zeros(b, dtype=torch.float, device=device))
        mark = x_mark_enc[:, -1, :]
        month_raw = mark[:, -2]
        if month_raw.max() <= 1.0:
            month = (month_raw * 11.0 + 1.0).clamp(1, 12)
        else:
            month = month_raw.clamp(1, 12)
        month_int = month.round().long().clamp(1, 12)
        season_map = torch.tensor(
            [3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3], device=device)
        season_q = season_map[month_int - 1]
        if mark.shape[-1] >= 5:
            year_q = mark[:, -1]
        else:
            year_q = torch.zeros(b, device=device, dtype=torch.float)
        return season_q, year_q.float()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        season_q, year_q = self._extract_meta(x_mark_enc, x_enc)
        z_out, sigma, aux = self.core(
            x_enc, season_q, year_q,
            gap_years=self.camel_gap_years,
            update_memory=self.training)
        out = self.temporal_proj(z_out.permute(0, 2, 1)).permute(0, 2, 1)
        return out, sigma, aux

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            out, sigma, aux = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec)
            return out, {'sigma': sigma, 'aux_losses': aux}
        raise NotImplementedError(
            f"Task '{self.task_name}' not supported yet.")