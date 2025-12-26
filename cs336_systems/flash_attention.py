import torch
from jaxtyping import Float
from torch import autograd

from einops import einsum, rearrange


class FlashAttentionPytorch(autograd.Function):
    b_q: int = 32
    b_k: int = 32

    @staticmethod
    def forward(ctx, q: Float[torch.Tensor, "n_q d_model"], k: Float[torch.Tensor, "n_k d_model"],
                v: Float[torch.Tensor, " n_k d_model"], is_causal=False):
        #print(f"!!!! ruizhen q shape = {q.shape}")
        B_q, n_q, D = q.shape
        B_k, n_k, D = k.shape
        assert B_q == B_k
        B = B_q
        t_q: int = n_q // FlashAttentionPytorch.b_q
        t_k: int = n_k // FlashAttentionPytorch.b_k

        q_transformed = rearrange(q, "B n_q D -> (B n_q) D")
        k_transformed = rearrange(k, "B n_k D -> (B n_k) D")
        v_transformed = rearrange(v, "B n_k D -> (B n_k) D")
        #print(f"!!!! ruizhen q_transformed shape = {q_transformed.shape}")

        O = torch.zeros(B, n_q, D)
        O_transformed = rearrange(O, "B n_q D -> (B n_q) D")
        L = torch.zeros(B, n_q, )
        L_transformed = rearrange(L, "B n_q -> (B n_q)")
        for b_idx in range(B):
            q_off = b_idx * n_q
            k_off = b_idx * n_k
            l_off = b_idx * n_q
            for i in range(t_q):
                Q_i = q_transformed[q_off+i*FlashAttentionPytorch.b_q : q_off+(i+1)*FlashAttentionPytorch.b_q, :]
                O_i_pre = torch.zeros(FlashAttentionPytorch.b_q, D)
                l_i_pre = torch.zeros(FlashAttentionPytorch.b_q, )
                min_val = torch.finfo(q.dtype).min
                m_i_pre = torch.full((FlashAttentionPytorch.b_q, ), min_val)

                for j in range(t_k):
                    K_j = k_transformed[k_off + j * FlashAttentionPytorch.b_k : k_off + (j + 1) * FlashAttentionPytorch.b_k, :]
                    V_j = v_transformed[k_off + j * FlashAttentionPytorch.b_k : k_off + (j + 1) * FlashAttentionPytorch.b_k, :]
                    S_i_j = einsum(Q_i, K_j, "b_q d_model, b_k d_model -> b_q b_k")
                    S_i_j = S_i_j / (D ** 0.5)
                    m_i_j = torch.max(S_i_j, dim=1).values
                    m_i_j = torch.maximum(m_i_pre, m_i_j)

                    P_i_j: Float[torch.Tensor, "b_q b_k"] = torch.exp(S_i_j - m_i_j.unsqueeze(1))
                    m_temp = torch.exp(m_i_pre - m_i_j)
                    l_i_j = m_temp * l_i_pre + torch.sum(P_i_j, dim=1)

                    P_V = einsum(P_i_j, V_j, "b_q b_k, b_k d_model -> b_q d_model")
                    m_temp_diag: Float[torch.Tensor, "b_q b_q"] = torch.diag(m_temp)

                    O_i_j = einsum(m_temp_diag, O_i_pre, "b_q b_q, b_q d_model -> b_q d_model")
                    O_i_j = O_i_j + P_V

                    # update pre
                    O_i_pre = O_i_j
                    l_i_pre = l_i_j
                    m_i_pre = m_i_j

                diag_temp: Float[torch.Tensor, "b_q b_q"] = torch.diag(1.0 / l_i_pre)

                O_i = einsum(diag_temp, O_i_pre, "b_q b_q, b_q d_model -> b_q d_model")
                L_i = m_i_pre + torch.log(l_i_pre)

                print(O_i)

                O_transformed[q_off + i*FlashAttentionPytorch.b_q:q_off + (i+1)*FlashAttentionPytorch.b_q, :] = O_i
                L_transformed[l_off + i*FlashAttentionPytorch.b_q:l_off + (i+1)*FlashAttentionPytorch.b_q, ] = L_i

        O = rearrange(O_transformed, "(B n_q) D -> B n_q D", B = B, n_q = n_q, D = D)
        L = rearrange(L_transformed, "(B n_q) -> B n_q", B=B, n_q=n_q)
        ctx.save_for_backward(q, k, v, L)
        ctx.is_causal = is_causal

        return O







    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        B_q, n_q, d_model = q.shape
        B_k, n_k, d_model = k.shape

        scale = 1.0 / (d_model ** 0.5)

        dO = grad_output
        S = einsum(q, k, "B n_q d_model, B n_k d_model -> B n_q n_k")
        S = S * scale

        if is_causal:
            _, n_q, n_k = S.shape
            mask = torch.tril(
                torch.ones(n_q, n_k, device=q.device, dtype=torch.bool),
                diagonal=0
            )
            S.masked_fill(~mask, -1e6)

        P_i_j = torch.exp(S - L.unsqueeze(-1))
        dV = einsum(P_i_j, dO, "B n_q n_k, B n_q d_model -> B n_k d_model")
        dP = einsum(dO, v, "B n_q d_model, B n_k d_model -> B n_q n_k")

        D = P_i_j * dP
        # B n_q
        D = torch.sum(D, dim=-1)
        dS_i_j = P_i_j * (dP - D.unsqueeze(-1))

        dQ = einsum(dS_i_j, k, "B n_q n_k, B n_k d_model -> B n_q d_model")
        dQ = dQ * scale

        dK = einsum(dS_i_j, q, "B n_q n_k, B n_q d_model -> B n_k d_model")
        dK = dK * scale

        return dQ, dK, dV, None
