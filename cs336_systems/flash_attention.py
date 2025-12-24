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
        B, n_q, D = q.shape
        B, n_k, D = k.shape
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

        for i in range(t_q):
            Q_i = q_transformed[i*FlashAttentionPytorch.b_q:(i+1)*FlashAttentionPytorch.b_q, :]
            O_i_pre = torch.zeros(FlashAttentionPytorch.b_q, D)
            l_i_pre = torch.zeros(FlashAttentionPytorch.b_q, )
            min_val = torch.finfo(torch.float32).min
            m_i_pre = torch.full((FlashAttentionPytorch.b_q, ), min_val)

            for j in range(t_k):
                K_j = k_transformed[j * FlashAttentionPytorch.b_k:(j + 1) * FlashAttentionPytorch.b_k, :]
                V_j: Float[torch.Tensor, "b_k d_model"] = v_transformed[j * FlashAttentionPytorch.b_k:(j + 1) * FlashAttentionPytorch.b_k, :]
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

            O_transformed[i*FlashAttentionPytorch.b_q:(i+1)*FlashAttentionPytorch.b_q, :] = O_i
            L_transformed[i*FlashAttentionPytorch.b_q:(i+1)*FlashAttentionPytorch.b_q, ] = L_i

        O = rearrange(O_transformed, "(B n_q) D -> B n_q D", B = B, n_q = n_q, D = D)
        L = rearrange(L_transformed, "(B n_q) -> B n_q", B=B, n_q=n_q)
        ctx.save_for_backward(q, k, v, O, L)
        print(f"!!!! ruizhen O shape = {O.shape},  L = {L.shape}")

        return O







    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
