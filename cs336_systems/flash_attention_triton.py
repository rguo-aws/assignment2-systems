import triton
import triton.language as tl

import torch
from jaxtyping import Float
from torch import autograd

from einops import einsum, rearrange


@triton.jit
def flash_fwd_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, stride_qb, stride_qq, stride_qd, stride_kb, stride_kk,
                     stride_kd, stride_vb, stride_vk, stride_vd, stride_ob, stride_oq, stride_od, stride_lb, stride_lq,
                     N_QUERIES, N_KEYS, scale, D: tl.constexpr, Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb, shape=(N_QUERIES, D),
                                    strides=(stride_qq, stride_qd), offsets=(query_tile_index * Q_TILE_SIZE, 0),
                                    block_shape=(Q_TILE_SIZE, D), order=(1, 0), )

    K_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb, shape=(N_KEYS, D),
                                    strides=(stride_kk, stride_kd), offsets=(query_tile_index * K_TILE_SIZE, 0),
                                    block_shape=(K_TILE_SIZE, D), order=(1, 0), )

    V_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb, shape=(N_KEYS, D),
                                    strides=(stride_vk, stride_vd), offsets=(query_tile_index * K_TILE_SIZE, 0),
                                    block_shape=(K_TILE_SIZE, D), order=(1, 0), )

    O_block_ptr = tl.make_block_ptr(O_ptr + batch_index * stride_ob, shape=(N_QUERIES, D),
                                    strides=(stride_oq, stride_od), offsets=(query_tile_index * Q_TILE_SIZE, 0),
                                    block_shape=(Q_TILE_SIZE, D), order=(1, 0), )

    L_block_ptr = tl.make_block_ptr(L_ptr + batch_index * stride_lb, shape=(N_QUERIES,),
                                    strides=(stride_lq,), offsets=(query_tile_index * Q_TILE_SIZE,),
                                    block_shape=(Q_TILE_SIZE,), order=(0,), )

    t_k = tl.cdiv(N_KEYS, K_TILE_SIZE)

    O_i_pre = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i_pre = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i_pre = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)

    offs_m = tl.arange(0, Q_TILE_SIZE)
    offs_n = tl.arange(0, Q_TILE_SIZE)

    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    for j in range(t_k):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # b_q d_model, b_k d_model -> b_q b_k
        S_i_j = tl.dot(Q_i, tl.trans(K_j))
        S_i_j = S_i_j / scale
        m_i_j = tl.max(S_i_j, dim=1).values
        m_i_j = tl.maximum(m_i_pre, m_i_j)

        P_i_j = tl.exp(S_i_j - m_i_j[:, None])
        m_temp = tl.exp(m_i_pre - m_i_j)
        l_i_j = m_temp * l_i_pre + tl.sum(P_i_j, dim=1)

        # b_q b_k, b_k d_model -> b_q d_model
        P_V = tl.dot(P_i_j.to(V_j.dtype), V_j)
        # generate diag in triton : "b_q b_q"
        m_temp_diag = tl.where(offs_m[:, None] == offs_n[None, :], m_temp[:, None], 0.0)
        m_temp_diag = m_temp_diag.to(O_i_pre.dtype)

        # b_q b_q, b_q d_model -> b_q d_model
        O_i_j = tl.dot(m_temp_diag, O_i_pre)
        O_i_j = O_i_j + P_V

        # update pre
        O_i_pre = O_i_j
        l_i_pre = l_i_j
        m_i_pre = m_i_j

        # forward ptr
        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((0, K_TILE_SIZE))

    l_i_pre_rev = 1.0 / l_i_pre
    # "b_q b_q"
    diag_temp = tl.where(offs_m[:, None] == offs_n[None, :], l_i_pre_rev[:, None], 0.0)
    diag_temp = diag_temp.to(O_i_pre.dtype)

    # b_q b_q, b_q d_model -> b_q d_model
    O_i = tl.dot(diag_temp, O_i_pre)
    L_i = m_i_pre + tl.log(l_i_pre)

    # tl.store(output_block_ptr, output, boundary_check=(0,))
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


class FlashAttentionTriton(autograd.Function):
    b_q: int = 32
    b_k: int = 32

    @staticmethod
    def forward(ctx, q: Float[torch.Tensor, "n_q d_model"], k: Float[torch.Tensor, "n_k d_model"],
                v: Float[torch.Tensor, " n_k d_model"], is_causal=False):
        B_q, n_q, D = q.shape
        B_k, n_k, D = k.shape
        assert B_q == B_k
        B = B_q

        t_q: int = n_q // FlashAttentionTriton.b_q

        # q_transformed = rearrange(q, "B n_q D -> (B n_q) D")
        # k_transformed = rearrange(k, "B n_k D -> (B n_k) D")
        # v_transformed = rearrange(v, "B n_k D -> (B n_k) D")

        O = torch.zeros(B, n_q, D)
        # O_transformed = rearrange(O, "B n_q D -> (B n_q) D")
        L = torch.zeros(B, n_q, )
        # L_transformed = rearrange(L, "B n_q -> (B n_q)")

        flash_fwd_kernel[(t_q, B,)](Q_ptr=q,
                                    K_ptr=k,
                                    V_ptr=v,
                                    O_ptr=O,
                                    L_ptr=L,
                                    stride_qb=q.stride(0), stride_qq=q.stride(1), stride_qd=q.stride(2),
                                    stride_kb=k.stride(0), stride_kk=k.stride(1), stride_kd=k.stride(2),
                                    stride_vb=v.stride(0), stride_vk=v.stride(1), stride_vd=v.stride(2),
                                    stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
                                    stride_lb=L.stride(0), stride_lq=L.stride(1),
                                    N_QUERIES=n_q, N_KEYS=n_k, scale=D ** 0.5,
                                    D=D, Q_TILE_SIZE=FlashAttentionTriton.b_q, K_TILE_SIZE=FlashAttentionTriton.b_k)

        ctx.save_for_backward(q, k, v, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError