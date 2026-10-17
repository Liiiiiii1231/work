import numpy as np

from config import SystemConfig
from state import ResourceState


def normalize_vector(
    x: np.ndarray,
    eps: float = 1e-14,
) -> np.ndarray:
    """向量单位范数归一化。"""

    norm = np.linalg.norm(x)

    if norm <= eps:
        raise ValueError(
            "Cannot normalize a near-zero vector."
        )

    return x / norm


def hermitianize(
    a: np.ndarray,
) -> np.ndarray:
    """消除数值误差产生的非 Hermitian 部分。"""

    return 0.5 * (a + a.conj().T)


def relative_change(
    new: float,
    old: float,
) -> float:
    """以后内外层算法的相对停止判据。"""

    return abs(new - old) / max(
        1.0,
        abs(old),
    )


def check_finite(
    x: np.ndarray | float,
    name: str,
) -> None:
    """检查 NaN 和 Inf。"""

    if not np.all(np.isfinite(x)):
        raise ValueError(
            f"{name} contains NaN or Inf."
        )


def check_resource_constraints(
    resources: ResourceState,
    cfg: SystemConfig,
    atol: float = 1e-8,
) -> None:
    """检查当前资源变量是否合法。"""

    q_matrix = resources.q_matrix
    q_ul = resources.q_ul
    b_ul = resources.b_ul
    u_s = resources.u_s

    if q_matrix.shape != (
        cfg.n_tx,
        cfg.d_stream,
    ):
        raise ValueError(
            "Q dimension is incorrect."
        )

    if q_ul.shape != (cfg.j_ul,):
        raise ValueError(
            "q_ul dimension is incorrect."
        )

    if b_ul.shape != (
        cfg.j_ul,
        cfg.n_rx,
    ):
        raise ValueError(
            "b_ul dimension is incorrect."
        )

    if u_s.shape != (
        cfg.l_target,
        cfg.n_rx,
    ):
        raise ValueError(
            "u_s dimension is incorrect."
        )

    tx_power = (
        np.linalg.norm(q_matrix, "fro") ** 2
    )

    if tx_power > cfg.p_dl_max + atol:
        raise ValueError(
            "Transmit power constraint violated."
        )

    if np.any(q_ul < -atol):
        raise ValueError(
            "Negative UL amplitude."
        )

    if np.any(
        q_ul
        > np.sqrt(cfg.p_ul_max) + atol
    ):
        raise ValueError(
            "UL power constraint violated."
        )

    b_norms = np.linalg.norm(
        b_ul,
        axis=1,
    )

    u_norms = np.linalg.norm(
        u_s,
        axis=1,
    )

    if not np.allclose(
        b_norms,
        1.0,
        atol=atol,
    ):
        raise ValueError(
            "b_j is not unit norm."
        )

    if not np.allclose(
        u_norms,
        1.0,
        atol=atol,
    ):
        raise ValueError(
            "u_l is not unit norm."
        )
def split_transmit_matrix(
    q_matrix: np.ndarray,
    cfg: SystemConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """从合并发射矩阵 Q=[W,V] 中拆出 W 和 V。

    参数
    ----------
    q_matrix:
        合并发射波束矩阵 Q。
        shape = (N_T, K + R_s)

        前 K 列：
            W = [w_1, ..., w_K]

        后 R_s 列：
            V = [v_1, ..., v_Rs]

    cfg:
        系统参数。

    返回
    ----------
    w_matrix:
        下行通信波束矩阵 W。
        shape = (N_T, K)

    v_matrix:
        感知波束矩阵 V。
        shape = (N_T, R_s)
    """

    expected_shape = (
        cfg.n_tx,
        cfg.d_stream,
    )

    if q_matrix.shape != expected_shape:
        raise ValueError(
            f"Q.shape={q_matrix.shape}, "
            f"expected {expected_shape}."
        )

    # Q 的前 K 列是下行通信波束矩阵 W。
    w_matrix = q_matrix[
        :,
        : cfg.k_dl,
    ]

    # Q 的后 R_s 列是感知波束矩阵 V。
    v_matrix = q_matrix[
        :,
        cfg.k_dl :,
    ]

    return w_matrix, v_matrix
    
    