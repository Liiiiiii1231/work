"""检查科研计算环境是否安装正确。"""

import cvxpy as cp
import matplotlib
import numpy as np
import scipy


def main() -> None:
    print("NumPy:", np.__version__)
    print("SciPy:", scipy.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("CVXPY:", cp.__version__)
    print("CVXPY solvers:", cp.installed_solvers())

    # 检查复数矩阵运算
    a = np.array(
        [[1 + 1j, 2 - 1j], [3 + 0j, 4 + 2j]],
        dtype=np.complex128,
    )
    x = np.array([1 + 0j, 2 - 1j], dtype=np.complex128)

    print("A shape:", a.shape)
    print("x shape:", x.shape)
    print("A @ x:", a @ x)
    print("Environment check passed.")


if __name__ == "__main__":
    main()