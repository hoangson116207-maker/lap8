import numpy as np


def loss_function(w1, w2):
    """Hàm mất mát giả lập: Loss(w1, w2) = (w1 - 3)^2 + (w2 - 3)^2"""
    return (w1 - 3) ** 2 + (w2 - 3) ** 2


# =====================================================================
# 1. MÔ PHỎNG RÀNG BUỘC L1 (Lasso)
# =====================================================================
def solve_constrained_l1(C=2, num_points=1000):
    """Duyệt trên các cạnh của hình thoi |w1| + |w2| <= C để tìm điểm tối tiểu hóa Loss."""
    # Vì tâm sai số nằm ở (3, 3) (góc phần tư thứ I),
    # điểm tiếp xúc tối ưu chắc chắn nằm trên đoạn w1 + w2 = C với w1, w2 >= 0.
    w1_vals = np.linspace(0, C, num_points)
    w2_vals = C - w1_vals  # w1 + w2 = C

    losses = loss_function(w1_vals, w2_vals)
    min_idx = np.argmin(losses)

    optimal_w = (w1_vals[min_idx], w2_vals[min_idx])
    min_loss = losses[min_idx]
    return optimal_w, min_loss


# =====================================================================
# 2. MÔ PHỎNG RÀNG BUỘC L2 (Ridge)
# =====================================================================
def solve_constrained_l2(radius_sq=4, num_points=1000):
    """Tìm điểm tiếp xúc trên đường tròn w1^2 + w2^2 <= radius_sq."""
    R = np.sqrt(radius_sq)