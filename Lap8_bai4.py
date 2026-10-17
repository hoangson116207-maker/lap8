def gradient_descent_1d(initial_w, learning_rate, epochs):
    w = initial_w
    loss_history = []

    for epoch in range(epochs):
        # Tính giá trị hàm mất mát L(w) = w^2 - 4*w + 4
        loss = w**2 - 4 * w + 4
        loss_history.append(loss)

        # Đạo hàm: L'(w) = 2*w - 4
        grad = 2 * w - 4

        # Cập nhật w: w_new = w - learning_rate * grad
        w = w - learning_rate * grad

    # Lưu lại loss tại điểm w tối ưu cuối cùng (tùy chọn)
    # loss_history.append(w**2 - 4*w + 4)

    return w, loss_history


# --- Test case mẫu ---
if __name__ == "__main__":
    initial_w = 0.0  # Điểm khởi tạo
    learning_rate = 0.1  # Tốc độ học (alpha)
    epochs = 20  # Số vòng lặp

    w_opt, history = gradient_descent_1d(initial_w, learning_rate, epochs)

    print(f"Trọng số tối ưu w*: {w_opt:.4f}")
    print(f"Loss ban đầu: {history[0]:.4f}")
    print(f"Loss cuối cùng: {history[-1]:.4f}")