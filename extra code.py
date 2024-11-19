# Plotting
fig, ax = plt.subplots(5, 2, figsize=(12, 10))

# Plot "Before Training" (Early Trial)
ax[0, 0].plot(u, label="u (Stimulus)", color="black")
ax[1, 0].plot(r, label="r (Reward)", color="blue")
ax[2, 0].plot(v_early, label="v (Prediction)", color="orange")
ax[3, 0].plot(delta_v_early, label="Δv", color="purple")
ax[4, 0].plot(delta_early, label="δ (TD Error)", color="red")

# Titles and labels for before training
ax[0, 0].set_title("Before Training")
for i in range(5):
    ax[i, 0].legend(loc="upper right")
    ax[i, 0].set_xlim([0, time_steps])

# Plot "After Training" (Late Trial)
ax[0, 1].plot(u, label="u (Stimulus)", color="black")
ax[1, 1].plot(r, label="r (Reward)", color="blue")
ax[2, 1].plot(v_late, label="v (Prediction)", color="orange")
ax[3, 1].plot(delta_v_late, label="Δv", color="purple")
ax[4, 1].plot(delta_late, label="δ (TD Error)", color="red")

# Titles and labels for after training
ax[0, 1].set_title("After Training")
for i in range(5):
    ax[i, 1].legend(loc="upper right")
    ax[i, 1].set_xlim([0, time_steps])

# Display plot
fig.tight_layout()
plt.show()