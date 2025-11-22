# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: congestion-bogota
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Set plot style for better visualization
plt.style.use("seaborn-v0_8-whitegrid")


# %%
class WeightedMSELoss(nn.Module):
    """
    A custom weighted Mean Squared Error loss function for traffic forecasting.

    This loss function applies two types of weights to the standard MSE:
    1. Jam-Level Weight: Increases the loss for observations with higher true jam levels.
       This helps the model focus on predicting important high-traffic events in an
       unbalanced dataset. The weight is calculated as (1 + alpha * y_true).
    2. Time-Step Weight: Increases the loss for predictions further in the future.
       This encourages the model to maintain accuracy over a long forecast horizon.
       The weight is calculated as (1 + beta * t) for a linear increase.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.05):
        """
        Initializes the WeightedMSELoss.

        Args:
            alpha (float): Hyperparameter to control the jam-level weighting.
                           Higher values penalize errors on high-jam instances more.
            beta (float): Hyperparameter to control the time-step weighting.
                          Higher values penalize errors on distant future predictions more.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the weighted MSE loss.

        Args:
            y_pred (torch.Tensor): The model's predictions. Shape: [B, N, T].
            y_true (torch.Tensor): The ground truth values. Shape: [B, N, T].

        Returns:
            torch.Tensor: The final computed loss value.
        """
        assert y_pred.shape == y_true.shape, (
            "Prediction and ground truth must have the same shape"
        )

        squared_error = torch.pow(y_pred - y_true, 2)

        # These weights are based on the ground truth values
        jam_weights = 1 + self.alpha * y_true
        time_steps = torch.arange(
            y_true.shape[2], device=y_true.device, dtype=y_true.dtype
        )
        time_weights = 1 + self.beta * time_steps

        # Reshape time_weights for broadcasting to [1, 1, T]
        time_weights = time_weights.view(1, 1, -1)

        # Apply both weights to the squared error
        weighted_squared_error = jam_weights * time_weights * squared_error

        loss = torch.mean(weighted_squared_error)
        return loss


# %%
# Instantiate a standard MSE loss for comparison
mse_loss_fn = nn.MSELoss()

# %% [markdown]
# ## Test 1: Weight of jam level (fixed time)

# %%
weighted_loss_fn_alpha_1 = WeightedMSELoss(alpha=1.0, beta=0.1)
weighted_loss_fn_alpha_2 = WeightedMSELoss(alpha=2.0, beta=0.1)

true_jams = torch.linspace(0, 4, 100)
# Introduce a fixed error of +1.0
pred_jams = true_jams + 1.0

# Reshape for loss function (B=1, N=1, T=1)
true_jams_reshaped = true_jams.view(1, 1, -1)
pred_jams_reshaped = pred_jams.view(1, 1, -1)

# Calculate loss for each point
weighted_losses_jam_1 = []
weighted_losses_jam_2 = []
mse_losses_jam = []

for i in range(len(true_jams)):
    # We pass single values with the correct shape
    w_loss_1 = weighted_loss_fn_alpha_1(
        pred_jams_reshaped[:, :, i : i + 1], true_jams_reshaped[:, :, i : i + 1]
    )
    w_loss_2 = weighted_loss_fn_alpha_2(
        pred_jams_reshaped[:, :, i : i + 1], true_jams_reshaped[:, :, i : i + 1]
    )
    m_loss = mse_loss_fn(
        pred_jams_reshaped[:, :, i : i + 1], true_jams_reshaped[:, :, i : i + 1]
    )

    weighted_losses_jam_1.append(w_loss_1.item())
    weighted_losses_jam_2.append(w_loss_2.item())
    mse_losses_jam.append(m_loss.item())

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(
    true_jams.numpy(),
    mse_losses_jam,
    label="Standard MSE Loss",
    linestyle="--",
    color="gray",
)
plt.plot(
    true_jams.numpy(),
    weighted_losses_jam_1,
    label=f"WeightedMSELoss ($\\alpha = {weighted_loss_fn_alpha_1.alpha}$)",
    color="orangered",
    linewidth=2.5,
)
plt.plot(
    true_jams.numpy(),
    weighted_losses_jam_2,
    label=f"WeightedMSELoss ($\\alpha = {weighted_loss_fn_alpha_2.alpha}$)",
    color="blue",
    linewidth=2.5,
)

plt.title("Impact of Jam-Level on Loss (for a fixed error of 1.0)", fontsize=16)
plt.xlabel("True Jam Level ($y_{true}$)", fontsize=12)
plt.ylabel("Calculated Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# %% [markdown]
# ### Observations
#
# - The Standard MSE loss is constant because the squared error $(1.0)^2 = 1.0$ is the same regardless of the true jam level.
# - The `WeightedMSELoss` increases linearly with the true jam level. An error on a road with `jam_level==4` contributes significantly more to the total loss than the same error on a road with `jam_level=0`.

# %% [markdown]
# ## Test 2: Weight of time (fixed jam level)

# %%
T = 12
time_steps = np.arange(T)

weighted_loss_fn_beta_1 = WeightedMSELoss(alpha=1.0, beta=0.1)
weighted_loss_fn_beta_2 = WeightedMSELoss(alpha=2.0, beta=0.2)

# Fixed true value and prediction
y_true_val = 2.0
y_pred_val = 3.0

weighted_losses_time = []
mse_losses_time = []

# We create tensors of shape [1,1,T] and compute the overall loss,
# then show how each component contributes
y_true_series = torch.full((1, 1, T), y_true_val)
y_pred_series = torch.full((1, 1, T), y_pred_val)

# To see the loss contribution of each timestep, we calculate it manually
base_error_sq = (y_pred_val - y_true_val) ** 2  # This is 1.0
w_loss_components_1 = (
    base_error_sq
    * (1 + weighted_loss_fn_beta_1.alpha * y_true_val)
    * (1 + weighted_loss_fn_beta_1.beta * time_steps)
)
w_loss_components_2 = (
    base_error_sq
    * (1 + weighted_loss_fn_beta_2.alpha * y_true_val)
    * (1 + weighted_loss_fn_beta_2.beta * time_steps)
)
m_loss_components = np.full(T, base_error_sq)


# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(
    time_steps,
    m_loss_components,
    label="Standard MSE Loss",
    linestyle="--",
    color="gray",
    marker="o",
)
plt.plot(
    time_steps,
    w_loss_components_1,
    label=f"WeightedMSELoss ($\\beta = {weighted_loss_fn_beta_1.beta}$)",
    color="dodgerblue",
    linewidth=2.5,
    marker="o",
)
plt.plot(
    time_steps,
    w_loss_components_2,
    label=f"WeightedMSELoss ($\\beta = {weighted_loss_fn_beta_2.beta}$)",
    color="green",
    linewidth=2.5,
    marker="o",
)

plt.title("Impact of Time-Step on Loss (for a fixed error)", fontsize=16)
plt.xlabel("Future Time Step ($t$)", fontsize=12)
plt.ylabel("Loss Contribution of that Time Step", fontsize=12)
plt.xticks(time_steps)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

# %% [markdown]
# ### Observations
#
# - With Standard MSE, the loss contributed by each time step is identical, as the error is the same.
# - With WeightedMSELoss, the loss contribution grows linearly for each step into the future. An error at $t=11$ is penalized much more than the same error at $t=0$, pushing the model to learn long-term dependencies.

# %% [markdown]
# ## Test 3: Combined Penalization

# %%
weighted_loss_fn = WeightedMSELoss(alpha=1.0, beta=0.1)

T = 12
time_steps_grid = np.arange(T)
true_jams_grid = np.linspace(0, 4, 5)  # Test at jam levels 0, 1, 2, 3, 4

loss_matrix = np.zeros((len(true_jams_grid), T))

# Fixed error of +1.0
error_sq = 1.0

for i, jam in enumerate(true_jams_grid):
    for j, t in enumerate(time_steps_grid):
        jam_weight = 1 + weighted_loss_fn.alpha * jam
        time_weight = 1 + weighted_loss_fn.beta * t
        loss_matrix[i, j] = error_sq * jam_weight * time_weight

fig, ax = plt.subplots(figsize=(12, 7))
cax = ax.imshow(loss_matrix, cmap="Reds", aspect="auto")

ax.set_xticks(np.arange(T))
ax.set_xticklabels(time_steps_grid)
ax.set_yticks(np.arange(len(true_jams_grid)))
ax.set_yticklabels([f"{j:.1f}" for j in true_jams_grid])

plt.title("Combined Loss Behavior (Heatmap)", fontsize=16)
plt.xlabel("Future Time Step ($t$)", fontsize=12)
plt.ylabel("True Jam Level ($y_{true}$)", fontsize=12)
fig.colorbar(cax, label="Calculated Loss Value")

# Add text annotations
for i in range(len(true_jams_grid)):
    for j in range(T):
        ax.text(
            j,
            i,
            f"{loss_matrix[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=9,
        )

plt.show()

# %% [markdown]
# ### Observations
#
# - The heatmap clearly shows the combined effect. The loss is
#     - lowest at the bottom-left (low jam, near-term forecast), and
#     - highest at the top-right (high jam, long-term forecast).
# - This visualization confirms that the loss function is correctly prioritizing the two conditions we wanted to specify.

# %% [markdown]
# ## Bonus: Focal Loss

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class RegressionFocalLossWithTimeWeighting(nn.Module):
    """
    Custom loss function that adapts the Focal Loss concept for regression tasks
    and combines it with time-step weighting for forecasting.

    The loss is calculated as:
    Loss = focal_weight * time_weight * base_loss

    1.  Base loss is the MSE (y_pred - y_true)^2.
    2.  Time weight penalizes errors in distant future predictions more heavily.
        Calculated as (1 + beta * t).
    3.  Focal weight down-weights the loss for common (easy) target values and
        up-weights the loss for rare (hard) target values.
        Calculated as (1 - P(y_true))^gamma, where P(y_true) is the pre-calculated
        frequency (probability) of the true jam level.
    """

    def __init__(
        self,
        target_dist: torch.Tensor,
        beta: float = 0.05,
        gamma: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initializes the loss function.

        Args:
            target_dist (torch.Tensor): A 1D tensor representing the frequency distribution
                                        of the target variable. The index of the tensor
                                        should correspond to the discretized target value.
                                        For example, target_dist[i] = P(y_true = i * bin_size).
            beta (float): Hyperparameter for time-step weighting. Controls the slope of
                          the linear penalty for future time steps.
            gamma (float): The focusing parameter for the focal loss. Higher values
                           result in stronger down-weighting of common values.
            device (str): The device ('cpu' or 'cuda') where the tensors should be.
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        # Normalize the target distribution to sum to 1
        self.target_dist = (target_dist / target_dist.sum()).to(device)
        self.device = device
        # Pre-calculate focal weights for efficiency: (1 - p)^gamma
        self.focal_weights_map = torch.pow(1 - self.target_dist, self.gamma)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the final loss.

        Args:
            y_pred (torch.Tensor): The model's predictions. Shape: [B, N, T].
            y_true (torch.Tensor): The ground truth values. Shape: [B, N, T].

        Returns:
            torch.Tensor: The final computed loss value.
        """
        assert y_pred.shape == y_true.shape, (
            "Prediction and ground truth must have the same shape"
        )

        base_loss = torch.pow(y_pred - y_true, 2)

        time_steps = torch.arange(
            y_true.shape[2], device=self.device, dtype=y_true.dtype
        )
        time_weights = 1 + self.beta * time_steps

        # Reshape for broadcasting [1, 1, T]
        time_weights = time_weights.view(1, 1, -1)

        # We need to map continuous y_true values to discrete indices of our distribution
        # Assuming y_true values are continuous from 0 to 4, and target_dist has a bin for each integer
        # We round y_true to the nearest integer to get the index
        y_true_indices = torch.round(y_true).long()

        # Clamp indices to be within the valid range of our distribution tensor
        y_true_indices.clamp_(0, len(self.target_dist) - 1)

        # Calculate focal weights based on the rarity of the true values
        # Use the indices to look up the pre-calculated focal weights
        focal_weights = self.focal_weights_map[y_true_indices]

        # Apply both weights to the squared error
        final_loss_matrix = focal_weights * time_weights * base_loss

        loss = torch.mean(final_loss_matrix)
        return loss


# %%
# The data distribution should be calculated from the training dataset
# Here we create a mock dataset that is highly unbalanced:
# 80% zeros, 10% ones, 5% twos, 3% threes, 2% fours
mock_training_data = np.concatenate(
    [np.zeros(8000), np.ones(1000), np.full(500, 2), np.full(300, 3), np.full(200, 4)]
)
np.random.shuffle(mock_training_data)

# Calculate the distribution (frequency of each value from 0 to 4)
# We create 5 bins for values 0, 1, 2, 3, 4 (this would be the round of the actual jam levels)
hist = np.histogram(mock_training_data, bins=np.arange(6) - 0.5, density=True)[0]
target_distribution = torch.from_numpy(hist).float()

print("Target Distribution (Probability of each jam level)")
print(target_distribution)
print("-" * 55)

device = "cpu"
criterion = RegressionFocalLossWithTimeWeighting(
    target_dist=target_distribution, beta=0.1, gamma=2.0, device=device
)

B, N, T = 4, 50, 12

# Mock prediction and ground truth tensors
y_pred_mock = torch.rand(B, N, T, device=device) * 4
y_true_mock = (
    torch.from_numpy(
        np.random.choice([0, 1, 2, 3, 4], size=(B, N, T), p=target_distribution.numpy())
    )
    .float()
    .to(device)
)

loss = criterion(y_pred_mock, y_true_mock)

print(f"Calculated Loss: {loss.item():.4f}")

# %%
fig, ax1 = plt.subplots(figsize=(10, 6))

jam_levels = np.arange(len(target_distribution))
ax1.bar(
    jam_levels,
    target_distribution.cpu().numpy(),
    color="skyblue",
    label="Original Probability P(y_true)",
)
ax1.set_xlabel("Jam Level", fontsize=12)
ax1.set_ylabel("Probability (in Training Set)", color="skyblue", fontsize=12)
ax1.tick_params(axis="y", labelcolor="skyblue")
ax1.set_xticks(jam_levels)

ax2 = ax1.twinx()
ax2.plot(
    jam_levels,
    criterion.focal_weights_map.cpu().numpy(),
    color="orangered",
    marker="o",
    linestyle="--",
    linewidth=2.5,
)
ax2.set_ylabel(
    f"Focal Weight ($\\gamma = {criterion.gamma}$)", color="orangered", fontsize=12
)
ax2.tick_params(axis="y", labelcolor="orangered")

plt.title("Focal Weighting vs. Target Probability", fontsize=16)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Observations
#
# The core idea of this Regression Focal Loss is to down-weight the loss for common `jam_level` values (like 0) and up-weight the loss for rare `jam_level` values (like +3). This requires us to first understand the distribution of the target variable in the training set.
#
# - **Focal Weighting** reduces the loss contribution from frequent `jam_level` values and increases it for rare ones. This is controlled by a `gamma` ($\gamma$) parameter and requires us to provide the distribution of the target variable in the training set. In our case we can round the `jam_level` continuous column and get the discrete bins from that.
# - **Time-Step Weighting** is still applieds to give an increasing penalty to predictions further in the future, encouraging better long-term forecasting. It's controlled by a `beta` ($\beta$) parameter.

# %% [markdown]
#

# %% [markdown]
#
