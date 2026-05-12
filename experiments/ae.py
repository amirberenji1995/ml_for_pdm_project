import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod

# --- Base Abstract Class ---


class BaseAutoencoder(nn.Module, ABC):
    """
    Abstract Base Class for Autoencoders to handle training,
    history tracking, and reconstruction utilities.
    """

    def __init__(self, lr=1e-3, device="cuda:0"):
        super(BaseAutoencoder, self).__init__()
        self.lr = lr
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": []}

    @abstractmethod
    def forward(self, x):
        pass

    def encode_data(self, x):
        """Standardized encoding method for external use."""
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.forward_encoder(x)

    @abstractmethod
    def forward_encoder(self, x):
        """Subclasses must implement their specific encoding logic here."""
        pass

    def fit(self, x_data, epochs=50, batch_size=32, lr=None, val_split=0.2):
        current_lr = lr if lr is not None else self.lr
        optimizer = optim.Adam(self.parameters(), lr=current_lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(x_data, x_data)
        val_size = int(len(dataset) * val_split)
        train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            self.train()
            t_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                output = self(batch_x)

                # Ensure output matches input dimensions (for interpolation-based AE)
                if output.shape != batch_x.shape:
                    output = torch.nn.functional.interpolate(
                        output, size=batch_x.shape[2]
                    )

                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()

            # Validation
            self.eval()
            v_loss = 0
            with torch.no_grad():
                for batch_v, _ in val_loader:
                    batch_v = batch_v.to(self.device)
                    v_out = self(batch_v)
                    if v_out.shape != batch_v.shape:
                        v_out = torch.nn.functional.interpolate(
                            v_out, size=batch_v.shape[2]
                        )
                    v_loss += criterion(v_out, batch_v).item()

            avg_t = t_loss / len(train_loader)
            avg_v = v_loss / len(val_loader) if val_size > 0 else 0
            self.history["train_loss"].append(avg_t)
            self.history["val_loss"].append(avg_v)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Train: {avg_t:.6f} | Val: {avg_v:.6f}"
                )

    def plot_history(self, title="Training History"):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss", linestyle="--")
        plt.title(title)
        plt.ylabel("MSE")
        plt.xlabel("Epochs")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def get_reconstruction_errors(self, x_data):
        self.eval()
        errors = []
        loader = DataLoader(TensorDataset(x_data), batch_size=32, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(self.device)
                reconstructed = self(batch_x)
                if reconstructed.shape != batch_x.shape:
                    reconstructed = torch.nn.functional.interpolate(
                        reconstructed, size=batch_x.shape[2]
                    )

                # Calculate MSE per sample
                per_sample_mse = torch.mean(
                    (batch_x - reconstructed) ** 2, dim=list(range(1, batch_x.ndim))
                )
                errors.extend(per_sample_mse.cpu().numpy())
        return np.array(errors)


# --- Implementations ---


class WideConvAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        input_channels=3,
        input_length=1000,
        kernel_size=25,
        stride=5,
        hidden_channels=30,
        pooling_dim=32,
        latent_dim=50,
        lr=1e-3,
        device="cuda:0",
    ):
        super().__init__(lr=lr, device=device)

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(
                input_channels,
                hidden_channels,
                kernel_size,
                stride,
                padding=kernel_size // 2,
            ),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(pooling_dim),
        )
        self.encoder_fc = nn.Sequential(nn.Flatten(), nn.LazyLinear(latent_dim))

        # Determine spatial size for reconstruction
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_length)
            self.conv_output_len = self.encoder_conv[0](dummy).shape[2]

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_channels * pooling_dim)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (hidden_channels, pooling_dim)),
            nn.Upsample(size=self.conv_output_len),
            nn.ConvTranspose1d(
                hidden_channels,
                input_channels,
                kernel_size,
                stride,
                padding=kernel_size // 2,
                output_padding=stride - 1,
            ),
        )
        self.to(self.device)

    def forward_encoder(self, x):
        return self.encoder_fc(self.encoder_conv(x))

    def forward(self, x):
        z = self.forward_encoder(x)
        x_hat = self.decoder_fc(z)
        return self.decoder_conv(x_hat)


class DnnAutoencoder(BaseAutoencoder):
    def __init__(self, input_dim=500, latent_dim=25, lr=1e-3, device="cuda:0"):
        super().__init__(lr=lr, device=device)

        # We assume 1D input but if it comes in as (B, C, L), we need to handle it
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 400),
            nn.Tanh(),
            nn.Linear(400, 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Linear(100, latent_dim),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 200),
            nn.Tanh(),
            nn.Linear(200, 300),
            nn.Tanh(),
            nn.Linear(300, 400),
            nn.Tanh(),
            nn.Linear(400, input_dim),
            nn.Tanh(),
        )
        self.to(self.device)

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        # Reshape back to original if input was (Batch, Channels, Length)
        return out
