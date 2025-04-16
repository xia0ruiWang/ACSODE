import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, random_split
# Augmenting dimensions for better learning capacity
mpl.rcParams['text.usetex'] = True
class DynamicODESolver:
    def __init__(self, func, y0, gfunc=None, ufunc=None, u0=None, step_size=None, interp="linear", atol=1e-6, norm=None):
        self.func = func
        self.y0 = y0
        self.gfunc = gfunc
        self.ufunc = ufunc
        self.u0 = u0 if u0 is not None else y0
        self.step_size = step_size
        self.interp = interp
        self.atol = atol
        self.norm = norm

    def _before_integrate(self, t):
        pass

    def _advance(self, next_t):
        t0 = self.t
        y0 = self.y
        u0 = self.u
        dt = next_t - t0
        if self.ufunc is None:
            u1 = u0
            udot = u1
        else:
            udot = self.ufunc(t0, u0)
            u1 = udot * dt + u0
        if self.gfunc is None:
            gu1 = udot
        else:
            gu1 = self.gfunc(t0, udot)
        dy, f0 = self._step_func(t0, dt, next_t, y0, gu1)
        y1 = y0 + dy
        if self.interp == "linear":
            y_next = self._linear_interp(t0, next_t, y0, y1, next_t)
            u_next = self._linear_interp(t0, next_t, u0, u1, next_t)
        elif self.interp == "cubic":
            f1 = self.func(next_t, y1) + self.gfunc(next_t, self.gu)
            y_next = self._cubic_hermite_interp(t0, y0, f0, next_t, y1, f1, next_t)
        else:
            y_next = y1
            u_next = u1
        self.t = next_t
        self.y = y_next
        self.u = u_next
        return y_next

    def integrate(self, t):
        if self.step_size is None:
            self.step_size = t[1] - t[0]
        self.t = t[0]
        self.y = self.y0
        self.u = self.u0
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        t = t.to(self.y0.device, self.y0.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
        return solution

    def _step_func(self, t0, dt, t1, y0, gu1):
        f0 = self.func(t0, y0) + gu1
        dy = f0 * dt
        return dy, f0

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
class ODEFunc(nn.Module):
    def __init__(self,augment_dim=2):
        super(ODEFunc, self).__init__()
        self.fc1 = nn.Linear(2+augment_dim, 32).to(device)  # 2 for the two ODE variables + augment_dim
        self.fc2 = nn.Linear(32, 64).to(device)
        self.fc3 = nn.Linear(64,32).to(device)
        self.fc4 = nn.Linear(32,16)
        self.fc5 = nn.Linear(16, 2+augment_dim ).to(device)  # Output must match the augmented input

    def forward(self, t, y_aug):
        # y_aug is the augmented state: [original 2-dim state, augmented dims]
        out = torch.rrelu(self.fc1(y_aug)).to(device)
        out = torch.rrelu(self.fc2(out)).to(device)
        out = torch.rrelu(self.fc3(out)).to(device)
        out = torch.rrelu(self.fc4(out)).to(device)
        y = self.fc5(out).to(device)
        return y

class ODEFuncG(nn.Module):
    def __init__(self, input_dim, device):
        super(ODEFuncG, self).__init__()
        self.linear = nn.Linear(input_dim + 3, input_dim + 3)
        self.device = device
        self.input_dim = input_dim

    def forward(self, t, y):
        t_vec = torch.ones(y.shape[0], 1).to(self.device) * t
        t_and_y = torch.cat([t_vec, y], 1)
        y = self.linear(t_and_y)[:, :self.input_dim+2]
        return y

# Neural ODE system with augmented state
class CSNeuralODE(nn.Module):
    def __init__(self, func, gfunc, input_dim,augment_dim=2):
        super(CSNeuralODE, self).__init__()
        self.func = func
        self.gfunc = gfunc
        self.input_dim = input_dim-1
        self.augment_dim = augment_dim

    def forward(self, y0, t):
        y0_aug = torch.cat([y0.to(device), torch.zeros(y0.shape[0], self.augment_dim).to(device)], dim=-1).to(device)
        solver = DynamicODESolver(self.func, y0_aug, ufunc=self.gfunc, u0=y0_aug)
        out = solver.integrate(t).permute(1, 0, 2)
        # out = out.view(-1, t.shape[0], self.input_dim)
        return out[..., :2]  # Return only the original 2-variable state (drop augmented dimensions)



# Training the model
def train_anode(model, t, y_true, epochs=1001, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss().to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Predict the trajectory using the ANODE model
        y_pred = model(y_true[0].unsqueeze(0), t).to(device)
        loss = loss_fn(y_pred.squeeze(), y_true).to(device)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch} | Loss: {loss.item()}')

# Predict the trajectory
def predict(model, t, initial_state):
    with torch.no_grad():
        y_pred = model(initial_state.unsqueeze(0), t)
    return y_pred.squeeze()

def plot_trajectories(obs=None, noiseless_traj=None, times=None, trajs=None, save=None, title=None):
        fig = plt.figure()
        mpl.rcParams['axes.unicode_minus'] = False
        font = {'family': 'Times New Roman',
                'weight': 800,
                'size': 18,
                }
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(r'$i_\alpha$[$A$]', fontdict=font)
        ax.set_ylabel(r'$i_\beta$[$A$]', fontdict=font)
        plt.legend(loc='best')
        obsc = obs.cpu()
        z = np.array([o.detach().numpy() for o in obsc])
        ax.plot(z[:, 0], z[:, 1], color='r', alpha=0.5, label='prediction')
        trajsc = trajs.cpu()
        z = np.array([o.detach().numpy() for o in trajsc])
        ax.plot(z[:, 0], z[:, 1], color='b', alpha=0.3, label='target')
        plt.legend(loc='best')
        # time.sleep(0.1)
        '''plt.savefig(r"augcsodefanxiepo.png", format="png")'''
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim=2
    # Generate sample time series data
    data_test = pd.read_csv(r"D:\pycharmproject\Augment node for PMSM\SPMSMfanjieyue500qian.csv", header=None)
    x = np.array(data_test.values)
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    y_true = torch.tensor(x).float().to(device)
    t = torch.linspace(0, 1, 500).to(device)
    # Initialize the Augmented Neural ODE model
    model = CSNeuralODE(func=ODEFunc(augment_dim=2),gfunc=ODEFuncG(2,device),input_dim=2).to(device)
    train_size = int(0.95 * len(y_true))
    y_train = y_true[:train_size]
    t_train = t[:train_size]
    # Train the model
    train_anode(model, t_train, y_train)

    # Make predictions
    initial_state = y_true[0].to(device)
    y_pred = predict(model, t, initial_state).to(device)
    x_test_actual = scaler.inverse_transform(y_true.cpu().numpy())  # Inverse scale to get original values
    predictions = scaler.inverse_transform(y_pred.cpu().numpy())

    # Compute metrics on the entire dataset
    mae = mean_absolute_error(x_test_actual, predictions)
    mse = mean_squared_error(x_test_actual, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(x_test_actual, predictions, multioutput='uniform_average')

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R² Score:", r2)
    plot_trajectories(obs=y_pred,trajs=y_true)

