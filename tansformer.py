import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib as mpl
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib.font_manager import FontProperties
import latex
mpl.rcParams['text.usetex'] = True


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
    '''plt.savefig(r"tansformerzhengxiepo300.png", format="png")'''
    plt.show()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Generate synthetic data
n_samples1 = 500
t = np.linspace(0, 1, n_samples1).reshape(-1, 1)
t_size = int(0.95 * len(t))
t_train,t_test = t[:t_size],t
data_test = pd.read_csv(r"D:\pycharmproject\Augment node for PMSM\SPMSMfanjieyue500qian.csv", header=None)
x = np.array(data_test.values)
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
train_size = int(0.95 * len(x))
x_train, x_test = x[:train_size],x
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test,dtype= torch.float32).to(device)
t_train = torch.tensor(t_train, dtype=torch.float32).to(device)
t_test = torch.tensor(t_test,dtype=torch.float32).to(device)
# Step 1: Define the 3-variable ODE (e.g., Lorenz system)


# Step 3: Prepare the data
input_seq = torch.tensor(x_train[:-1], dtype=torch.float32).to(device)
output_seq = torch.tensor(x_train[1:], dtype=torch.float32).to(device)

dataset = TensorDataset(input_seq, output_seq)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# Step 4: Define the Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.fc_in = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, src):
        src = self.fc_in(src).to(device)  # Initial linear layer to match model_dim
        src = src.unsqueeze(1).to(device)# Add sequence dimension for Transformer
        output = self.transformer_encoder(src).to(device)
        output = output.squeeze(1)  # Remove sequence dimension
        output = self.fc_out(output)  # Final linear layer to match input_dim
        return output


input_dim = 2
model_dim = 16
num_heads = 4
num_layers = 2

model = TransformerModel(input_dim, model_dim, num_heads, num_layers).to(device)

# Step 5: Train the Model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1001

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch} | Loss: {loss.item()}')

# Step 6: Evaluate the Model
# Test on new initial conditions
input_test_seq = torch.tensor(x_test[:-1], dtype=torch.float32)

model.eval()
with torch.no_grad():
    predictions = model(input_test_seq)
x_test =x_test[1:]
plot_trajectories(obs=predictions,trajs=x_test)
# Ensure both 'x' and 'predictions' are aligned
x_test_actual = scaler.inverse_transform(x_test.cpu().numpy())  # Inverse scale to get original values
predictions = scaler.inverse_transform(predictions.cpu().numpy())
# Compute metrics on the entire dataset
mae = mean_absolute_error(x_test_actual, predictions)
mse = mean_squared_error(x_test_actual, predictions)
rmse = sqrt(mse)
r2 = r2_score(x_test_actual, predictions, multioutput='uniform_average')

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)


'''fig = plt.figure()

ax = fig.add_subplot(111,projection = '3d')

# Make data.

data_we =pd.read_csv(r"D:\pycharmproject\second try\we1000.csv", header=None)
we = np.array(data_we.values)
we = torch.from_numpy(we).to(device)
we = we[1:]
obs = torch.cat((we, predictions), 1).to(device)
obsc = obs.cpu()
z = np.array([o.detach().numpy() for o in obsc])
z = np.reshape(z, [-1, 3])
X = z[:, 0]
Y = z[:, 1]
Z = z[:, 2]
# Plot the surface.
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 12,
        }
ax.set_xlabel(' $\omega_m$[rpm]',fontdict=font)
ax.set_ylabel('$i_d$[A]',fontdict=font)
ax.set_zlabel('$i_q$[A]',fontdict=font)
plt.rc('font',family='Times New Roman')

surf = ax.plot_trisurf(X,Y,Z, cmap=cm.coolwarm,

                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_zlim(-1, 1)


ax.zaxis.set_major_locator(LinearLocator(5))

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.

fig.colorbar(surf, shrink=0.5, aspect=5,pad = 0.2)
plt.savefig(r"tansformer.svg", format="svg")
plt.show()'''