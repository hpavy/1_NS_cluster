from deepxrte.geometry import Rectangle
import torch 
import torch.nn as nn 
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from model import PINNs
from utils import read_csv, write_csv
from train import train
from pathlib import Path
import time 
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

time_start = time.time()

############# LES VARIABLES ################

folder_result = '4_new_interval'  # le nom du dossier de résultat


torch.manual_seed(42537)

##### Le modèle de résolution de l'équation de la chaleur
nb_itt = 6000      # le nb d'epoch
resample_rate = 3000  # le taux de resampling
display = 500       # le taux d'affichage
poids = [1, 1]   # les poids pour la loss
    
n_data = 5000         # le nb de points initiaux
n_pde = 5000          # le nb de points pour la pde

n_data_test = 5000
n_pde_test  = 5000

L = 0.05
V0 = 1.
Re = 100

lr = 1e-4

##### Le code ###############################
###############################################

# La data
df = pd.read_csv('data.csv')


# On adimensionne la data
df_modified = df[(df['Points:0']>= -0.07) & (df['Points:1']>= -0.1) & (df['Points:1']<= 0.1)]
x, y, t = np.array(df_modified['Points:0']), np.array(df_modified['Points:1']), np.array(df_modified['Time'])
u, v, p = np.array(df_modified['Velocity:0']), np.array(df_modified['Velocity:1']), np.array(df_modified['Pressure'])
x_ad = (x-x.min())/L
y_ad = (y-y.min())/L
t_ad = t*V0/L                 # Nouvel adimensionnement

p_ad = p/(V0**2)
u_ad = u/V0
v_ad = v/V0
print(f"u {u.max()}")

X = np.array([x_ad, y_ad, t_ad], dtype=np.float32).T
U = np.array([u_ad, v_ad, p_ad], dtype=np.float32).T

t_ad_min = t_ad.min()
t_ad_max = t_ad.max()
t_max = t.max()

x_ad_max = x_ad.max()
y_ad_max = y_ad.max()

print(f"t_ad_max:{t_ad_max}, t_ad_min:{t_ad_min}, ")


# On regarde si le dossier existe 
dossier = Path(folder_result)
dossier.mkdir(parents=True, exist_ok=True)

rectangle = Rectangle(x_max = x_ad_max, y_max = y_ad_max,
                      t_min=t_ad_min, t_max=t_ad_max)    # le domaine de résolution


# les points initiaux du train 
# Les points de pde 

### Pour train
points_pde = rectangle.generate_random(n_pde).to(device)   # les points pour la pde
points_data_train = np.random.choice(len(X), n_data, replace=False)
inputs_train_data = torch.from_numpy(X[points_data_train]).requires_grad_().to(device)
outputs_train_data = torch.from_numpy(U[points_data_train]).requires_grad_().to(device)

### Pour test
X_test_pde = rectangle.generate_random(n_pde_test).to(device)
points_coloc_test = np.random.choice(len(X), n_data_test, replace=False)
X_test_data = torch.from_numpy(X[points_coloc_test]).requires_grad_().to(device)
U_test_data = torch.from_numpy(U[points_coloc_test]).requires_grad_().to(device)


# Initialiser le modèle
model = PINNs().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss = nn.MSELoss()

# On plot les print dans un fichier texte 
with open(folder_result+'/print.txt', 'a') as f:
    # On regarde si notre modèle n'existe pas déjà 
    if Path(folder_result+'/model_weights.pth').exists() :
        model.load_state_dict(torch.load(folder_result+'/model_weights.pth'))
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        train_loss = read_csv(folder_result+'/train_loss.csv')['0'].to_list()
        test_loss = read_csv(folder_result+'/test_loss.csv')['0'].to_list()
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")
        
    else : 
        print('Nouveau modèle\n', file=f)
        print('Nouveau modèle\n')
        train_loss = []
        test_loss = []


    ######## On entraine le modèle 
    ###############################################
    train(nb_itt=nb_itt, train_loss=train_loss, test_loss=test_loss, resample_rate=resample_rate,
          display=display, poids=poids, inputs_train_data=inputs_train_data,
          outputs_train_data=outputs_train_data, points_pde=points_pde,
          model=model, loss=loss, optimizer=optimizer, X=X, U=U, n_pde=n_pde, X_test_pde=X_test_pde,
          X_test_data=X_test_data, U_test_data=U_test_data, n_data=n_data, rectangle=rectangle,
          device=device, Re=Re, time_start=time_start, f=f)

    ####### On save le model et les losses
    torch.save(model.state_dict(), folder_result+'/model_weights.pth')
    write_csv(train_loss, folder_result, file_name='/train_loss.csv')
    write_csv(test_loss, folder_result, file_name='/test_loss.csv')
    