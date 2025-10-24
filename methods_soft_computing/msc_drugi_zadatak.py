import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def f1(x, y):
    return x / (np.sin(2 * y) + 1.4)

def f2(x, y):
    return x + (x - 2) * y ** 2

x_vals = np.linspace(0, 5, 100)
y_vals = np.linspace(0, 5, 100)
xx, yy = np.meshgrid(x_vals, y_vals)

Z1 = f1(xx, yy)
Z2 = f2(xx, yy)

#original
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(xx, yy, Z1)
ax1.set_title("Originalna funkcija f1(x,y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f1(x,y)")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(xx, yy, Z2)
ax2.set_title("Originalna funkcija f2(x,y)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f2(x,y)")

plt.tight_layout()
plt.show()

#train
n_train = 1000
x_train = np.random.uniform(0, 5, (n_train, 1))
y_train = np.random.uniform(0, 5, (n_train, 1))
X_train = np.hstack([x_train, y_train])
y1_train = f1(x_train, y_train)
y2_train = f2(x_train, y_train)

#tets 40x40
x_test = np.linspace(0, 5, 40)
y_test = np.linspace(0, 5, 40)
xx_test, yy_test = np.meshgrid(x_test, y_test)
X_test = np.vstack([xx_test.ravel(), yy_test.ravel()]).T

y1_test = f1(X_test[:, 0], X_test[:, 1])
y2_test = f2(X_test[:, 0], X_test[:, 1])


def build_model(n_hidden_layers, total_neurons):
#ukupna broj neurona podeljnen sa brojem slojeva
#poslednjem sloju se dodaje visak pri deljenju
    base_neurons = total_neurons // n_hidden_layers
    remainder = total_neurons % n_hidden_layers
    model = keras.Sequential()
    model.add(layers.Input(shape=(2,)))
    
    layer_structure = []
    for i in range(n_hidden_layers):
        n_units = base_neurons + (remainder if i == n_hidden_layers - 1 else 0)
        layer_structure.append(n_units)
        #aktivaciona je tan hiperbolicki zbog nelineranosti i konvergencije
        model.add(layers.Dense(n_units, activation='tanh'))

    model.add(layers.Dense(1)) 
    model.compile(optimizer='adam', loss='mse')


    #print(f"  -> Struktura slojeva: {layer_structure} (ukupno {sum(layer_structure)} neurona)")
    return model



configs = [(1, 20), (2, 20), (3, 20), (1, 35), (2, 35), (3, 35)]
results = []

for n_layers, n_neurons in configs:
    print(f"\nTreniranje: {n_layers} slojeva, {n_neurons} neurona ukupno")

    # f1
    model1 = build_model(n_layers, n_neurons)
    model1.fit(X_train, y1_train, epochs=200, verbose=0)
    y1_train_pred = model1.predict(X_train)
    y1_test_pred = model1.predict(X_test)

    mse_train_1 = mean_squared_error(y1_train, y1_train_pred)
    mae_train_1 = mean_absolute_error(y1_train, y1_train_pred)
    mse_test_1 = mean_squared_error(y1_test, y1_test_pred)
    mae_test_1 = mean_absolute_error(y1_test, y1_test_pred)

    # f2
    model2 = build_model(n_layers, n_neurons)
    model2.fit(X_train, y2_train, epochs=200, verbose=0)
    y2_train_pred = model2.predict(X_train)
    y2_test_pred = model2.predict(X_test)

    mse_train_2 = mean_squared_error(y2_train, y2_train_pred)
    mae_train_2 = mean_absolute_error(y2_train, y2_train_pred)
    mse_test_2 = mean_squared_error(y2_test, y2_test_pred)
    mae_test_2 = mean_absolute_error(y2_test, y2_test_pred)

    results.append({
        "Slojevi": n_layers,
        "Neuroni": n_neurons,
        "f1_MSE_train": mse_train_1, "f1_MAE_train": mae_train_1,
        "f1_MSE_test": mse_test_1, "f1_MAE_test": mae_test_1,
        "f2_MSE_train": mse_train_2, "f2_MAE_train": mae_train_2,
        "f2_MSE_test": mse_test_2, "f2_MAE_test": mae_test_2
    })

df = pd.DataFrame(results)
#print("\nRezultati metrika (MSE i MAE):")
#print(df.round(6))

#best one
df["TotalError_f1"] = df["f1_MSE_test"] + df["f1_MAE_test"]
df["TotalError_f2"] = df["f2_MSE_test"] + df["f2_MAE_test"]

best_f1 = df.loc[df["TotalError_f1"].idxmin()]
best_f2 = df.loc[df["TotalError_f2"].idxmin()]

print("\nNajbolji model za f1:", best_f1[["Slojevi", "Neuroni"]].to_dict())
print("Najbolji model za f2:", best_f2[["Slojevi", "Neuroni"]].to_dict())


#f1
best_model_f1 = build_model(int(best_f1["Slojevi"]), int(best_f1["Neuroni"]))
best_model_f1.fit(X_train, y1_train, epochs=300, verbose=0)
Z1_pred = best_model_f1.predict(X_test).reshape(xx_test.shape)
Z1_true = f1(xx_test, yy_test)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.plot_surface(xx_test, yy_test, Z1_true, cmap='viridis')
ax1.set_title("Originalna funkcija f1(x,y)")
ax2.plot_surface(xx_test, yy_test, Z1_pred,  cmap='viridis')
ax2.set_title("Estimirana funkcija f1(x,y)")
plt.show()

#f2
best_model_f2 = build_model(int(best_f2["Slojevi"]), int(best_f2["Neuroni"]))
best_model_f2.fit(X_train, y2_train, epochs=300, verbose=0)
Z2_pred = best_model_f2.predict(X_test).reshape(xx_test.shape)
Z2_true = f2(xx_test, yy_test)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.plot_surface(xx_test, yy_test, Z2_true,  cmap='viridis')
ax1.set_title("Originalna funkcija f2(x,y)")
ax2.plot_surface(xx_test, yy_test, Z2_pred,  cmap='viridis')
ax2.set_title("Estimirana funkcija f2(x,y)")
plt.show()

#tabela
def print_clean_2x3(df, func):
    # napravi 2×3 matrice za MSE i MAE (TEST)
    mse_tab = (df.pivot(index="Neuroni", columns="Slojevi", values=f"{func}_MSE_test")
                 .reindex(index=[20, 35], columns=[1, 2, 3])).round(6)
    mae_tab = (df.pivot(index="Neuroni", columns="Slojevi", values=f"{func}_MAE_test")
                 .reindex(index=[20, 35], columns=[1, 2, 3])).round(6)

    print(f"\n{func.upper()} — TEST (2×3)  [redovi: Neuroni 20/35; kolone: Slojevi 1/2/3]")
    header = "Neuroni |   1           2           3"
    print(header)
    print("-"*len(header))
    for neu in [20, 35]:
        mserow = "  ".join(f"{mse_tab.loc[neu, c]:>10.6f}" for c in [1,2,3])
        maerow = "  ".join(f"{mae_tab.loc[neu, c]:>10.6f}" for c in [1,2,3])
        print(f"{neu:<7} | MSE {mserow}")
        print(f"{'':7} | MAE {maerow}")

print_clean_2x3(df, "f1")
print_clean_2x3(df, "f2")

