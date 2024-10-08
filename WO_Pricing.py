import numpy as np

number_of_assets = 10
coeff_matrix = np.array([
    [1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]])

# Décomposition de Cholesky sur la matrice de corrélation
R = np.linalg.cholesky(coeff_matrix)

T = 1 # Maturité de l'option
N = 252 # Fréquence de l'actualisation des prix des sous-jacents, 1 fois par jour ouvré
K = 100 # Prix d'exercice

asset_price = np.full((number_of_assets,N), 100.0) # Initialisation des prix des sous-jacents

volatility = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
r = 0.02
dt = T/N
M = 10000 # Nombre de trajectoires à simuler
final_asset_price = np.zeros((number_of_assets,M))

for i in range(M):
    for t in range(1, N):
        # Simulation de tirages aléatoires suivant la loi normale
        random = np.random.standard_normal(number_of_assets)
        
        # On génére des epsilon corrélés
        epsilon = np.inner(random, R)
       
       # On simule l'évolution du prix de chaque sous-jacents
        for j in range(number_of_assets):
            s = asset_price[j, t-1]
            v = volatility[j]
            eps = epsilon[j]
            asset_price[j,t] = s * np.exp((r - 0.5 * v**2) * dt + v * np.sqrt(dt) * eps)
    
    # On ne garde que le prix final de chaque sous-jacent
    final_asset_price[:,i] = asset_price[:,N-1]

# On calcule le payoff moyen de l'option
best_of_asset = np.max(final_asset_price,axis=0)
payoff = best_of_asset - K
for i in range(M):
    if payoff[i]<0:
        payoff[i]=0

MeanPayoff = np.mean(payoff)

# On calcule le prix moyen de l'option grâce au payoff moyen obtenu
discount_factor = np.exp(-r*T)
option_price = discount_factor * MeanPayoff

print(option_price)