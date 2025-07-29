import numpy as np
import matplotlib.pyplot as plt

def plot_data(found_sets, func):
    class_0 = np.array([(1, 2), (2, 3), (1.5, 2.5), (3, 3), (2.5, 4), #aprašomi taškai
                         (3.5, 2.5), (4, 4.5), (2, 2), (3, 2), (4, 3)])
    class_1 = np.array([(6, 7), (7, 6), (6.5, 6.5), (8, 7), (7.5, 8),
                         (8.5, 6.5), (9, 9.5), (7, 7), (8, 6), (9, 8)])
    
    # Nupiešiami taškai
    plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Klasė 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Klasė 1')
    
    # pagal aprašytus taškus randamos ribos (vaizdavimo patogumui)
    all_points = np.concatenate((class_0, class_1), axis=0)
    x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
    y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1
    
    colors = ['green', 'purple', 'orange'] #spalvos tiesėms ir jų svorių vektoriams
    for i, (w1, w2, b) in enumerate(found_sets):
        # tiesės piešiamos pasitelkiant 100 taškų ant x ašies tarp ribų
        x_vals = np.linspace(x_min, x_max, 100)
        # y reikšmės randamos pagal formulę ir sugeneruotus x taškus
        y_vals = -(w1 * x_vals + b) / w2
        
        # nupiešiama tiesė
        plt.plot(x_vals, y_vals, color=colors[i], label=f'{func} tiesė {i+1}')
        
        # nupiešti svorio vektoriui pasirenkama x reikšmė tarp ribų
        x0 = np.random.uniform(x_min, x_max)  
        y0 = -(w1 * x0 + b) / w2 # randama atitinkama y reikšmė
        
        # patikrinama ar x ir y bus matomos grafike. Jeigu ne, ieškomos kitos reikšmės.
        if y0 < y_min or y0 > y_max:
            y0 = np.random.uniform(y_min, y_max)
            x0 = (w2 * y0 + b) / (-w1) 
        
        # Svorio vektorius normalizuojamas dėl estetikos
        norm = np.sqrt(w1**2 + w2**2)
        w1_norm, w2_norm = w1 / norm, w2 / norm
        
        # nupiešiamas vektorius
        plt.arrow(x0, y0, w1_norm, w2_norm, color=colors[i], head_width=0.3, head_length=0.3, length_includes_head=True,
                  label=f'{i+1} tiesės svorio vektorius')
    
    plt.title(f'Naudojama {func} funkcija')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

# aktyvacijos funkcija, priema funkcijos tipą ir a reikšmę, išveda 1 arba 0
def activation(value, func='step'):
    if func == 'step':
        return 1 if value >= 0 else 0
    elif func == 'sigmoid':
        return round(1 / (1 + np.exp(-value)))
# neuronas, priimantis reikšmes reikalingas a apskaičiuoti ir funkcijos tipą aktyvacijai.
def neuron(x, y, w1, w2, b, func='step'):
    a = x * w1 + y * w2 + b
    return activation(a, func)
# funkcija apskaičiuoti svoriams
def find_weights(func='step'):
    
    if func not in ['step', 'sigmoid']:
        raise ValueError(f"Invalid activation function '{func}'. Expected 'step' or 'sigmoid'.")
    
    class_0 = [(1, 2), (2, 3), (1.5, 2.5), (3, 3), (2.5, 4),# taškai
               (3.5, 2.5), (4, 4.5), (2, 2), (3, 2), (4, 3)]
    class_1 = [(6, 7), (7, 6), (6.5, 6.5), (8, 7), (7.5, 8),
               (8.5, 6.5), (9, 9.5), (7, 7), (8, 6), (9, 8)]
    
    found_sets = [] # rasti skaičių komplektai
    while len(found_sets) < 3:# ieškoma kol randami trys komplektai
        w1, w2, b = np.random.uniform(-5, 5, 3)# reikšmės ieškomos šiame intervale
        
        correct = True
        # Jeigu svoriai ir poslinkis netinka visiems taškams klasifikuoti, ieškomos kitos reikšmės.
        for x, y in class_0:
            if neuron(x, y, w1, w2, b, func) != 0:
                correct = False
                break
        for x, y in class_1:
            if neuron(x, y, w1, w2, b, func) != 1:
                correct = False
                break
        
        if correct:# jeigu reikšmės tinka, informacija išvedama 
            found_sets.append((w1, w2, b))
            print(f'Found weights ({func} activation): w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}')
    
    return found_sets
        
if __name__ == "__main__":
    print("Using step function:")
    step_weights = find_weights('step')
    plot_data(step_weights, 'slenkstinė')
    print("Using sigmoid function:")
    sigmoid_weights = find_weights('sigmoid')
    plot_data(sigmoid_weights, 'sigmoidinė')
