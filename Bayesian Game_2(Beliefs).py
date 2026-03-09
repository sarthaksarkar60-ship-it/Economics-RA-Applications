import math
import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
def bern(p):
    if np.random.rand()>p:
        return 0
    else:
        return 1
def random_joint_distribution(n):
    probs = np.random.rand(n)  # random numbers from uniform[0,1)
    return probs / probs.sum()

def bayesian_climate_game(a,n,l):
    b = 0
    param = np.random.rand(5)  # beta, alpha, gamma, pl and ph
    strategy_matrix = np.zeros((a,n))

    while b<a:
       type_matrix = np.zeros((3, n))
       param = np.random.rand(6)
       for i in range(type_matrix.shape[0]):               #Initialising the type matrix 0 = energy type, 1 = political type, 2 = climate type
           for j in range(type_matrix.shape[1]):
                type_matrix[i,j]= bern(0.5)
       S = np.random.rand()
       m = 0                                               #Let it be a partial pooling equlibrium where high political types play lead whereas others play veto
       while m<n:
           lead = 0
           swing = 0
           veto = 0
           k = 0
           bel = np.random.rand(n)
           approx  = np.zeros((l,n))
           for c in range(approx.shape[0]):
               approx[c,:] = bel
               #if approx[c,:4].sum()>=n*(1/2):             #This line assumes that first 4 types are enough to force a coalition
               if bern(bel[m]) == 1 :
                  lead += (-type_matrix[0,m]*(1/param[0])+type_matrix[1,m]-param[3]*type_matrix[2,m]-param[5])
                  swing +=  (
                              -type_matrix[0, m] + param[1]*type_matrix[1, m] - (param[3] * type_matrix[2, m]+param[5]))
                  veto += (
                          (-type_matrix[0, m] * (param[0]) )+ (-type_matrix[1, m]*(1/param[1])) - (param[3]*(1/param[2]) * type_matrix[2, m]))
               else:
                   lead += (
                           (param[0]*type_matrix[0, m])+param[1]*type_matrix[1, m] - (param[4] *param[2]* type_matrix[2, m]))
                   swing += (
                           type_matrix[0, m] - param[1] * type_matrix[1, m] - (
                               param[4] * type_matrix[2, m]))
                   veto += (
                           type_matrix[0, m]  + type_matrix[1, m] - (
                               param[4] * type_matrix[2, m]))
               #bel = [approx[c,i]/approx[c,:].sum() for i in range(7)]
           ap_lead = lead/l
           ap_swing = swing/l
           ap_veto = veto/l
           if ap_lead>= ap_swing and ap_lead>= ap_veto:
               strategy_matrix[b,m] = 2
           elif ap_swing>= ap_lead and ap_swing>=ap_veto:
               strategy_matrix[b, m] = 1
           elif ap_veto>= ap_lead and ap_veto>= ap_swing:
               strategy_matrix[b, m] = 0
           m+=1
       b+=1
    return strategy_matrix,type_matrix
blegh,blegh2 = bayesian_climate_game(100,50,100)
print(blegh)
# 1. Overall strategy frequencies (Histogram)
def plot_strategy_distribution(strategy_matrix):
    plt.figure(figsize=(6, 4))
    plt.hist(strategy_matrix.flatten(), bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.7, color='skyblue')
    plt.xticks([0, 1, 2], labels=['Veto', 'Swing', 'Lead'])
    plt.title("Overall Strategy Frequencies")
    plt.xlabel("Strategy")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_strategy_evolution(strategy_matrix):
    proportions = np.zeros((strategy_matrix.shape[0], 3))  # Rows = rounds, Columns = strategy frequencies

    for i, row in enumerate(strategy_matrix):
        for strat in [0, 1, 2]:
            proportions[i, strat] = np.sum(row == strat) / strategy_matrix.shape[1]

    plt.plot(proportions[:, 0], label='Veto', color='red')
    plt.plot(proportions[:, 1], label='Swing', color='orange')
    plt.plot(proportions[:, 2], label='Lead', color='green')
    plt.title("Strategy Proportions Over Time")
    plt.xlabel("Round")
    plt.ylabel("Proportion")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
# 1. Count coalition formation across rounds
def count_coalitions(strategy_matrix):
    """
    Counts the number of coalition-forming rounds.
    Coalition forms if n_Lead + n_Swing >= n_v
    """
    coalition_formed = []
    for row in strategy_matrix:
        n_L = np.sum(row == 2)
        n_S = np.sum(row == 1)
        n_v = np.sum(row == 0)
        coalition_formed.append(1 if (n_L + n_S) >= n_v else 0)
    return coalition_formed, sum(coalition_formed)

# 2. Plot coalition formation over simulation rounds
def plot_coalition_formation(coalition_list):
    plt.figure(figsize=(8, 3))
    plt.plot(coalition_list, drawstyle='steps-mid', label="Coalition Formed", color='blue')
    plt.xlabel("Round")
    plt.ylabel("Coalition Status")
    plt.title("Coalition Formation Over Rounds")
    plt.yticks([0, 1], labels=["No", "Yes"])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# 3. Plot strategy proportions alongside coalition outcome
def plot_combined(strategy_matrix, coalition_list):
    proportions = np.zeros((strategy_matrix.shape[0], 3))
    for i, row in enumerate(strategy_matrix):
        for strat in [0, 1, 2]:
            proportions[i, strat] = np.sum(row == strat) / strategy_matrix.shape[1]

    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(proportions[:, 0], label='Veto', color='red', linestyle='--')
    ax1.plot(proportions[:, 1], label='Swing', color='orange', linestyle='--')
    ax1.plot(proportions[:, 2], label='Lead', color='green', linestyle='--')
    ax1.set_ylabel("Strategy Proportion")
    ax1.set_xlabel("Round")
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(coalition_list, color='blue', label='Coalition Formed', linewidth=2, alpha=0.6)
    ax2.set_ylabel("Coalition Status")
    ax2.set_yticks([0, 1])

    plt.title("Strategy Proportions and Coalition Formation")
    fig.tight_layout()
    plt.show()


def type_strategy_association_with_graphs(strategy_matrix, type_matrix):
    """
    Associates type realisations with chosen strategies and plots results.

    Parameters:
    strategy_matrix : np.array (a x n)
        Strategy choices for each agent in each run.
    type_matrix : np.array (3 x n)
        Type realisations for each agent (energy, political, climate types).
    """
    assoc_dict = {i: {'lead': 0, 'swing': 0, 'veto': 0} for i in range(8)}

    n_agents = type_matrix.shape[1]

    # Convert types to unique integers 0-7
    type_codes = (type_matrix[0, :].astype(int) << 2) + (type_matrix[1, :].astype(int) << 1) + (
        type_matrix[2, :].astype(int))

    # Count strategy choices for each type
    for run in strategy_matrix:
        for idx in range(n_agents):
            tcode = int(type_codes[idx])
            strat = run[idx]
            if strat == 2:
                assoc_dict[tcode]['lead'] += 1
            elif strat == 1:
                assoc_dict[tcode]['swing'] += 1
            elif strat == 0:
                assoc_dict[tcode]['veto'] += 1

    # Prepare data for plotting
    types = [f"{i:03b}" for i in range(8)]
    leads = [assoc_dict[i]['lead'] for i in range(8)]
    swings = [assoc_dict[i]['swing'] for i in range(8)]
    vetos = [assoc_dict[i]['veto'] for i in range(8)]

    x = np.arange(8)
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, vetos, width=width, color='red', label='Veto (0)')
    plt.bar(x, swings, width=width, color='orange', label='Swing (1)')
    plt.bar(x + width, leads, width=width, color='green', label='Lead (2)')

    plt.xticks(x, types)
    plt.xlabel('Type Realisation (Energy, Political, Climate)')
    plt.ylabel('Total Count Across All Runs')
    plt.title('Strategy Choices by Type')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

    return assoc_dict


plot_strategy_distribution(blegh)
plot_strategy_evolution(blegh)
type_strategy_association_with_graphs(blegh, blegh2)