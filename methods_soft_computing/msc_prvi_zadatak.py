import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

CSV_PATH = "poslovi.csv"
data = pd.read_csv(CSV_PATH)

plt.figure()
plt.bar(data.Posao, data['Vreme izvrsavanja'])
plt.title('Poslovi i njihovo vreme izvrsavanja')
plt.xlabel('Poslovi')
plt.ylabel('Vreme (min)')
plt.show()

machines = 3
num_jobs = len(data)
bits_per_job = 2
# svaki hromozom je duzine 32 bita gde se svaka masina kodira sa po dva bita i onda za 
# svaki posao gledamo na kojoj se masini odvija i za sta je kriterijumska funk najmanaj
chromosome_length = num_jobs * bits_per_job

population_size = 320
generations = 250
crossover_rate = 0.7
mutation_rate = 0.01
generation_gap = 0.02


def random_chromosome():
    # nasumicna inicijalizacija duzine 32
    return ''.join(random.choice(['0', '1']) for _ in range(chromosome_length))


def decode(chromosome):
    # dekodira hromozom masina/posao
    assignments = []
    for i in range(0, chromosome_length, bits_per_job):
        bits = chromosome[i:i+bits_per_job]
        value = int(bits, 2)
        machine = value % machines  
        assignments.append(machine)
    return assignments 


def fitness(chromosome):
    # sto brze izvrsavanje svih poslova sa sto manjom razlikom izmedju masina
    assignment = decode(chromosome)
    loads = [0.0] * machines
    for job_idx, machine_idx in enumerate(assignment):
        t = data.iloc[job_idx]['Vreme izvrsavanja']
        loads[machine_idx] += t
    cmax = max(loads)
    balance = np.std(loads)
    score = cmax + balance
    return 1 / (1 + score)


def tournament_selection(pop, fitnesses, k=3):
    # biraju se 3 jedinke iz cele popoulacije i bira se ona sa najvecim fitnesom
    selected = random.sample(list(zip(pop, fitnesses)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]


def crossover(p1, p2):
    # krosover sa 1-point sa nasumicnim izborom pozicije za presecanje hromozoma
    if random.random() > crossover_rate:
        return p1, p2
    point = random.randint(1, chromosome_length - 1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2


def mutate(chromosome):
    bits = list(chromosome)
    for i in range(chromosome_length):
        if random.random() < mutation_rate:
            bits[i] = '1' if bits[i] == '0' else '0'
    return ''.join(bits)


population = [random_chromosome() for _ in range(population_size)]

best_scores = []
avg_scores = []


for gen in range(generations):
    fitnesses = [fitness(ch) for ch in population]
    sorted_pop = [ch for _, ch in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]

    num_survivors = int(population_size * generation_gap)
    survivors = sorted_pop[:num_survivors]

    new_population = survivors.copy()
    while len(new_population) < population_size:
        p1 = tournament_selection(sorted_pop, fitnesses)
        p2 = tournament_selection(sorted_pop, fitnesses)
        c1, c2 = crossover(p1, p2)
        new_population.append(mutate(c1))
        if len(new_population) < population_size:
            new_population.append(mutate(c2))

    population = new_population
    fit = [fitness(ch) for ch in population]
    best_scores.append(max(fit))
    avg_scores.append(np.mean(fit))


final_fitnesses = [fitness(ch) for ch in population]
best_idx = np.argmax(final_fitnesses)
best_chromosome = population[best_idx]
best_assignment = decode(best_chromosome)

loads = [0.0] * machines
machine_jobs = [[] for _ in range(machines)]
for job_idx, machine_idx in enumerate(best_assignment):
    t = data.iloc[job_idx]['Vreme izvrsavanja']
    name = data.iloc[job_idx]['Posao']
    loads[machine_idx] += t
    machine_jobs[machine_idx].append((name, t))

print("\n=== NAJBOLJE REŠENJE ===")
print("Hromozom:", best_chromosome)
for m in range(machines):
    jobs_str = ", ".join(f"{n}({t:g})" for n, t in machine_jobs[m])
    print(f"Mašina {m+1}: {jobs_str} | Ukupno vreme: {loads[m]:.2f}")

print(f"\nCmax: {max(loads):.2f}")
print(f"Std opterećenja: {np.std(loads):.4f}")
print(f"Fitnes: {final_fitnesses[best_idx]:.5f}")


plt.figure(figsize=(8, 5))
plt.plot(best_scores, label='Najbolja jedinka')
plt.plot(avg_scores, label='Prosečan fitnes')
plt.title('Promena fitnesa kroz generacije Gp=0.02')
plt.xlabel('Generacija')
plt.ylabel('Fitnes')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(9, 5))
for m in range(machines):
    start = 0
    for job_name, duration in machine_jobs[m]:
        plt.barh(y=m, width=duration, left=start, height=0.5)
        plt.text(start + duration/2, m, job_name, ha='center', va='center', fontsize=8)
        start += duration

plt.yticks(range(machines), [f"Mašina {i+1}" for i in range(machines)])
plt.xlabel("Vreme (minuti)")
plt.title("Gantt dijagram Gp=0.98")
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


def genetic_algorithm_elitism():
    population = [random_chromosome() for _ in range(population_size)]

    best_scores = []
    avg_scores = []

    for gen in range(generations):
        fitnesses = [fitness(ch) for ch in population]
        # sort population by fitness descending
        sorted_pairs = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        sorted_pop = [ch for ch, _ in sorted_pairs]
        sorted_fit = [f for _, f in sorted_pairs]
#prepisi najbolju jedinku
        elite = sorted_pop[0]
        num_survivors = int(population_size * generation_gap)
        survivors = sorted_pop[:num_survivors]
        new_population = [elite]

        while len(new_population) < population_size:
            p1 = tournament_selection(sorted_pop, fitnesses)
            p2 = tournament_selection(sorted_pop, fitnesses)
            c1, c2 = crossover(p1, p2)
            new_population.append(mutate(c1))
            if len(new_population) < population_size:
                new_population.append(mutate(c2))

        population = new_population
        fit = [fitness(ch) for ch in population]
        best_scores.append(max(fit))
        avg_scores.append(np.mean(fit))

    final_fitnesses = [fitness(ch) for ch in population]
    best_idx = np.argmax(final_fitnesses)
    best_chromosome = population[best_idx]
    best_assignment = decode(best_chromosome)

    loads = [0.0] * machines
    machine_jobs = [[] for _ in range(machines)]
    for job_idx, machine_idx in enumerate(best_assignment):
        t = data.iloc[job_idx]['Vreme izvrsavanja']
        name = data.iloc[job_idx]['Posao']
        loads[machine_idx] += t
        machine_jobs[machine_idx].append((name, t))

    print("\n=== NAJBOLJE REŠENJE (elitism version) ===")
    print("Hromozom:", best_chromosome)
    for m in range(machines):
        jobs_str = ", ".join(f"{n}({t:g})" for n, t in machine_jobs[m])
        print(f"Mašina {m+1}: {jobs_str} | Ukupno vreme: {loads[m]:.2f}")

    print(f"\nCmax: {max(loads):.2f}")
    print(f"Std opterećenja: {np.std(loads):.4f}")
    print(f"Fitnes: {final_fitnesses[best_idx]:.5f}")


    plt.figure(figsize=(8, 5))
    plt.plot(best_scores, label='Najbolja jedinka (elitism)')
    plt.plot(avg_scores, label='Prosečan fitnes (elitism)')
    plt.title('Promena fitnesa kroz generacije Gp=0.02 + elitism')
    plt.xlabel('Generacija')
    plt.ylabel('Fitnes')
    plt.legend()
    plt.grid(True)
    plt.show()
