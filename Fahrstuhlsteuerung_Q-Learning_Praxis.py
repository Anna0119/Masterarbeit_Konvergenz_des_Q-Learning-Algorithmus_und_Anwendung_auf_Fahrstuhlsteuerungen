
### Import wichitger Bibliotheken ###
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
from datetime import datetime

# Zur Messung der Trainingsdauer
duration_start_time = datetime.now()

### globale Variablen ###
number_floors = 5
number_elevators = 1
lambdas_poisson = np.array([0.02, 0.0003, 0.003, 0.009, 0.01])
starting_time = 0
max_iterations = 10000
time_to_destination = (0, 5, 8, 10, 12) # in Sekunden
elevator_capacity = 5


### Poisson-Prozess zur Generierung der Ankunftszeiten ###
def interarrival_times(Lambda):

    x = rd.uniform(0,1)
    interarrival_time = -(np.log(1-x))/Lambda

    return interarrival_time


def arrival_times(start_time, stop_time):

    arrival_times = list()

    for i in range (0,number_floors):
            
            time = start_time
            interarrival_for_floor = list()
            
            while(time <= stop_time):
                interarrival_time = interarrival_times(lambdas_poisson[i])
                time = time + interarrival_time
                if (time <= stop_time):
                    interarrival_for_floor.append(time)

            arrival_times.append(interarrival_for_floor)
        
    return arrival_times


### Klasse für den Aufzug ###
class Elevator:

    def __init__(self, position = 0):
        self.position = position
        self.target = [0]*number_floors

        self.next_destination = 0
        self.time_to_dest = 0
        self.passengers = 0
        self.count_all_passengers = 0
        
    # Funktion, die die Aktion als nächstes Ziel speichert
    def set_new_destination(self, action):
        self.next_destination = action

    # Funktion, um die Länge der Periode zu berechnen
    def driving_time(self):
        self.time_to_dest = time_to_destination[abs(self.position - self.next_destination)]

    # Funktion, um Passagiere aussteigen zu lassen
    def passengers_out(self):
        count_targets = sum(1 for x in self.target if x!=0)  # Menge der aussteigenden Passagiere ist abhängig von der Anzahl der verbleibenen Ziele
        if self.target[self.position] != 0:
            max_passengers_out = self.passengers - count_targets + 1
            if count_targets == 1: # Wenn es nur ein Ziel gibt, steigen alle Passagiere aus
                self.passengers = 0
            else:
                self.passengers -= rd.randint(1,max_passengers_out) # Wenn es mehrere Ziele gibt, steigt eine zufällige Anzahl an Passagieren aus

        self.target[self.position] = 0

    # Funktion, um Passagiere einsteigen zu lassen
    def passengers_in(self, state):
        # Passagiere steigen zu, wenn es call requests an der aktuellen Position gibt
        if state.call_requests[self.position] != 0:
            count_new_passengers = 0
            # Kapazität eines Aufzugs sind 5 Passagiere
            while self.passengers < 5 and state.arrival_floor[self.position] > 0:
                self.passengers += 1
                state.arrival_floor[self.position] -= 1

                count_new_passengers += 1

            # Wenn neue Passagiere eingestiegen sind:
            if count_new_passengers > 0:
                # Solange keine Passagiere mehr in der Etage warten: Call requests auf 0 setzen
                if state.arrival_floor[self.position] == 0:
                    state.call_requests[self.position] = 0
                
                # Bestimmen die Anzahl neuer Ziele, in Abhängigkeit der Anzahl neuer Passagiere
                number_new_targets = rd.randint(1,count_new_passengers)
                self.count_all_passengers += count_new_passengers # Wichtig für die Bestimmung der relativen Wartezeit am Ende

                # Bestimmung neuer Ziele zufällig
                for i in range (number_new_targets):
                    new_target = rd.randint(0,number_floors-1)

                    while(new_target == self.position):
                        new_target = rd.randint(0,number_floors-1)

                    if (self.target[new_target] != 1):
                        self.target[new_target] = 1

    # Bewegung des Aufzuges
    def move(self, state):
        self.position = self.next_destination
        self.time_to_dest = 0

        self.passengers_out()



### Klasse für den Zustand ###
class State:

    def __init__(self, position = 0, call_requests = [0]*number_floors, arrival_floor = [0]*number_floors):
        self.elevator = Elevator(position)
        self.call_requests = call_requests

        self.num_of_floors = number_floors
        self.num_of_elevators = number_elevators

        self.arrival_floor = arrival_floor

    # Funktion, um den aktuellen Zustand wiederzugeben
    def return_state(self):
        return {'call_requests: ': self.call_requests,
                'elevator_position: ': self.elevator.position,
                'elevator_target: ': self.elevator.target}

    
    # Funktion, für ein Update des Zustandes
    def step_to_target(self, action, time):

        self.elevator.set_new_destination(action)
        self.elevator.driving_time()

        time_new = time + self.elevator.time_to_dest

        # Initialisierung der Kosten
        cost_position = 0
        
        self.elevator.move(self)

        # Einbinden der Neu-Ankömmlinge in der Periode
        new_arrivals_period = arrival_times(time, time_new)
        for i in range(number_floors):
            self.arrival_floor[i] += len(new_arrivals_period[i])
            if len(new_arrivals_period[i])>0 and self.call_requests[i] == 0:
                self.call_requests[i] = 1

        # Neue Passagiere steigen in den Aufzug ein
        self.elevator.passengers_in(self)

        # Berechnung der Kosten und damit der Rückmeldung an den Agenten in Abhängigkeit der gewählten Aktion
        if self.arrival_floor != [0]*number_floors or self.elevator.target != [0]*number_floors or self.call_requests != [0]*number_floors:
            cost_position += 1

        cost = cost_position

        return cost, time_new


### Klasse Q-Learning-Agent ###
class QLearningAgent:

    def __init__(self):
        self.number_actions = number_floors

        self.q_table = defaultdict(lambda: np.full((self.number_actions,), 0.0))

        self.update_count = defaultdict(lambda: np.full((self.number_actions,), 0.0))
        # self.update_count = 0

    # Funktion, um die Aktion zu bestimmen, in Abhängigkeit der Q-Tabellen
    def get_action(self, state, epsilon_greedy):
        x = rd.uniform(0, 1)
        if x < epsilon_greedy:
            best_action = rd.randint(0,4)
        else:
            state_observation = state.return_state()
            state_key = (tuple(state_observation['call_requests: ']), state_observation['elevator_position: '], tuple(state_observation['elevator_target: ']))
            best_action = np.argmin(self.q_table[state_key])
        return best_action

    # Update der Q-Werte
    def update_q_table(self, state, last_state, action, cost):
        # beobachteter neuer Zustand
        next_state = state.return_state()
        state_key = (tuple(next_state['call_requests: ']), next_state['elevator_position: '], tuple(next_state['elevator_target: ']))
        best_next_action = np.argmin(self.q_table[state_key])

        # Zustand, für den der Q-Wert aktualisiert wird
        state_key_update = (tuple(last_state['call_requests: ']), last_state['elevator_position: '], tuple(last_state['elevator_target: ']))

        self.update_count[state_key_update][action] += 1
        learning_rate = 1/self.update_count[state_key_update][action]
        # self.update_count += 1
        # learning_rate = 1/self.update_count

        # Aktualisierung des Q-Wertes für das betrachtete Zustands-Aktions-Paar
        update = self.q_table[state_key_update][action] +learning_rate*(cost - self.q_table[state_key_update][action] + self.q_table[state_key][best_next_action])
        self.q_table[state_key_update][action] = update


### Funktion, mit der eine Aktion gewählt wird ###
def choose_action(state, agent, epsilon_greedy):
    action = agent.get_action(state, epsilon_greedy)
    return action

### Funktion, mit der die euklidische Distanz zweier Approximationsfunktionen berechnet wird ###
def euclidean_q_distance(q_old: defaultdict, q_new: defaultdict):
    all_states = set(q_old.keys()) | set(q_new.keys())
    total = 0.0

    for state in all_states:
        value_old = q_old[state]
        value_new = q_new[state]
        total += sum((a - b) ** 2 for a, b in zip(value_old, value_new))

    return math.sqrt(total)


### Main ###


### Training ###

# Hier wird anstelle der ganzen Approximationsfunktion in jeder Iteration lediglich ein Q-Wert aktualisiert
# Dieses Programm dient der Einordnung der Ergebnisse des Algorithmus (1 in der Arbeit), für den die Konvergenz bewiesen wurde

# Initialisierung des Q-Learning-Agenten
agent = QLearningAgent()

improper_count = 0
improper_count_list = []

old_q_table = copy.deepcopy(agent.q_table)
global_delta = []

continue_simulation_2 = True
epsilon = 0.001
epsilon_greedy = 0.05
k,j =0,0
distance = 1

# Führe k Simulationen aus, um den Q-Learning-Agenten zu trainieren
while continue_simulation_2 == True:
    k += 1
    # Unabhängige Initialisierung des Startzustandes
    call_requests = [0]*number_floors
    arrival_floor = [0]*number_floors
    start_passengers = rd.randint(1,5)
    for i in range(1,start_passengers+1):
        start_passenger_floor = rd.randint(0,4)
        if call_requests[start_passenger_floor] == 0:
            call_requests[start_passenger_floor] = 1
        arrival_floor[start_passenger_floor] += 1
    start_position = rd.randint(0,number_floors-1)
    state = State(start_position, call_requests, arrival_floor)

    # Führe eine Simulation solange durch, wie Call-Requests oder Passagiere im System sind (maximal aber 100 Iterationen, im Falle von improper Strategien)
    continue_simulation = True
    i = 0
    time = starting_time

    # Durchführen einer Simulation
    while continue_simulation == True:
        i += 1
        j += 1
        # epsilon_greedy = 0.05/j
        # this_state = state.return_state()
        this_state = {'call_requests: ': copy.deepcopy(state.call_requests), 'elevator_position: ': copy.deepcopy(state.elevator.position), 'elevator_target: ': copy.deepcopy(state.elevator.target)}
        action = choose_action(state, agent, epsilon_greedy)
        cost, time_new = state.step_to_target(action, time)
        agent.update_q_table(state, this_state, action, cost)
        time = time_new

        # eine Simulation wird beendet, sobald ein absorbierender Zustand erreicht wird
        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False

        # Die Distanz von zwei Approximationsfunktionen wird nach 25.600 einzelner Aktualisierungen Approximationsfunktionen gemessen, um eine Vergleichbarkeit zum Algorithmus 1 (in der Arbeit zu gewährleisten)
        if(j%25600 == 0):
            distance = euclidean_q_distance(old_q_table, agent.q_table)
            global_delta.append(distance)
            old_q_table = copy.deepcopy(agent.q_table)

    # Die Anzahl von verwendeten improper Strategien innerhalb von 1000 Simulationen wird gemessen
    if(i==10000):
        improper_count += 1
    if(k%1000==0):
        improper_count_list.append(improper_count)
        improper_count = 0

    # Abbruchbedingung
    if distance < epsilon:
        continue_simulation_2 = False
    print(distance)

    # print(k)

duration_end_time = datetime.now()
print('\nDauer des Trainings: {}'.format(duration_end_time - duration_start_time))
print('')

test_state = ((1,0,0,0,1),2,(0,0,0,0,0))
for j in range(0,5):
    print("approximierter Q-Wert für Aktion ", j, agent.q_table[test_state][j])


### Test-Simulationen ###

iteration_count_testing = []
for k in range(101):
    # Unabhängige Initialisierung des Startzustandes
    call_requests = [0]*number_floors
    arrival_floor = [0]*number_floors
    start_passengers = rd.randint(1,5)
    for i in range(1,start_passengers+1):
        start_passenger_floor = rd.randint(0,4)
        if call_requests[start_passenger_floor] == 0:
            call_requests[start_passenger_floor] = 1
        arrival_floor[start_passenger_floor] += 1
    start_position = rd.randint(0,number_floors-1)
    state = State(start_position, call_requests, arrival_floor)

    # Führe eine Simulation solange durch, wie Call-Requests oder Passagiere im System sind (maximal aber 100 Iterationen, im Falle von improper Strategien)
    continue_simulation = True
    i = 0
    time = starting_time

    # Durchführen einer Simulation
    while continue_simulation == True:
        i += 1
        this_state = state.return_state()
        action = choose_action(state, agent, 0.0)
        cost, time_new = state.step_to_target(action, time)
        time = time_new

        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False

    if(i<10000):
        iteration_count_testing.append(i)




### Darstellung der Ergebnisse in Abbildungen ###

# Anzahl an improper Strategien in 100 Test Simulationen
y_improper_count = improper_count_list
x_i_c = [x for x in range(len(improper_count_list))]

fig = plt.figure()
plt.plot(x_i_c, y_improper_count, label ='Anzahl >improper< Strategien', color='red', alpha=0.5)
plt.xlabel('Blöcke an 1000 Simulationen')
plt.ylabel('Anzahl an >improper< Ausführungen innerhalb eines Blockes')
plt.legend()
plt.show()
plt.close()

# Boxplot und Histogramm der Schrittanzahlen in 100 Test-Simulationen
y_number_iterations_test = iteration_count_testing
x_n_i_t = [x for x in range(len(iteration_count_testing))]

# Boxplot
fig = plt.figure()
plt.boxplot([y_number_iterations_test], labels=["Boxplot der Schrittanzahl im Test"])
plt.ylabel("Schritte pro Simulation")
plt.title("Performance des Agenten innerhalb von 100 Test-Episoden")
plt.legend()
plt.grid(True)
plt.show()
plt.close()

# Histogramm
fig = plt.figure()
plt.hist(y_number_iterations_test, bins=10)
plt.xlabel("Schritte pro Simulation")
plt.ylabel("Häufigkeit")
plt.title("Verteilung von Schritten pro Simulation im Test")
plt.legend()
plt.grid(True)
plt.show()
plt.close()


# Abbildung der Konvergenz des euklidischen Abstandes zwischen Approximationsfunktionen
y_global_delta = global_delta
x_g_d = [x for x in range(len(global_delta))]

fig = plt.figure()
plt.yscale("log")
plt.plot(x_g_d, y_global_delta, label='normierter Abstand zweier Approximationsfunktionen', color = 'blue', alpha = 0.5)
fig.suptitle('Abstand zweier Approximationsfunktionen nach 25600 einzelnen Updates')
plt.xlabel('Vergleichsinstanzen (25600 Updates pro Instanz)')
plt.ylabel('euklidischer Abstand zwischen zwei Approximationsfunktionen')
plt.legend()
plt.show()
plt.close()