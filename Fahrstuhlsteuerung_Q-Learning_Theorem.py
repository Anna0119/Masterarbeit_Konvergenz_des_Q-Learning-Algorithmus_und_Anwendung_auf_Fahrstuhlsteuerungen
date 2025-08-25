
### Import wichitger Bibliotheken ###
import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
import itertools
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

        self.update_count = 0

    # Funktion, um die Aktion zu bestimmen, in Abhängigkeit der Q-Tabellen
    def get_action(self, state):
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

        # Lernrate erfüllt die Voraussetzungen des Theorems 5.2.1
        learning_rate = 1/self.update_count

        # Aktualisierung des Q-Wertes für das betrachtete Zustands-Aktions-Paar
        update = self.q_table[state_key_update][action] +learning_rate*(cost - self.q_table[state_key_update][action] + self.q_table[state_key][best_next_action])
        #print(update)
        self.q_table[state_key_update][action] = update


### Funktion, mit der eine Aktion gewählt wird ###
def choose_action(state, agent):
    action = agent.get_action(state)
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

# Alle möglichen Zustände werden in einer Liste zusammengefasst
all_states = []

for call_req in itertools.product([0, 1], repeat=number_floors):
    for elev_pos in range(number_floors):
        for elev_target in itertools.product([0, 1], repeat=number_floors):
            state_key = (call_req, elev_pos, elev_target)
            all_states.append(state_key)

# Initialisierung des Q-Learning-Agenten
agent = QLearningAgent()

### Trainings-Simulation ###
# Implementierung des Algorithmus, der die Voraussetzungen des Theorems 5.2.1 erfüllt (Algorithmus 1 in der Arbeit)

# Initialisierung wichtiger Variablen für die Messung der Distanz und die Beobachtung der Konvergenz der Approximationsfunktion
old_q_table = copy.deepcopy(agent.q_table)
global_delta = []

# Legt fest, wann das Training abgeschlossen wird
continue_simulation = True
epsilon = 0.001 

while continue_simulation == True:
    agent.update_count += 1  # Wichtig für die Definition der Lernrate
    for i in all_states:  # Wählt einen Zustand aus
        first_state = list(i)
        for j in range(0,5):  # Wählt eine Aktion aus
            time = 0
            
            # Initialisiert die Zustandsklasse und die Aufzugsklasse mit dem ausgewählten Zustand
            state = State(first_state[1], list(first_state[0]), list(first_state[0]))
            state.elevator.target = list(first_state[2])
            state.elevator.passengers = sum(state.elevator.target)
            this_state = {'call_requests: ': list(first_state[0]), 'elevator_position: ': first_state[1], 'elevator_target: ': list(first_state[2])}

            # Unterscheidung, ob es sich um einen absorbierenden Zustand handelt; wenn ja: Q-Wert bleibt 0
            if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors)):
                state_key = (tuple(this_state['call_requests: ']), this_state['elevator_position: '], tuple(this_state['elevator_target: ']))
                agent.q_table[state_key][j] = 0.0
            # Wenn nein: Q-Wert wird mithilfe der Funktion in der Q-Agent-Klasse aktualisiert
            else:
                cost, time_new = state.step_to_target(j, time)
                agent.update_q_table(state, this_state, j, cost)
    # Berechnung der Distanz zwischen zwei aufeinanderfolgenden Approximationsfunktionen
    distance = euclidean_q_distance(old_q_table, agent.q_table)
    if(agent.update_count%100 == 0):  # Es wird immer nur die 100. Distanz zur Liste der Distanzen hinzugefügt,
        global_delta.append(distance) # weil der Plot sonst nicht angezeigt werden kann

    # Die neue Approximationsfunktion wird vor der Aktualisierung zwischengespeichert für die Distanzberechnung
    old_q_table = copy.deepcopy(agent.q_table)

    # Prüft, ob festgelegte Abbruchgrenze erreicht wurde
    if distance <= epsilon:
        continue_simulation = False

# Gibt aus, wie lange das Training vom Start des Programms gedauert hat
duration_end_time = datetime.now()
print('Dauer des Trainings: {}'.format(duration_end_time - duration_start_time))

test_state = ((1,0,0,0,1),2,(0,0,0,0,0))
for j in range(0,5):
    print("approximierter Q-Wert für Aktion ", j, agent.q_table[test_state][j])


### Test-Simulationen ###
# Durchführen von 100 Simulationen, in der ein Fahrstuhl fährt, bis ein absorbierender Zustand erreicht wurde

# Initialisierung wichtiger Variablen für die Darstellung der Ergebnisse
iteration_count_testing = []
improper_count = 0

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

    # Führe eine Simulation solange durch, wie Call-Requests oder Passagiere im System sind (maximal aber 10000 Iterationen, im Falle von improper Strategien)
    continue_simulation = True
    i = 0
    time = starting_time

    # Durchführen einer Simulation
    while continue_simulation == True:
        i += 1
        this_state = state.return_state()
        action = choose_action(state, agent)
        cost, time_new = state.step_to_target(action, time)
        time = time_new

        # Abbruch einer Simulation sobald ein absorbierender Zustand erreicht wurde
        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False
    
    # Füge i zur Liste der Schrittanzahlen hinzu, wenn eine proper Strategie verwendet wurde; 
    # wenn nicht, dann würde die Abbildung unleserlich erscheinen
    if(i<max_iterations):
        iteration_count_testing.append(i)
    else:
        improper_count += 1

print('Anzahl verwendeter improper Strategien in den Test-Simulationen: {}'.format(improper_count))


### Darstellen der Ergebnisse in Abbildungen ###

# Erstellen von Histogramm und Boxplot
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


# Abbildung der Entwicklung des euklidischen Abstands zweier aufeinanderfolgender Approximationsfunktionen
y_global_delta = global_delta
x_g_d = [100*x+100 for x in range(len(global_delta))]

fig = plt.figure()
plt.yscale("log")
plt.plot(x_g_d, y_global_delta, label='normierter Abstand zweier Approximationsfunktionen', color = 'blue', alpha = 0.5)
fig.suptitle('Veränderung der Distanz zwischen zwei Approximationsfunktionen')
plt.xlabel('Q-Updates')
plt.ylabel('Norm des Abstandes zwischen zwei Approximationsfunktionen')
plt.legend()
plt.show()
plt.close()
