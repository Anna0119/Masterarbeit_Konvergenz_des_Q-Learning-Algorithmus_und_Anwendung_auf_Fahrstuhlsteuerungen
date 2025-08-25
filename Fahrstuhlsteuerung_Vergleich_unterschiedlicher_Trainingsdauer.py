
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

    def __init__(self, position=0):
        self.position = position
        self.target = [0]*number_floors

        self.next_destination = 0
        self.time_to_dest = 0
        self.passengers = 0
        self.count_all_passengers = 0
        self.passengers_out_per_step = 0 # Abwandlung: zum Zählen der ausgestiegenen Passagiere
        
    # Funktion, die die Aktion als nächstes Ziel speichert
    def set_new_destination(self, action):
        self.next_destination = action

    # Funktion, um die Länge der Periode zu berechnen
    def driving_time(self):
        self.time_to_dest = time_to_destination[abs(self.position - self.next_destination)]

    # Funktion, um Passagiere aussteigen zu lassen
    def passengers_out(self):
        self.passengers_out_per_step = 0
        count_targets = sum(1 for x in self.target if x!=0)  # Menge der aussteigenden Passagiere ist abhängig von der Anzahl der verbleibenen Ziele
        if self.target[self.position] != 0:
            max_passengers_out = self.passengers - count_targets + 1
            if count_targets == 1: # Wenn es nur ein Ziel gibt, steigen alle Passagiere aus
                self.count_all_passengers += self.passengers
                self.passengers_out_per_step = self.passengers
                self.passengers = 0
            else:
                x = rd.randint(1,max_passengers_out)
                self.passengers -= x # Wenn es mehrere Ziele gibt, steigt eine zufällige Anzahl an Passagieren aus
                self.count_all_passengers += x
                self.passengers_out_per_step = x


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
                # self.count_all_passengers += count_new_passengers # Wichtig für die Bestimmung der relativen Wartezeit am Ende

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


# Initialisierung Q-Learning-Agent
agent = QLearningAgent() 

### Trainings-Simulation ### 
# Implementierung des Algorithmus, der die Voraussetzungen des Theorems 5.2.1 erfüllt (Algorithmus 1 in der Arbeit)

# Im folgenden wird das Training zunächst bis zu einer Abbruchgrenze von epsilon=1 durchgeführt, 
# anschließend wird ein Test mit 100 Simulationen durchgeführt, um zu zählen, wie viele improper Strategien noch verwendet werden. 
# Zuletzt wird eine Simulation durchgeführt, die zählt, wie viele Schritte gebraucht werden, bis 100 Passagiere transportiert wurden
# Anschließend wird das Training fortgesetzt, mit Abbruchgrenze epsilon=0.01
# Das wird auch für Abbruchgrenzen 0.01, 0.001, 0.0001 wiederholt

# Initialisierung wichtiger Variablen für die Messung der Distanz und die Berechnung der Konvergenz der Approximationsfunktion
old_q_table = copy.deepcopy(agent.q_table)
global_delta = []
step_count = []
improper_count = 0
improper_count_list = []

# Legt fest, wann das Training pausiert wird/ abgeschlossen ist
continue_simulation = True
epsilon = 1

while continue_simulation == True:
    agent.update_count += 1  # Wichtig für die Definition der Lernrate
    for i in all_states:  # Wählt einen Zustand aus
        first_state = list(i)
        for j in range(0,5):  # Wählt eine Aktion aus
            time = 0

            # Initialisiert die Zustandsklasse und die Aufzugsklasse mit dem ausgewählten Zustand
            state = State(first_state[1], list(first_state[0]), [0]*number_floors)
            state.elevator.target = list(first_state[2])
            state.arrival_floor = list(first_state[0])
            state.elevator.passengers = sum(state.elevator.target)
            this_state = {'call_requests: ': list(first_state[0]), 'elevator_position: ': first_state[1], 'elevator_target: ': list(first_state[2])}

            # Unterscheidung, ob es sich um einen absorbierenden Zsutand handelt; wenn ja: Q-Wert bleibt 0
            if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors)):
                state_key = (tuple(this_state['call_requests: ']), this_state['elevator_position: '], tuple(this_state['elevator_target: ']))
                agent.q_table[state_key][j] = 0.0
            # Wenn nein: Q-Wert wird mithilfe der Funktion in der Q-Agent-Klasse aktualisiert
            else:
                cost, time_new = state.step_to_target(j, time)
                agent.update_q_table(state, this_state, j, cost)

    # Berechnung der Distanz zwischen zwei aufeinanderfolgenden Approximationsfunktionen
    distance = euclidean_q_distance(old_q_table, agent.q_table)

    # Die neue Approximationsfunktion wird vor der Aktualisierung zwischengespeichert für die Distanzberechnung später
    old_q_table = copy.deepcopy(agent.q_table)

    # Prüft, ob festgelegte Abbruchgrenze erreicht wurde
    if distance <= epsilon:
        continue_simulation = False

step_count.append(agent.update_count)

# Simulation, um die Verwendung von improper Strategien zu überwachen
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

    # Führe eine Simulation solange durch, wie Call-Requests oder Passagiere im System sind (maximal aber 100 Iterationen, im Falle von improper Strategien)
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
        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False

    if(i==max_iterations):
        improper_count += 1

# Simulation, um zu zählen, wie viele Schritte für den Transport von 100 Passagieren benötigt werden
amount_passengers = 0
amount_iterations = 0
while amount_passengers < 100:
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
    # Oder sollte das Ziel von 100 Passagieren innerhalb einer Simulation erreicht werden, dann beende die Simulation
    continue_simulation = True
    i = 0
    time = starting_time

    # Durchführen einer Simulation
    while continue_simulation == True:
        i += 1
        amount_iterations += 1
        #agent.exploration_rate = 0/total_steps
        this_state = state.return_state()
        action = choose_action(state, agent)
        cost, time_new = state.step_to_target(action, time)

        # wurden 100 Passagiere transportiert, wird die Simulation beendet
        amount_passengers += state.elevator.passengers_out_per_step
        if(amount_passengers >= 100):
            continue_simulation = False

        time = time_new
        # andernfalls wird die Simulation abgebrochen, sobald ein absorbierender Zustand erreicht wird
        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False
    
# Training wird weitergeführt: Dieser Wechsel von Training und Test, wird  zwei weitere Male durchgeführt
continue_simulation = True
epsilon = 0.1

while continue_simulation == True:
    agent.update_count += 1
    for i in all_states:
        first_state = list(i)
        for j in range(0,5):
            time = 0
            state = State(first_state[1], list(first_state[0]), [0]*number_floors)
            state.elevator.target = list(first_state[2])
            state.arrival_floor = list(first_state[0])
            state.elevator.passengers = sum(state.elevator.target)
            this_state = {'call_requests: ': list(first_state[0]), 'elevator_position: ': first_state[1], 'elevator_target: ': list(first_state[2])}
            if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors)):
                state_key = (tuple(this_state['call_requests: ']), this_state['elevator_position: '], tuple(this_state['elevator_target: ']))
                agent.q_table[state_key][j] = 0.0

            else:
                cost, time_new = state.step_to_target(j, time)
                agent.update_q_table(state, this_state, j, cost)

    distance = euclidean_q_distance(old_q_table, agent.q_table)
    old_q_table = copy.deepcopy(agent.q_table)
    if distance <= epsilon:
        continue_simulation = False


step_count.append(agent.update_count)

# Wiederholung: Verwendung improper Strategien für epsilon 0.01
improper_count_2 = 0
for k in range(101):
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

    continue_simulation = True
    i = 0
    time = starting_time
    while continue_simulation == True:
        i += 1
        this_state = state.return_state()
        action = choose_action(state, agent)
        cost, time_new = state.step_to_target(action, time)
        time = time_new
        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False

    if(i==max_iterations):
        improper_count_2 += 1


# Wiederholung: Schrittanzahl für 100 Passagiere für epsilon 0.01
amount_passengers_2 = 0
amount_iterations_2 = 0
while amount_passengers_2 < 100:
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

    continue_simulation = True
    i = 0
    time = starting_time
    while continue_simulation == True:
        i += 1
        amount_iterations_2 += 1
        this_state = state.return_state()
        action = choose_action(state, agent)
        cost, time_new = state.step_to_target(action, time)
        amount_passengers_2 += state.elevator.passengers_out_per_step
        if(amount_passengers_2 >= 100):
            continue_simulation = False
        time = time_new
        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False
    

# Wiederholung: Training wird wieder gestartet
continue_simulation = True
epsilon = 0.01

while continue_simulation == True:
    agent.update_count += 1
    for i in all_states:
        first_state = list(i)
        for j in range(0,5):
            time = 0
            state = State(first_state[1], list(first_state[0]), [0]*number_floors)
            state.elevator.target = list(first_state[2])
            state.arrival_floor = list(first_state[0])
            state.elevator.passengers = sum(state.elevator.target)
            this_state = {'call_requests: ': list(first_state[0]), 'elevator_position: ': first_state[1], 'elevator_target: ': list(first_state[2])}
            if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors)):
                state_key = (tuple(this_state['call_requests: ']), this_state['elevator_position: '], tuple(this_state['elevator_target: ']))
                agent.q_table[state_key][j] = 0.0

            else:
                cost, time_new = state.step_to_target(j, time)
                agent.update_q_table(state, this_state, j, cost)

    distance = euclidean_q_distance(old_q_table, agent.q_table)
    old_q_table = copy.deepcopy(agent.q_table)
    if distance <= epsilon:
        continue_simulation = False

step_count.append(agent.update_count)

# Wiederholung: Verwendung improper Strategien für epsilon 0.001
improper_count_3 = 0
for k in range(101):
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

    continue_simulation = True
    i = 0
    time = starting_time

    while continue_simulation == True:
        i += 1
        this_state = state.return_state()
        action = choose_action(state, agent)
        cost, time_new = state.step_to_target(action, time)
        time = time_new
        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False

    if(i==max_iterations):
        improper_count_3 += 1


# Wiederholung: Schrittanzahl für 100 Passagiere für epsilon 0.001
amount_passengers_3 = 0
amount_iterations_3 = 0
while amount_passengers_3 < 100:
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

    continue_simulation = True
    i = 0
    time = starting_time
    
    while continue_simulation == True:
        i += 1
        amount_iterations_3 += 1
        this_state = state.return_state()
        action = choose_action(state, agent)
        cost, time_new = state.step_to_target(action, time)
        amount_passengers_3 += state.elevator.passengers_out_per_step
        if(amount_passengers_3 >= 100):
            continue_simulation = False
        time = time_new
        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False
    

# Wiederholung: Training wird wieder gestartet
continue_simulation = True
epsilon = 0.001

while continue_simulation == True:
    agent.update_count += 1
    for i in all_states:
        first_state = list(i)
        for j in range(0,5):
            time = 0
            state = State(first_state[1], list(first_state[0]), [0]*number_floors)
            state.elevator.target = list(first_state[2])
            state.arrival_floor = list(first_state[0])
            state.elevator.passengers = sum(state.elevator.target)
            this_state = {'call_requests: ': list(first_state[0]), 'elevator_position: ': first_state[1], 'elevator_target: ': list(first_state[2])}
            if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors)):
                state_key = (tuple(this_state['call_requests: ']), this_state['elevator_position: '], tuple(this_state['elevator_target: ']))
                agent.q_table[state_key][j] = 0.0

            else:
                cost, time_new = state.step_to_target(j, time)
                agent.update_q_table(state, this_state, j, cost)

    distance = euclidean_q_distance(old_q_table, agent.q_table)
    old_q_table = copy.deepcopy(agent.q_table)
    if distance <= epsilon:
        continue_simulation = False

step_count.append(agent.update_count)

# Wiederholung: Verwendung improper Strategien für epsilon 0.0001
improper_count_4 = 0
for k in range(101):
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
    
    continue_simulation = True
    i = 0
    time = starting_time

    while continue_simulation == True:
        i += 1
        this_state = state.return_state()
        action = choose_action(state, agent)
        cost, time_new = state.step_to_target(action, time)
        time = time_new
        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False

    if(i==max_iterations):
        improper_count_4 += 1


# Wiederholung: Schrittanzahl für 100 Passagiere für epsilon 0.0001
amount_passengers_4 = 0
amount_iterations_4 = 0
while amount_passengers_4 < 100:
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

    continue_simulation = True
    i = 0
    time = starting_time

    while continue_simulation == True:
        i += 1
        amount_iterations_4 += 1
        this_state = state.return_state()
        action = choose_action(state, agent)
        cost, time_new = state.step_to_target(action, time)
        amount_passengers_4 += state.elevator.passengers_out_per_step
        if(amount_passengers_4 >= 100):
            continue_simulation = False
        time = time_new

        if ((state.call_requests == [0]*number_floors and state.elevator.target == [0]*number_floors) or i == max_iterations):
            continue_simulation = False
    
print("\n Ergebnisse: \n")

y_werte = [improper_count, improper_count_2, improper_count_3, improper_count_4]
print("Anzahl verwendeter improper Strategien: {}".format(y_werte))

y_werte_2 = [amount_iterations, amount_iterations_2, amount_iterations_3, amount_iterations_4]
print("Anzahl verwendeter Schritte um 100 Passagiere zu transportieren: {}".format(y_werte_2))

print("Anzahl benötigter Updates für die betrachteten Abbruchgrenzen: {}".format(step_count))