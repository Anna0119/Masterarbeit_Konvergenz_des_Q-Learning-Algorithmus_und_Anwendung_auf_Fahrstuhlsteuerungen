# Masterarbeit_Konvergenz_des_Q-Learning-Algorithmus_und_Anwendung_auf_Fahrstuhlsteuerungen

Für die dargestellten Programme wurde python Version 3.11.5 verwendet.

## Programme:

### Fahrstuhlsteuerung_Q-Learning_Theorem.py :

Dieses Programm formuliert die Zustands-Klasse, die Aufzugs-Klasse und die Klasse des Q-Learning Agenten. 
Für vergleichbare Ergebnisse, wurde eine analoge Formulierung für alle anderen Programme verwendet.

Das Programm implementiert den Q-Learning-Algorithmus, wie er in der Masterarbeit in Algorithmus 1 dargestellt ist.
Dabei wird in jedem Schritt die Q-Funktion vollständig aktualisiert.

Das System wurde unter Berücksichtigungen der Voraussetzungen des Konvergenztheorems modelliert.

Präsentiert wird:
* Die Konvergenz der Q-Funktion, da der euklidische Abstand zwischen aufeinanderfolgenden Q-Funktionen gegen 0 konvergiert
* Die Ergebnisse der erzeugten Strategie (mittels der benötigten Schrittanzahl); Boxplot und Histogramm
* Die Dauer des Trainings

### Fahrstuhlsteuerung_Vergleich_Parametersensitivität_lambda.py :

Dieses Programm vergleicht den Einfluss verschiedener Ankunftsraten auf die Konvergenz der Q-Funktion und damit des Q-Learning-Algorithmus.

Dazu werden drei verschiedene Q-Learning-Agenten mit unterschiedlichen Ankunftsraten trainiert und verglichen.

Verglichen wird:
* Die Konvergenz der Q-Funktionen aller drei Agenten
* Die Güte der Strategien (mittels Schrittanzahlen)

### Fahrstuhlsteuerung_Vergleich_unterschiedlicher_Trainingsdauer.py :

Dieses Programm beschäftigt sich mit der Entwicklung der Güte der Strategie während die Q-Funktion konvergiert.

Dazu wird der Algorithmus nach unterschiedlich langem Training pausiert. Zunächst werden 100 Test-Simulationen durchgeführt, um zu überprüfen, wie viele improper Strategien verwendet werden. Anschließend wird die Anzahl der Schritte gemessen, die die jeweils aktuelle Strategie benötigt, um 100 Passagiere zu transportieren.

Ausgegeben wird:
* Eine Liste mit der Anzahl verwendeter improper Strategien nach unterschiedlicher Trainingslänge
* Eine Liste mit der Anzahl benötigter Schritte um 100 Passagiere zu befördern nach unterschiedlicher Trainingslänge
* Eine Liste mit der Anzahl benötigter Iterationen des Algorithmus, um verschiedene Abbruchgrenzen zu erreichen

### Fahrstuhlsteuerung_Q-Learning_Praxis.py :

Dieses Programm konzentriert sich auf die Darstellung eines Q-Learning-Algorithmus, der in jeder Iteration lediglich einen Q-Wert anstelle der ganzen Q-Funktion aktualisiert.

Dieser dient dem Vergleich zum ersten Algorithmus.

Präsentiert wird:
* Die Trainingsdauer, die dieser Algorithmus vergleichsweise benötigt.
* Die Konvergenz der Q-Funktion mittels euklidischer Distanz (hier nach der gleichen Anzahl Schritten, die im ersten Algorithmus in einer Iteration durchgeführt werden)
* Die Ergebnisse der erzeugten Strategie (mittels der benötigten Schrittanzahl); Boxplot und Histogramm
