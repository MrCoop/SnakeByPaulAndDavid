# 🐍 Reinforcement Learning für Snake – Projektbericht

Dieses Projekt entstand im Rahmen der Lehrveranstaltung *"Aktuelle Data Science Entwicklungen"* an der Dualen Hochschule Baden-Württemberg (DHBW) im 6. Semester. Ziel war es, einen Reinforcement-Learning-Agenten zu entwickeln, der das klassische Spiel Snake eigenständig und effizient meistern kann. Als Kernalgorithmus kam Deep Q-Learning (DQN) zum Einsatz. Die Snake-Umgebung wurde dafür eigenständig in Python unter Verwendung von `pygame` erstellt.

Die Implementierung des Agenten basiert auf PyTorch. Das neuronale Netz erhält als Input den flachen Zustand der Umgebung und gibt Q-Werte für alle möglichen Aktionen aus. Der Agent entscheidet sich entweder zufällig (mit Wahrscheinlichkeit ε) oder wählt die beste bekannte Aktion aus dem Netzwerk (mit 1–ε). Der Epsilon-Wert wird dabei schrittweise reduziert, um den Lernprozess vom Explorieren hin zum Exploiteren zu steuern.

Um das Training zu beschleunigen, wurde auf eine visuelle Ausgabe während des Trainings verzichtet. Das grafische Rendern mit `pygame` kann optional zugeschaltet werden, z. B. zur Evaluation eines trainierten Modells. Nach Abschluss des Trainings wird das Modell in einer `.pth`-Datei gespeichert und kann in einem separaten Skript (`play_trained.py`) geladen werden, um den Agenten automatisch spielen zu lassen. Dadurch lässt sich das Modell wiederverwenden, ohne das Training erneut durchlaufen zu müssen.

Die Installation aller notwendigen Bibliotheken erfolgt über die Datei `requirements.txt`. Nach erfolgreicher Installation kann das Training mit dem Skript `train_agent.py` gestartet werden. Das trainierte Modell wird anschließend automatisch gespeichert. Zur Visualisierung der Lernkurve dient eine einfache Matplotlib-Grafik, die den Reward-Verlauf über die Episoden hinweg darstellt.

### Projektstruktur

Das Projekt ist in folgende Komponenten unterteilt:

- `src/snake_env.py` enthält die Snake-Umgebung
- `src/agent.py` enthält die DQN-Implementierung
- `train_agent.py` führt den Trainingsprozess aus
- `play_trained.py` lädt ein trainiertes Modell und spielt es ab
- `dqn_snake.pth` ist die gespeicherte Modell-Datei
- `README.md` enthält die Projektdokumentation

### Verwendete Technologien

Für das Projekt kamen folgende Bibliotheken und Technologien zum Einsatz:

- Python 3.10
- PyTorch
- pygame
- numpy
- matplotlib

### Aufgabenverteilung innerhalb der Gruppe

Das Projekt wurde in einer Zweiergruppe von **David** und **Paul** bearbeitet. Die Aufgabenverteilung orientierte sich an individuellen Stärken und Interessen, wurde aber regelmäßig gemeinsam abgestimmt und abgestimmt umgesetzt.

Paul übernahm schwerpunktmäßig die Entwicklung der Snake-Umgebung mit `pygame`, sowie die Strukturierung des Codes (Projektaufbau, Dateistruktur, Speicherpfade). Zusätzlich kümmerte er sich um die Modell-Speicherung, das Render-Handling und die Integration der Trainingsvisualisierung.

David implementierte den DQN-Agenten inklusive Modellarchitektur, Replay Buffer, Epsilon-gesteuerter Aktionswahl und Target-Netzwerk. Außerdem betreute er das Training, die Hyperparameterwahl und die grafische Auswertung der Trainingsperformance. Die Visualisierung des Lernverlaufs und die Ergebnisinterpretation wurden ebenfalls durch ihn umgesetzt.

Beide Gruppenmitglieder haben sich gemeinsam mit der RL-Theorie, der Erstellung des wissenschaftlichen Berichts und dem Schreiben dieser Dokumentation befasst.


### Abschließende Hinweis

Dieses Projekt dient ausschließlich akademischen Zwecken im Rahmen der DHBW-Lehre. Alle verwendeten Bibliotheken sind Open Source.
