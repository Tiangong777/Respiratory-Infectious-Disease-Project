<img src="misc/images/header.png" width="1100" height="140">

Dieser Code enthält ein Modell zur Klassifizierung infektiöser Atemwegserkrankungen auf Basis angegebener Symptome des 
Patienten. Dieser Code ist entstanden im Rahmen des Andwendungsfall "Atemwegsinfektion" im LeMeDaRT Projekt.

Bei Fragen, bitte eine Mail an [Marcus Buchwald](mailto:marcus.buchwald@medma.uni-heidelberg.de).

# Table of contents
1. [Daten](#Daten)
2. [Modelldetails](#Modelldetails)
3. [Prädiktion](#Prädiktion)
4. [Inferenz](#Inferenz)
5. [Output](#Output)
6. [Pip-Installierung](#Pip-Installierung)
7. [Docker-Installierung](#Docker-Installierung)
8. [TO-DOs](#TO-DOs)


--------------------- 

## Daten <a name="Daten"></a>

Die Daten (`\src\data\inf_dis_v2.csv`) basieren auf einem simulierten Datensatz welcher (Kontaktperson: Bern Genser - bernd.genser@high5data.de). 
Der Datensatz enthält 8 Differentialdiagnosen, mit jeweils 1000 Patienten pro Differentialdiagnose. 
Pro Patient wurden 14 Tage simulierte Symptome (Skala 0-5) generiert auf Basis von ärztlichen Erfahrungswerten. 
Die Differentialdiagnosen umfangen: 

<img src="misc/images/diseases.png" width="500" height="200">

Die 25 berücksichtigten Symptome umfassen (die rot unterlegten Symptome wurden entfernt): 

<img src="misc/images/symptoms.png" width="600" height="350">

--------------------- 


## Details zum Modell / Training <a name="Modelldetails"></a>

### Vorverarbeitung der Daten

Der Originaldatensatz wurde geändert auf Basis von Michael Bock & Adrian Krotz geändert. 
Details hierzu sind in der Funktion `.modify_dataset()` in `\src\utils.py` gegeben.

Außerdem wird den Symptom-Werten randomisiert +-1 addiert, um die Ungenauigkeiten der Patienten-Angaben abzubilden.


### Modell

Verwendet wird ein Bayesian Neural Network (BNN) um die Klassifikation auf Basis der angegebenen Symptome zu berechnen.
Trainiert und analysiert wurden eine Vielzahl an Modellen von (Credits: Jiang Hanmeng) für das Projekt. Die analysierten 
Modelle sind:

- Random Forest (RF)
- Support Vector Machines (SVM) 
- XGB Boost
- BNN (variational inference and multi-chain monte carlo implementation)

Das am besten performende Modell war ein BNN trainiert mit MCMC (Details: 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut1_students_with_answers.html)

---------------------

## Prädiktion <a name="Prädiktion"></a>

Das Modell gibt Wahrscheinlichkeiten der Krankheiten auf Basis der Symptome an. Hierfür können unterschiedliche 
Netzwerke trainiert werden je nach Festlegung des Inputs, d.h. wie viele und welche Tage als Input verwendet werden.

Es wurde herausgefunden, dass 2 aufeinanderfolgende Tage bis Tag 7 mit der Tages-Index Information am besten abschneidet.
Ein Modell ohne Tagesinformation und unterschiedlichen langen aufeinanderfolgenden Tagen kann auch trainiert werden 
(siehe Argumente von `train()` in `train.py`).

Die wichtigsten Argumente der `inference.py` sind:
 - restrict_to_one_week (Ob nur die erste Woche betrachtet werden soll)
 - n_consecutive (wie viele aufeinanderfolgende Tage verwendet werden soll)

Innerhalb der `train.py` und der `test.py` gibt es noch zusätzlich folgenden Parameter:
 - with_date_information (ob die Information des Tages-Index vorhanden ist/verwendet werden soll)

Während der Inferenz des Modells (`inference.py`) wird geschaut ob im Datn

Eine Analyse welche Symptom-Tage (zusätzlich zum ersten Symptom-Tag) am aussagekräftigsten sind, liefert das Ergebniss, 
dass die ersten Tage am aussagekräftigsten sind. Daher kann das Netz auf Symptomen in der ersten Woche 
(restrict_to_one_week) mit einer variablen Anzahl an aufeinanderfolgenden Tagen (n_consecutive) sind die Symptome den 
trainiert werden. Außerdem kann das Modell mit und ohne Tagesindex Information (with_date_information) trainiert werden. 

<img src="misc/images/day_analysis_recall_precision.png" width="650" height="700">

<img src="misc/images/prediction.png" width="700" height="400">

---------------------

## Inferenz / Verwendung des trainierten Modells <a name="Inferenz"></a>

Die Daten zum Testen des Modells sollen für Symptomverläufe von Patienten mit einer `sub_id` in folgendem Format als `inference.csv` in `/src/data/` abgelegt werden.

<img src="misc/images/data_structure_inference.png" width="700" height="150">

Die Inferenz wird via Docker aufgerufen. Alternativ kann `\scr\inference.py` aufgerufen werden.


---------------------

## Output <a name="Output"></a>

Eine CSV wird erstellt in `\scr\output\result_inference.csv` welche die mit folgender Struktur:

<img src="misc/images/output_inference.png" width="1000" height="40">

**Wichtiger Hinweis: Pro Patient wird eine Prädiktion gemacht. Wenn mehr Tages-Daten vorhanden sind als**
_n_consecutive_, **dann wird der Mittelwert und die Standardabweichung für alle möglichen Prädiktionen für diesen**
**Patienten gebildet.**

--------------------- 

## Installierung via Pip <a name="Pip-Installierung"></a>

Für eine schnelle Installation:

Installiere Pytorch via: 

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118``
```
Um alle anderen Packages zu installieren, wird folgender Code im Terminal ausgeführt nachdem eine neue Conda environment angelegt wurde.

```
pip install -r requirements.txt
```
--------------------- 

## Installierung als Docker <a name="Docker-Installierung"></a>

Erstellung des Docker Image via `Dockerfile`. Die `Dockerfile` muss mit Docker ausgeführt werden.

### Container auf GPU ausführen 

Um den Container auf der GPU auszuführen, muss  `nvidia-container-toolkit` auf dem Host installiert werden.
Die Dockerfile wird daraufhin via `sudo docker compose up` auf Basis der `compose.yml` file ausgeführt.

Zum Ausführen des Containers:

```
sudo docker compose up
```

#### Anleitung zum Installieren von "nvidia-container-toolkit"

Für das Installieren von `nvidia-container-toolkit` bitte folgende Befehle im Terminal ausführen.
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo chmod 0644 /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

```
sudo apt update
```

```
sudo apt-get install -y nvidia-container-toolkit
```

```
sudo nvidia-ctk runtime configure --runtime=docker
```

Überprüfung ob `nvidia-smi` angezeigt wird via Docker wenn das Docker Image `ubuntu` ausgeführt wird.
```
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Starte den "deamon" und "docker" neu:
```
sudo systemctl daemon-reload
```

```
sudo systemctl restart docker
```

--------------------- 

## TO-DOs <a name="TO-Dos"></a>

- **Modell-Loading in Abhängigkeit der Anzahl an Tages-Daten pro Patient (subj_id) definieren**



