The Living Crime Scene
Auteur: Berry Kriesels

Dit is de code repository voor het project The Living Crime Scene. In dit onderzoek wordt gekeken naar de toepassing van Spiking Neural Networks (SNN's) als sensoren voor digitale forensische analyse.

Het concept is gebaseerd op het idee dat een biologisch geïnspireerd netwerk op een specifieke manier reageert op verstoringen. Door het netwerk bloot te stellen aan gesimuleerde cyberaanvallen, kunnen we forensische sporen (zoals veranderingen in vuursnelheid of entropie) detecteren die in traditionele log-analyse mogelijk over het hoofd worden gezien.

De bestanden
De repository bestaat uit twee hoofdscripts die na elkaar gedraaid moeten worden.

1. living_crime_scene_snn_speedversion7.py
Dit script bevat de simulatie-engine. Het maakt gebruik van de Brian2 simulator om een netwerk van neuronen op te zetten. Dit netwerk wordt vervolgens blootgesteld aan verschillende scenario's:

Baseline (normaal gedrag)

Overstimulation (DDoS-achtig)

Poisoning (aanpassing van synaptische gewichten)

Drift (langzame degradatie van parameters)

sVCE (virale verspreiding over de graaf)

Het script genereert synthetische data en schrijft de resultaten weg naar een CSV-bestand (living_crime_scene_results.csv).

2. analyze_living_crime_scene8.py
Dit is het analyse-script. Het leest de data die door de simulatie is gegenereerd en voert statistische en visuele analyses uit. Het script kijkt onder andere naar:

Statistische significantie (Z-scores, ANOVA)

Visualisaties van de data (Boxplots, Heatmaps, PCA)

Classificatie via Machine Learning (Random Forest) om te testen of de aanvallen onderscheidbaar zijn.

Anomaly detection

Installatie
Om de scripts te gebruiken is een Python-omgeving nodig met de volgende libraries:

pip install brian2 pandas numpy matplotlib seaborn scikit-learn scipy

Gebruik
Draai eerst de simulatie. Dit kan enkele minuten duren, afhankelijk van de settings en je hardware.

python living_crime_scene_snn_speedversion7.py

Je kunt optioneel parameters meegeven om de intensiteit van de aanval of het aantal trials aan te passen (bijvoorbeeld -n 5 voor meer trials).

Zodra de simulatie klaar is, draai je de analyse:

python analyze_living_crime_scene8.py

De resultaten, waaronder grafieken en een samenvattend tekstrapport, worden opgeslagen in de mappen figures en enhanced_figures.

Licentie
Deze code is geschreven voor onderzoeksdoeleinden. Gebruik en aanpassingen zijn toegestaan, graag met bronvermelding.
