# Content

- Motivation (2 Seiten of which 1 Problem/Motivation 0.5 Seiten Ziele/Arbeitspakete Arbeit, 0.5 Strukturierung)
    - Definition Fahrradfreundlichkeit (Titel erklären)
    - Schon für Menschen schwierig zu erkennen (auf Satellitenbildern)
- Stand der Wissenschaft/Forschung/Technik (Methoden, Literaturrecherce)
    - Image (instance segmentation) semantic segmentation vs image detection (bounding boxes) vs klassifizierung 
    - Aktivierungsfunktionen ELU, RELU, Sigmoid
    - (Potentielle Probleme (mit kleinem Datensatz): Overfitting, Underfitting, Vanishing Gradient ...)
    - Metriken
        - Binary Cross Entropy
        - IoU
        - Dice bzw. f1
        - Quality (Correctness/Completeness)
    - (andere Architekturen zur image semantic segmentation?? Will Markus das?)
    - Architekturkomponenten
        - Dropout Layers (hilft gegen overfitting)
        - Batch Normalization (hilft aus gründen + Datensatz sonst nicht sonderlich normalized/standardized - für Inference werden durchschnittswerte verwendet)
        - U-NET
            - (Skip Connections (hilft gegen vanishing gradient und mit localization))
            - Gut für kleiner Datensatz (was ist schon ein kleiner Datensatz?)
    - Backbones (KURZE Vorstellung ausgewählter Netze)
        - vgg16
        - (inceptionv3)
        - (resnet34)
        - densenet121
    - Transfer Learning
        - insb. mit UNET
            - backbone strategie
            - verschiedene freeze-strategien
            - einfach bisherigen research kurz abbreißen
            - benchmarks
    - Straßenerkennung 
        - Datensätze
            - Massachuchettes (viel weiß)
            - Deep Globe (viel unterschiedliche szenerien)
            - LandCover.ai (zu viel acker)
            - ... (siehe Citavi)
        - Benchmarks
    - Herausarbeitung des Neuheitswertes (vgl gegen Straßenerkennung)
- Konzeption
    - Pretraining auf roads data set
        - Mass angepasst 
    - Image Semantic Segmentation (Warum?)
    - Datensatz 
        - gibt es keinen :(, aber wir haben einen selber gemacht :)
        - manuell gelabeled vs automatisch gelabeled ? 
        - zahlen daten fakten größe auflösung etc pp
        - unterschiedliche Städte
    - Architektur von unsere(m/n) Netz(en)
    - Hyperparameter
    - (Image Augmentation -> robuster)
    - IoU + Quality ganz gut für spätere Bewertung weil Mix aus stringent und locker + menschliche Einschätzung, Warum Dice und nicht Falpha oder BCE ? 
- Implementierung
    - KI bedienen
    - Datensatz bauen 
- Ergebnisse
    - Ergebnisse von pre-training auf raods
        - mit anpassung bessere ergebnisse als paper. Ohne Anpassung konnten Paper-Ergebnisse reproduziert werden.
    - Dropout sehr wichtig (sonst haben wir heftiges Overfitting)
- Diskussion
    - Warum gerade bei unserem Datensatz Dropout wichtig?
    - Metriken (IoU vs Quality gepaart mit empirischem empfinden)


- Schluss
    - Zusammenfassung
    - Kritische Reflexion
    - Ausblick
        - Post-Processing