# Keras human detection

In diesem Beispiel sollten eigene Bilder eingelesen, trainiert und für eine Live-Erkennung mittels Webcam optimiert werden. Dafür können mit *generate_dataset.py* Bilder generiert werden. Wir nahmen dafür 2000 Bilder mit einer Person und 2000 ohne Person auf. Diese wurden mittels *ImageDataGenerator* und einer geeigneten Ordnerstruktur eingelesen, normiert und trainiert. Das daraus resultierende Modell wurde für eine Live-Erkennung in *human_detecion.py* eingelesen und einzelne Webcam-Frames predicted. Die Resultate waren sehr gut. Aufgenommene Bilddateien werden durch die Größe der Daten nicht mitgeliefert können aber wie erklärt mittels *generate_dataset.py* über eine integrierte Webcam aufgezeichnet werden.