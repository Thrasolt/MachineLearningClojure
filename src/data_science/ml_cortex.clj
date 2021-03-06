(ns data-science.ml-cortex
  (:require
    [clojure-csv.core :as csv]
    [clojure.java.io :as io]
    [cortex.experiment.train :as train]
    [cortex.nn.execute :as execute]
    [cortex.nn.layers :as layers]
    [cortex.nn.network :as network]
    [incanter.core :as inc]
    [incanter.charts :as inc-charts]
    [clojure.string :as str]
    ))

; Wir haben in der vorherigen machine_learning_intro.clj gesehen, dass wir durch das Nutzen von schnellen
; Matrixoperationen große Datenmengen verarbeiten können. Eine Library, die diesen Umstand nutzt, ist die
; Neural Network Library Cortex, die das Herzstück dieser Lerneinheit bildet.


; Der erste Schritt in diesem Machine Learning mit Clojure und Cortex Tutorial besteht darin, das bekannte
; zweidimensionale Toy-dataset TwoMoons zu klassifizieren. Es besteht aus zwei opponierenden leicht versetzen
; Punktwolken, die wie Sichelmonde aussehen. Der obere und der untere Sichelmond bilden jeweils eine eigene Klasse.
; Unser Ziel ist jedem Punkt die richtige Klasse zuzuordnen. Durch die Sichelform der Klassen ist eine lineare
; Trennung der Daten nicht möglich.

; Dafür müssen wir diese Daten zuerst importieren. Um das zu erleichtern, findet Ihr im resources Ordner
; eine csv-Datei names two-moons.csv, die 5.000 klassifizierte Punkte enthält

(def two-moons-csv-file-name "resources/two_moons.csv")

(defn read-dataset-from-csv
  "Takes cvs file name and reads data."
  [filename delimiter]
  (vec (with-open [file (io/reader filename)]
    (doall (map (comp
                  (fn [line] (map (fn [num] (Double/parseDouble num)) (str/split line delimiter)))
                  first
                  first
                  csv/parse-csv) (line-seq file))))))

; Mithilfe dieser Funktion können wir unseren Datensatz nun einlesen.
(def two-moons-dataset (read-dataset-from-csv two-moons-csv-file-name #";"))

; Wie im Machine Learning üblich müssen wir unseren Datensatz noch in drei verschiedene
; Datensätze aufteilen. Die Trainingsdaten werden genutzt, um unser Model zu trainieren, die
; Validierungsdaten, um die Hyperparameter zu optimieren und die Testdaten um die Fähigkeit
; unseres Models zu generalisieren auf ungesehenen Daten zu überprüfen.

(defn train-val-test-split [data [train-quota val-quota]]
  (let [data-size (count data)
        train (* data-size train-quota)
        val (+ train (* data-size val-quota))]
      (mapv (fn [start end] (subvec data start end))
           [0 train val] [train val data-size])))

; Wir teilen den Datensatz dafür folgendermaßen auf:
; Training: 80%, Validierung 10% und Test 10%.

(def data-split (train-val-test-split two-moons-dataset [0.8 0.1]))

(def training-set (first data-split))
(def validation-set (second data-split))
(def test-set (last data-split))

; Da unsere Daten zweidimensional sind, können wir diese auch sehr leicht
; visualisieren dazu nutzen wir die Library Incanter.

; Wir extrahieren die Werte der x- und y-Achsen und zusätzlich die labels, um
; die beiden unterschiedlichen Klassen auch Farblich zu markieren

(def x-values (mapv first two-moons-dataset))
(def y-values (mapv second two-moons-dataset))
(def labels (mapv last two-moons-dataset))

; Dann definieren wir den Plot, indem wir x- und y-Werte übergeben und ein paar
; Optionen einstellen

(def two-moon-plot
  (inc-charts/scatter-plot
    x-values
    y-values
    :group-by labels
    :title "Two Moon Dataset"
    :x-label "x-achse"
    :y-label "y-achse"))

;Um den Plot anzuzeigen nutzen wir die view Funktion von incanter.
(inc/view two-moon-plot)

; Nun haben wir die notwendige Vorbereitung der Daten abgeschlossen und
; können damit beginnen unser Model, das heißt unser neuronales Netz zu
; definieren

; Wir wollen ein einfaches sogenanntes Fully-Connected-Neural-Network
; bauen, um die Two Moon Classification zu lernen. Deshalb werden wir
; mithilfe der Cortex-API ein neuronales Netz mit zwei Hidden Layer-n
; konstruieren. Aufgrund der wenigen Dimensionen unserer Trainingsdaten
; wählen wir den Tangens Hyperbolicus als Aktivierungsfunktion.

; Unser Model erwartet jeden Input in das Netzwerk als Map, mit dem Key
; :input für die eingehenden Daten und dem key :output für die ausgehenden.
; Dafür schreiben wir die folgenden Funktionen.

(defn prepare-2class-label [label] (if (< label 1) [1.0 0.0] [0.0 1.0]))

(defn transform-for-cortex [data]
  (map (fn [[x y label]] {:input [x y] :output (prepare-2class-label label)}) data))

(def training-input (transform-for-cortex training-set))
(def validation-input (transform-for-cortex validation-set))
(def test-input (map (fn [entry] (dissoc entry :output)) (transform-for-cortex test-set)))

; Die Dimensionen der Eingabedaten eines zweidimensionalen sind immer etwas trivial, aber es
; dient der lesbarkeit sie extra zu definieren. Deren Anordnung folgt der Konvention für Bilder.

(def data-width 2)
(def data-height 1)
(def data-channels 1)
(def two-moons-number-of-classes 2)
(def two-moons-hidden-size 16)

; Unser Model besteht aus einem Input-Layer, einem Output-Layer und zwei Hidden-Layern. Auf einen Hidden-Layer
; folgt eine nicht lineare Aktivierungsfunktion, damit das Netzwerk als Funktion nicht in eine lineare
; Abbildung zusammenfällt. Erstaunlich ist, dass durch simple nicht lineare Funktionen theoretisch bereits
; beliebige Funktionen approximiert werden können.

(def model (network/linear-network
             [(layers/input data-width data-height data-channels :id :input)
              (layers/linear->tanh two-moons-hidden-size)
              (layers/linear->tanh two-moons-hidden-size)
              (layers/linear two-moons-number-of-classes)
              (layers/softmax :id :output)
              ]))

; Der Wrapper "network/linear-network" ist an dieser Stelle optional und kann auch weggelassen werden, da
; "train/train-n" diesen dann intern aufruft.

; Wir trainieren das Netzwerk mit einen Batchsize von 200, das heißt, wir verarbeiten jeweils 200 Punkte
; gleichzeitig. Außerdem trainieren wir für 100 Epochen, was bedeutet, dass wir den Trainings-Loop 100 Mal
; ausführen werden.

(def trained  (train/train-n model training-input validation-input
                             :batch-size 200
                             :network-filestem "resources/two-moons-model"
                             :epoch-count 100))

; Lasst uns nach dem Training herausfinden, wie gut sich unser Netzwerk auf dem Test Datensatz schlägt

(def predictions (execute/run trained test-input))

(defn predictions-to-labels [predictions]
  (mapv (fn [[x1 x2]] (if (> x1 x2) 0 1)) (map :output predictions)))

(def predicted-labels (predictions-to-labels predictions))

(def x-test-values (mapv first test-set))
(def y-test-values (mapv second test-set))
(def test-labels (mapv last test-set))

(def two-moon-classification-plot
  (inc-charts/scatter-plot
    x-test-values
    y-test-values
    :group-by predicted-labels
    :title "Two Moon Classification"
    :x-label "x-achse"
    :y-label "y-achse"))

(inc/view two-moon-classification-plot)

; Wir sehen, dass unser Model fast alle Punkte aus dem Two-Moon-Dataset richtig klassifiziert hat
; In Zahlen ausgedrückt wird die Genauigkeit einen Klassifizierers so ausgerechnet.

(defn accuracy [labels predictions]
  (- 1.0 (/ (apply + (map (fn [label pred] (Math/abs (- label pred))) labels predictions))
          (count labels))))
(accuracy test-labels predicted-labels)
; => 0.998

; Unser Klassifizierer hat 99.8% der Punkte richtig klassifiziert

; Ausgehend von diesem Toy-Example versuchen wir nun das bekannte MNIST Dataset zu klassifizieren
; Wird laden diese runter und speichern sie unter resources/MNIST.

(defn download-data [uri file]
  (with-open [in (io/input-stream uri)
              out (io/output-stream file)]
    (io/copy in out)))

(def MNIST-training-csv-file-name "resources/MNIST/mnist_train.csv")
(def MNIST-test-csv-file-name "resources/MNIST/mnist_test.csv")

(download-data "https://pjreddie.com/media/files/mnist_train.csv", MNIST-training-csv-file-name)
(download-data "https://pjreddie.com/media/files/mnist_test.csv", MNIST-test-csv-file-name)

; Zuerst laden wir die Trainings- und Testdaten aus den zwei csv-Dateien.

(defn read-MNIST-dataset-from-csv
  "Takes cvs file name and reads data."
  [filename]
  (vec (with-open [file (io/reader filename)]
         (doall (map (comp
                       (fn [line] (mapv (fn [num] (Double/parseDouble num)) line))
                       first
                       csv/parse-csv) (line-seq file))))))

(def MNIST-train-raw (read-MNIST-dataset-from-csv MNIST-training-csv-file-name))
(def MNIST-test-raw (read-MNIST-dataset-from-csv MNIST-test-csv-file-name))

; Da wir Testdaten haben, brauchen wir hier nur einen Split der Trainings Daten in Test-
; und Validierungsdaten
(def MNIST-split (train-val-test-split MNIST-train-raw [0.8 0.2]))

(def training-set (first MNIST-split))
(def validation-set (second MNIST-split))

; Zu nächst definieren wir ein paar nützliche Konstanten, die sich aus den Eingabe-Daten, Bildern ergeben.
; Ein Bild ist immer vom Format Breite x Höhe x 3, wobei Breite und Höhe diese Werte in Pixeln beschreiben
; und die Drei für die drei Farbkanäle steht. Im MNIST Datensatz sind die Bilder vom Format 28x28x3.

(def image-width 28)
(def image-height 28)
(def num-of-color-channels 1)
(def number-of-classes 10)

; Das es sich bei den Daten des MNIST Dataset um Bilder handelt werden wir hier kein Fully-Connected-Neural-Network
; benutzen, sondern ein Convolutional-Neural-Network. Bei dem die Gewichte statt aus einfachen Matrizen
; aus komplexen Faltungsfiltern bestehen. Zur Klassifizierung selbst werden dann aber in den abschließenden
; Layern des Netzwerks auch noch Fully-Connected-Layer benutzt.

; Was bedeuten die Zahlen in "layers/convolutional 5 0 num-of-color-channels 20)"? Die fünf steht für für eine
; Filtergröße von 5x5, die 0 steht dafür, dass die Rändern der Bilder nicht mit Nullen aufgefüllt werden, Stichwort
; zero padding, die eins steht für einen stride von 1, das heißt, das der Filter mit einer Schrittweite von 1 über
; das Bild bewegt wird. Die 20 steht für die Anzahl der Filter, die auf das Bild angewandt werden.

; Grafik für besseres Verständnis:
; https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_17A-ConvolutionalNeuralNetworks-WHITEBG.png

; Max Pooling ist ein ähnlicher Prozess, der die Dimensionen des Inputs herunter skalieren soll.

(def model-description
  [(layers/input image-width image-height num-of-color-channels :id :input)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/linear 1000)
   (layers/relu)
   (layers/linear number-of-classes)
   (layers/softmax :id :label)])

(defn one-hot-vector [value number-of-classes]
  (mapv (fn [val] (if (= (* 1.0 val) value) 1.0 0.0)) (range number-of-classes))
  )

(defn transform-MNIST-for-cortex [data]
  (map (fn [entry] {:input (rest entry) :label (one-hot-vector (first entry) number-of-classes)}) data))

(def MNIST-training-input (transform-MNIST-for-cortex training-set))
(def MNIST-validation-input (transform-MNIST-for-cortex validation-set))
(def MNIST-test-input (map (fn [entry] (dissoc entry :label)) (transform-MNIST-for-cortex MNIST-test-raw)))

; Wir trainieren das Netzwerk mit einen Batchsize von 200, das heißt, wir verarbeiten jeweils 200 Bilder
; gleichzeitig. Außerdem trainieren wir für 10 Epochen, was bedeutet, dass wir den Trainings-Loop 10 Mal
; ausführen werden.

; Vor allem mit älterer Hardware oder CPU kann diese Ausführung über eine Stunde dauern.
(def trained-MNIST  (train/train-n model-description MNIST-training-input MNIST-validation-input
                             :batch-size 200
                             :network-filestem "resources/MNIST-model"
                             :epoch-count 1))


(defn argmax [seq]
  (first (reduce (fn [[index element] [arg acc]]
            (if (> element acc) [(* 1.0 index) element] [arg acc]))
          [0.0 (first seq)]
          (map vector (range) seq))))

(def MNIST-predictions (execute/run trained-MNIST MNIST-test-input))
(def MNIST-test-labels (map first MNIST-test-raw))

(def MNIST-prediction-labels (map argmax (map :label MNIST-predictions)))

; MNIST-test-accuracy
(/ (reduce + (map
  (fn [label pred] (if (= label pred) 1.0 0.0))
  MNIST-test-labels MNIST-prediction-labels)) (count MNIST-test-labels))
; => 0.8478

; Nun haben wir gesehen, wie wir komplexe Klassifizierer mit der Cortex Library schreiben und trainieren können.
; Allerdings ist die API dieser Library so sehr "high level", dass es schwierig ist nachvollziehen, was genau
; die einzelnen Schritte sind. Um das genauer zu beleuchten, sehen wir uns als nächstes eine Implementierung
; eines rudimentären Klassifizierers in reinem clojure an, siehe ml-clojure.clj an.



