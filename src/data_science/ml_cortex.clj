(ns data-science.ml-cortex
  (:require
    [clojure-csv.core :as csv]
    [clojure.java.io :as io]
    [cortex.experiment.train :as train]
    [cortex.nn.execute :as execute]
    [cortex.nn.layers :as layers]
    [cortex.nn.network :as network]
    [incanter.core :as inc]
    [incanter.io :as inc-io]
    [incanter.charts :as inc-charts]
    [clojure.string :as str]
    ))


; Der erste Schritt in diesem Machine Learning mit Clojure und Cortex tutorial besteht darin, das bekannte
; zweidimensionale Toy-dataset TwoMoons zu klassifizieren. Es besteht aus zwei opponierenden leicht versetzen
; Punktwolken, die wie Sichelmonde aussehen.

; Dafür müssen wir diese Daten zuerst importieren. Um das zu erleichtern, findet Ihr im resources Ordner
; eine csv-Datei names two-moons.csv, die 5.000 klassifizierte Punkte enthält

(def csv-file-name "resources/two_moons.csv")

(defn read-dataset-from-csv
  "Takes cvs file name and reads data."
  [filename]
  (vec (with-open [file (io/reader filename)]
    (doall (map (comp
                  (fn [line] (map (fn [num] (Double/parseDouble num)) (str/split line #";")))
                  first
                  first
                  csv/parse-csv) (line-seq file))))))

; Mithilfe dieser Funktion können wir unseren Datensatz nun einlesen
(def two-moons-dataset (read-dataset-from-csv csv-file-name))

; Wie im Machine Learning üblich müssen wir unseren Datensatz noch in drei verschiedene
; Datensätze aufteilen. Die Trainingsdaten werden genutzt um unser Model zu trainieren, die
; Validierungsdaten, um die Hyperparameter zu optimieren und die Testdaten um die Fähigkeit
; unseres Models zu generalisierung auf ungesehenen Daten zu überprüfen.

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

; Da diese unsere Daten zweidimensional sind, können wir diese auch sehr leicht
; visualisieren dazu nutzen wir die Library incanter



(def x-values (mapv first two-moons-dataset))
(def y-values (mapv second two-moons-dataset))
(def labels (mapv last two-moons-dataset))

(def my-plot
  (inc-charts/scatter-plot
    x-values
    y-values
    :group-by labels
    :title "Two Moon Dataset"
    :x-label "x-achse"
    :y-label "y-achse"))

(inc/view my-plot)





; Nun haben wir die notwendige Vorbereitung der Daten abgeschlossen und
; können damit beginnen unser Model, das heißt unser neuronales Netz zu
; definieren



