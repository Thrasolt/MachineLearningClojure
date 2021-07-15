(ns data-science.ml-clojure
  (:require
    [clojure.string :as str]
    [clojure-csv.core :as csv]
    [clojure.java.io :as io]
    [incanter.core :as inc]
    [incanter.io :as inc-io]
    [incanter.charts :as inc-charts]
    ))

; Um besser zu verstehen, was große Libraries wie Cortex machen, werden wir hier gemeinsame
; einen rudimentären neuronalen Klassifizierer in reinem Clojure schreiben,
; ohne die Hilfe von anderen Libraries. Dieser Klassifiziere soll dann das zweidimensionale Toy-dataset
; TwoMoons klassifizieren

; Grundlegende Operationen

; Zunächst müssen wir grundlegende Operation der Linearen Algebra definieren, denn die
; Matrix-Vektor-Multiplikationen bilden die Grundlage aller neuronalen Netzwerke

(defn dot-product [vec1 vec2]
  (reduce + (map * vec1 vec2)))

(defn scalar-multiplication [scalar tensor]
  (mapv
    (if (vector? (first tensor))
      (partial mapv (partial * scalar))
      (partial * scalar))
    tensor))

(defn vector-addition [left right]
  (mapv + left right))

(defn elementwise-multiplication [left right]
  (mapv
    (fn [l r] (if (vector? l) (mapv * l r) (* l r)))
    left right))

(defn elementwise-division[left right]
  (mapv
    (fn [l r] (if (vector? l) (mapv / l r) (/ l r)))
    left right))

(defn matrix-addition [left right]
  (mapv
    (fn [l r] (if (vector? l) (mapv + l r) (+ l r)))
    left right))

(defn matrix-subtraction [left right]
  (mapv
    (fn [l r] (if (vector? l) (mapv - l r) (- l r)))
    left right))

(defn vector-subtraction [left right]
  (mapv - left right))

(defn matrix-vector-mul [matrix vector]
  (mapv (partial dot-product vector) matrix))

(defn transpose [matrix]
  (apply mapv vector matrix))


; Neuronales Netz
; Da das TwoMoon Dataset aus zwei-dimensionalen Punkten besteht, die in aus zwei verschiedenen
; Klassen stammen, brauchen wir ein Neuronales Netzwerk, das zwei-dimensionale Punkte als Input
; akzeptiert und auch einen zwei-dimensionale Vektor ausgibt, der aus Klassen-Scores besteht,
; die abgeben, wie sicher das Netzwerk ist, dass die Eingabe zur entsprechenden Klasse gehört.
; Wir werden ein schlichtes Neuronales Netzwerk mit einem Hidden-Layer konstruieren, der aus
; 16 Neuronen besteht und zwei dimensionale Punkte als Input akzeptiert und einen einen
; Vektor mit zwei Klassen-Scores zurückgibt

; Weight Initialization
; Zunächst müssen wir die Weights und Biases der Neuronen in den beiden Layern initialisieren, es gibt
; verschiedene Ansätze der Gewichts initialisierung. Wir verwenden an dieser Stelle die random Funktion
; von Clojure und bilden diese auf einen Abschnitt zwischen -1 und 1 ab
; Anschließen verkleinern wir diese Werte noch um, das Problem der exploding Gradients zu vermeiden

(defn init-singly-entry []
  (* 0.01 (+ -1 (* 2 (Math/random)))))

(defn init-single-row [rows columns]
  (vec (repeatedly columns init-singly-entry)))

(defn init-weights [rows columns]
  (vec (repeatedly rows (fn [] (init-single-row rows columns)))))

(defn zeros [rows columns]
  (vec (repeat rows (vec (repeat columns 0.0)))))

; Wir legen zuerst die größe des Hidden Layer fest, die größer der anderen beiden ergeben sich
; automatisch anhand der Dimensionen von Input und Output

(def hidden-layer-size 16)

; Nun initialisieren wir die Weights und Biases die unserer Netzwerk definieren

(def Weight1 (init-weights hidden-layer-size 2))
(def bias1 (vec (repeat hidden-layer-size 0)))
(def Weight2 (init-weights 2 hidden-layer-size))
(def bias2 (vec (repeat 2 0)))

; Da das Neuronale Netzwerk lernt, indem seine Gewichte und Biases verändert werden, speichern wir
; diese als Map in einem Atom

(def parameters
  (atom {:weight1 Weight1 :weight2 Weight2 :bias1 bias1 :bias2 bias2}))

; Konstruktion des Netzwerks

; Der grundlegende Baustein jeden Fully-Connected Neuronalen Netzes ist, der lineare Layer, der
; aus einer Matrix-Vektor-Multiplikation besteht.

(defn linear-layer [w1 b1 input]
  (->> input (matrix-vector-mul w1) (vector-addition b1)))

; Als nicht-lineare Komponente benutzen wir aufgrund der geringen Anzahl von Dimensionen
; einen elementweisen Tangens Hyperbolikus, der die Ergebnisse des linearen Layers auf
;Werte zwischen -1 und 1 abbildet.

(defn tanh-forward [input]
  (mapv (fn [element] (Math/tanh element)) input))

; Nach dem letzten linearen Layer transformieren den Output mit der Softmax-Funktion
; und erhalten damit Klassen-Scores, die Werte zwischen 0 und 1 annehmen, indem wir
; den den Output des Hidden Layers exponenzieren aufsummieren und dann mit den exponenzierten
; Klassen-scores ins Verhältnis setzen .

(defn softmax [input]
  (let [den (reduce (fn [acc el] (+ acc (Math/exp el))) 0 input)]
    (mapv (fn [element] (/ (Math/exp element) den)) input)))

; Mit diesen Bausteinen und der Thread-Macro von Clojure können wir auf sehr simple Weise
; den Forward-Pass unserer Neuronalen Netzes definieren

(defn fc-nn-forward [params input]
  (let [w1 (params :weight1) b1 (params :bias1)
        w2 (params :weight2) b2 (params :bias2)]
    (mapv (fn [element] (->> element
                             (linear-layer w1 b1)
                             (tanh-forward)
                             (linear-layer w2 b2)
                             (softmax))) input)))

; Loss-Funktion

; Um das Lernen unseres Neuronalen Netzes zu ermöglichen, müssen wir die Parameter mithilfe von den Gradienten
; unserer Parameter optimieren. Um die Gradienten abzuspeichern, werden wir ein weiteres Atom benutzen, das eine
; Map mit den gleichen Schlüsseln wie unsere Parameter enthält.

(def grads
  (atom {:weight1 (zeros hidden-layer-size 2) :weight2 (zeros 2 hidden-layer-size) :bias1 (vec (repeat 2 0)) :bias2 (vec (repeat 2 0))}))

; Diese werden wir updaten, dividieren und auf 0 setzen müssen, wofür wir die folgenden Funktionen implementieren
; werden.

(defn save-grads [dic [& args]]
  (reduce (fn [acc [key val]] (update-in acc [key] matrix-addition val)) dic
          (mapv vector [:weight1 :bias1 :weight2 :bias2] args)))

(defn divide-grads [dic batch-size]
  (reduce-kv (fn [dic key val] (assoc dic key (scalar-multiplication (/ 1 batch-size) val))) {} dic))

(defn zero-grads [dic]
  {:weight1 (zeros hidden-layer-size 2) :weight2 (zeros 2 hidden-layer-size)
   :bias1 (vec (repeat 2 0)) :bias2 (vec (repeat 2 0))})

; Wir werden die Gradienten bezüglich einer Loss-Funktion berechnen, die hohe Werte für schlecht klassifizierte
; Werte und geringe Werte für korrekt klassifizierte Werte annimmt. Wir werden eine Standardlösung für
; Klassifizierungsprobleme verwenden, nämlich die Cross-Entropy. Dieser ist nur genau dann gleich 0,
; wenn alle Klassen-Scores der falschen Klassen 0 und nur der richtige 1 beträgt. Aus diesem Grund hat man auch
; bei kleinen Abweichungen immer noch nützliche Gradienten.

(defn cross-entropy-loss [scores label]
  (* -1.0 (Math/log (/ (Math/exp (nth scores label))
                       (reduce (fn [acc el] (+ acc (Math/exp el))) 0 scores)))))

(defn batch-loss [output labels]
  (mapv cross-entropy-loss output labels))

; Backpropagation

; Der Algorithmus, den wir benutzen werden, um die Gradienten bezüglich dieser Loss-Funktion zu berechnen
; heißt Backpropagation und ist eine Anwendung der Kettenregel der Analysis. Um die Backpropagation zu vereinfachen
; und mehrfache Berechnungen zu vermeiden, sollten wir im Forward-Pass einige Werte zwischen Speichern.

(def cache
  (atom {:relu-activation [] :inner-activation []}))

(defn update-cache [dict relu-activation inner-activation]
  (assoc dict :relu-activation relu-activation :inner-activation inner-activation))

; Dann müssen wir einen neuen Forward-Pass schreibe, der diesen Zwischenspeicher im Training berücksichtigt

(defn fc-nn-forward-training [[w1 w2] [b1 b2] input]
  (let [relu-activation (linear-layer w1 b1 input)
        inner-activation (tanh-forward relu-activation)]
    (swap! cache update-cache relu-activation inner-activation)
    (linear-layer w2 b2 inner-activation)))

; Zu diesem Forward pass müssen wir dann den sogenannten Backward-Pass schreiben, der uns die Gradienten zu unseren
; Parametern liefert. Dafür müssen für die einzelnen Layer unserer Forward-Passes gradienten berechnen und
; entsprechend der Kettenregel mit den Gradienten der vorderen Layer multiplizieren.

(defn one-hot [label num-of-classes]
  (mapv (fn [element] (if (= label (* 1.0 element)) 1.0 0.0)) (range num-of-classes)))

(defn weight-grad [input outer-grad]
  (mapv (fn [grad] (mapv (partial * grad) input)) outer-grad))

(defn input-grad [Weight outer-grad]
  (matrix-vector-mul (transpose Weight) outer-grad))

(defn tanh-backward [input outer-grad]
  (elementwise-multiplication outer-grad (mapv (fn [element] (- 1.0 (Math/pow (Math/tanh element) 2))) input)))

(defn cross-entropy-grad [output label]
  (let [num-of-classes (count output)
        label-vec (one-hot label num-of-classes)
        y1 (first label-vec) y2 (second label-vec)
        ef1 (Math/exp (first output)) ef2 (Math/exp (second output))
        denominator (* (+ (* y1 ef1) (* y2 ef2)) (+ ef1 ef2))
        score-sum (+ ef1 ef2)]
    (println denominator)
    [(/ (* ef1 (+ (* y2 ef2) (- (* y1 ef1) (* y1 score-sum))))
        denominator)
     (/ (* ef2 (+ (* y2 ef2) (- (* y1 ef1) (* y2 score-sum))))
        denominator)]))

(defn backward-step [relu-activation inner-activation weight2 input output label]
  (let [cross-grad (cross-entropy-grad output label)
        bias-2-grad cross-grad
        Weight-2-grad (weight-grad relu-activation cross-grad)
        inner-grad (input-grad weight2 cross-grad)
        inner-relu-grad (tanh-backward inner-activation inner-grad)
        bias-1-grad inner-relu-grad
        Weight-1-grad (weight-grad input inner-relu-grad)]
    [Weight-1-grad bias-1-grad Weight-2-grad bias-2-grad]))

(defn backward [input output label]
  (let [fn-cache @cache
        weight2 (@parameters :weight2 Weight2)
        relu-activation (fn-cache :relu-activation)
        inner-activation (fn-cache :inner-activation)
        ]
    (backward-step relu-activation inner-activation weight2 input output label)))

; Mit Forward- und Backward-Pass bewaffnet können wir nun beginnen, eine Funktion zu schreiben, die eine ganze
; Batch an Daten verarbeiten kann.

(defn batch-forward [model batch labels]
  (mapv
    (fn [entry label]
      (let [output (model entry)
            weights (backward entry output label)]
        (swap! grads save-grads weights)
        output))
    batch labels))

; Optimierung

; Als Strategie zur Optimierung unserer Parameter, verwenden wir den Klassiker schlechthin. Stochastik Gradient
; Descent (SGD) mit Momentum. Dabei berechnen wir Batch-weise Gradienten und verändern unsere Parameter anhand
; ihrer Richtungen skaliert mit einer Learning-Rate, speichern und berücksichtigen aber auch vergangene Gradienten.

(def momentum
  (atom {:weight1 (zeros hidden-layer-size 2) :weight2 (zeros 2 hidden-layer-size)
         :bias1 (vec (repeat 2 0)) :bias2 (vec (repeat 2 0))}))

(defn optimize-step [value grad momentum learning-rate rho]
  (let [velocity (matrix-addition (scalar-multiplication rho momentum) grad)]
    (matrix-subtraction value (scalar-multiplication learning-rate velocity))))

(defn optimize [learning-rate rho]
  (let [current-grads @grads current-momentum @momentum]
    (swap! parameters
           (fn [params]
             (reduce-kv
               (fn [dic key val]
                 (assoc dic key (optimize-step val (current-grads key) (current-momentum key) learning-rate rho)))
               {} params)))))

; Training

; Nun können wir einen Trainings-Loop definieren, der dem von Cortex in der anderen REPL-Session gleicht.

(defn train [forward data labels lr]
  (let [params @parameters
        weights [(params :weight1) (params :weight2)]
        biases [(params :bias1) (params :bias2)]
        output (batch-forward (partial forward weights biases) data labels)
        loss (/ (reduce + (batch-loss output labels)) (count labels))]
    (println loss)
    ))

(defn training [forward batches lr epochs]
  (dotimes [_ epochs]
    (doseq [batch batches]
      (let [data (mapv (partial take 2) batch) labels (mapv last batch)]
        (train forward data labels lr)
        (swap! grads divide-grads (count batch))
        (optimize lr 0.99)
        (swap! grads zero-grads)
        (swap! momentum zero-grads)))))

(defn argmax [seq]
  (first (reduce (fn [[index element] [arg acc]]
                   (if (> element acc) [(* 1.0 index) element] [arg acc]))
                 [0.0 (first seq)]
                 (map vector (range) seq))))

(defn accuracy [labels predictions]
  (- 1.0 (/ (apply + (map (fn [label pred] (Math/abs (- label pred))) labels predictions))
            (count labels))))

; Data Preprocessing

; Das Dat Processing verläuft hier analog zur anderen REPL-Session mit der Ausnahme, dass wir die Daten hier noch
; selbst normalisieren.
(def two-moons-csv-file-name "resources/two_moons.csv")

(defn read-dataset-from-csv
  [filename delimiter]
  (vec (with-open [file (io/reader filename)]
         (doall (map (comp
                       (fn [line] (map (fn [num] (Double/parseDouble num)) (str/split line delimiter)))
                       first
                       first
                       csv/parse-csv) (line-seq file))))))

(def two-moons-dataset (read-dataset-from-csv two-moons-csv-file-name #";"))

(defn calculate-mean [data]
  (scalar-multiplication
    (/ 1 (count data))  (reduce (fn [[x-acc y-acc] [x y]] [(+ x-acc x) (+ y-acc y)]) [0.0 0.0] data)))

(defn calculate-variance[data [x-mean y-mean]]
  (calculate-mean (map (fn [[x y]] [(Math/pow (- x x-mean) 2) (Math/pow (- y y-mean) 2)]) data)))

(defn calculate-std [[x-var y-var]]
  [(Math/sqrt x-var) (Math/sqrt y-var)])

(def mean (calculate-mean two-moons-dataset))
(def variance (calculate-variance two-moons-dataset mean))
(def std (calculate-std variance))

(defn normalize-data [data]
  (let [[x-mean y-mean] (calculate-mean data) variance (calculate-variance data mean) [x-std y-std] (calculate-std variance)]
    (mapv (fn [[x y label]] [(/ (- x x-mean) x-std) (/ (- y y-mean) y-std) label]) data)))


(defn train-val-test-split [data [train-quota val-quota]]
  (let [data-size (count data)
        train (* data-size train-quota)
        val (+ train (* data-size val-quota))]
    (mapv (fn [start end] (subvec data start end))
          [0 train val] [train val data-size])))

(def two-moons-dataset-normal (shuffle (normalize-data two-moons-dataset)))

(def data-split (train-val-test-split two-moons-dataset-normal [0.8 0.1]))

(def training-set (first data-split))
(def validation-set (second data-split))
(def test-set (last data-split))

;(defn batch-ify [data batch-size] (map normalize-data (partition-all batch-size batch-size data)))
(defn batch-ify [data batch-size] (partition-all batch-size batch-size data))

(def batch-size 100)
(def batches (batch-ify training-set batch-size))


(training fc-nn-forward-training batches 0.05 10)
; ...
; 0.212801313690543
; => nil


(def training-input (mapv (comp vec (partial take 2)) training-set))
(def training-labels (mapv last training-set))
(def training-predictions (fc-nn-forward @parameters training-input))
(def training-pred-labels (mapv argmax training-predictions))

(println (accuracy training-labels training-pred-labels))
;=> 0.887

(def validation-input (mapv (comp vec (partial take 2)) validation-set))
(def validation-labels (mapv last validation-set))
(def validation-predictions (fc-nn-forward @parameters validation-input))
(def validation-pred-labels (mapv argmax validation-predictions))

(accuracy validation-labels validation-pred-labels)
; => 0.902

(def test-input (mapv (comp vec (partial take 2)) test-set))
(def test-labels (mapv last test-set))
(def test-predictions (fc-nn-forward @parameters test-input))
(def test-pred-labels (mapv argmax test-predictions))

(accuracy test-labels test-pred-labels)
; => 0.9

; Um die Performance des Neuronalen Netzwerks zu verbessern, kann man noch sehr viele zusätzliche
; Verbesserungen vornehmen. Man kann zusätzliche Layer einführen, andere Activation-Functions nutzen,
; Regularization hinzufügen, Batch-Normalization einbauen und noch viele andere Techniken nutzen.
; Fühlt Euch frei, diese nach Eurem belieben einzubauen


(def x-test-values (mapv first test-input))
(def y-test-values (mapv second test-input))

(def two-moon-classification-plot
  (inc-charts/scatter-plot
    x-test-values
    y-test-values
    :group-by test-pred-labels
    :title "Two Moon Classification"
    :x-label "x-achse"
    :y-label "y-achse"))

(inc/view two-moon-classification-plot)