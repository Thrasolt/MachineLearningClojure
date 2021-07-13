(ns data-science.ml-clojure
  (:require
    [clojure.string :as str]
    [clojure-csv.core :as csv]
    [clojure.java.io :as io]
    ))

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

(defn train-val-test-split [data [train-quota val-quota]]
  (let [data-size (count data)
        train (* data-size train-quota)
        val (+ train (* data-size val-quota))]
    (mapv (fn [start end] (subvec data start end))
          [0 train val] [train val data-size])))

(def data-split (train-val-test-split two-moons-dataset [0.8 0.1]))

(def training-set (first data-split))
(def validation-set (second data-split))
(def test-set (last data-split))

(defn batch-ify [data batch-size] (partition-all batch-size batch-size data))

(def batch-size 100)
(def batches (batch-ify training-set batch-size))

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

(defn matrix-addition [left right]
  (mapv
    (fn [l r] (if (vector? l) (mapv + l r) (+ l r)))
    left right))

(defn vector-subtraction [left right]
  (mapv - left right))

(defn matrix-vector-mul [matrix vector]
  (mapv (partial dot-product vector) matrix))

(defn transpose [matrix]
  (apply mapv vector matrix))

(defn horizontal-sum [matrix]
  (mapv (partial reduce +) matrix))

(defn vertical-sum [matrix]
  (mapv (partial reduce +) (transpose matrix)))

(defn matrix-mul [left right]
  (mapv (partial matrix-vector-mul left) (transpose right))
  )

(defn random-matrix [rows columns]
  (vec (repeat rows (vec (repeatedly columns rand)))))

(defn init-singly-entry [rows columns]
  (let [sqr6 (Math/sqrt 6.0)]
    (/ (+ (* -1 sqr6) (* (Math/random) (* 2 sqr6))) (+ rows columns))))

(defn init-single-row [rows columns]
  (vec (repeatedly columns (fn [] (init-singly-entry rows columns)))))

(defn init-weights [rows columns]
  (vec (repeatedly rows (fn [] (init-single-row rows columns)))))

(def hidden-layer-size 16)

(def Weight1 (init-weights hidden-layer-size 2))
(def bias1 (vec (repeat hidden-layer-size 0)))
(def Weight2 (init-weights 2 hidden-layer-size))
(def bias2 (vec (repeat 2 0)))

(defn zeros [rows columns]
  (vec (repeat rows (vec (repeat columns 0.0)))))

(def cache
  (atom {:relu-activation [] :inner-activation []}))

(def parameters
  (atom {:weight1 Weight1 :weight2 Weight2 :bias1 bias1 :bias2 bias2}))

(def grads
  (atom {:weight1 (zeros hidden-layer-size 2) :weight2 (zeros 2 hidden-layer-size) :bias1 (vec (repeat 2 0)) :bias2 (vec (repeat 2 0))}))

(def momentum
  (atom {:weight1 (zeros hidden-layer-size 2) :weight2 (zeros 2 hidden-layer-size) :bias1 (vec (repeat 2 0)) :bias2 (vec (repeat 2 0))}))

(defn save-grads [dic [& args]]
  (reduce (fn [acc [key val]] (update-in acc [key] matrix-addition val)) dic
          (mapv vector [:weight1 :bias1 :weight2 :bias2] args))
  )

(defn divide-grads [dic batch-size]
  (reduce-kv (fn [dic key val] (assoc dic key (scalar-multiplication (/ 1 batch-size) val))) {} dic))

(defn zero-grads [dic] {:weight1 (zeros hidden-layer-size 2) :weight2 (zeros 2 hidden-layer-size) :bias1 (vec (repeat 2 0)) :bias2 (vec (repeat 2 0))})

(defn update-cache [dict relu-activation inner-activation]
  (assoc dict :relu-activation relu-activation :inner-activation inner-activation))

(defn relu-forward [input]
  (mapv (fn [element] (if (> element 0) element 0)) input))

(defn tanh-forward [input]
  (mapv (fn [element] (Math/tanh element)) input))

(defn tanh-backward [input outer-grad]
  (elementwise-multiplication outer-grad (mapv (fn [element] (- 1.0 (Math/pow (Math/tanh element) 2))) input)))

(defn linear-layer [w1 b1 input]
  (->> input (matrix-vector-mul w1) (vector-addition b1)))

(defn fc-nn-forward-training [[w1 w2] [b1 b2] input]
  (let [relu-activation (linear-layer w1 b1 input)
        inner-activation (tanh-forward relu-activation)]
    (swap! cache update-cache relu-activation inner-activation)
    (linear-layer w2 b2 inner-activation)))

(defn softmax [input]
  (let [den (reduce (fn [acc el] (+ acc (Math/exp el))) 0 input)]
    (mapv (fn [element] (/ (Math/exp element) den)) input)))

(defn fc-nn-forward [params input]
  (let [w1 (params :weight1) b1 (params :bias1)
        w2 (params :weight2) b2 (params :bias2)]
    (mapv (fn [element] (->> element
                             (linear-layer w1 b1)
                             (tanh-forward)
                             (linear-layer w2 b2)
                             (softmax))) input)))

(defn one-hot [label num-of-classes]
  (mapv (fn [element] (if (= label element) 1 0)) (range num-of-classes)))

(defn cross-entropy-loss [input label]
  (* -1.0 (Math/log (/ (Math/exp (nth input label))
                       (reduce (fn [acc el] (+ acc (Math/exp el))) 0 input)))))

(defn batch-loss [output labels]
  (mapv cross-entropy-loss output labels))

(defn cross-entropy-grad [output label]
  (let [num-of-classes (count output)
        out-probs (softmax output)
        label-vec (one-hot label num-of-classes)]
    (vector-subtraction out-probs label-vec)))

(defn bias2-grad [cross-entropy-grad] cross-entropy-grad)
(defn weight-grad [input outer-grad]
  (mapv (fn [grad] (mapv (partial * grad) input)) outer-grad))
(defn relu-grad [input outer-grad]
  (mapv (fn [grad input] (if (> input 0) grad 0.0)) outer-grad input))
(defn input-grad [Weight outer-grad]
  (matrix-vector-mul (transpose Weight) outer-grad))

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

(defn batch-forward [model batch labels]
  (mapv
    (fn [entry label]
      (let [output (model entry)
            weights (backward entry output label)]
        (swap! grads save-grads weights)
        output))
    batch labels))

(defn optimize-step [value grad momentum learning-rate rho]
  (let [velocity (matrix-addition (scalar-multiplication rho momentum) grad)]
    (matrix-addition value (scalar-multiplication (* -1.0 learning-rate) velocity))))

(defn optimize [learning-rate rho]
  (let [current-grads @grads current-momentum @momentum]
    (swap! parameters
           (fn [params]
             (reduce-kv
               (fn [dic key val]
                 (assoc dic key (optimize-step val (current-grads key) (current-momentum key) learning-rate rho)))
               {} params)))))

(defn train [forward data labels lr]
  (let [params @parameters
        weights [(params :weight1) (params :weight2)]
        biases [(params :bias1) (params :bias2)]
        output (batch-forward (partial forward weights biases) data labels)
        loss (/ (reduce + (batch-loss output labels)) (count labels))]
    (println loss)))

(defn training [forward batches lr epochs]
  (dotimes [_ epochs]
    (doseq [batch batches]
      (let [data (mapv (partial take 2) batch) labels (mapv last batch)]
        (train forward data labels lr)
        (swap! grads divide-grads (count batch))
        (optimize lr 0.90)
        (swap! grads zero-grads)
        (swap! momentum zero-grads)))))


(training fc-nn-forward-training batches 0.001 2)
; ...
; 0.6902198984851777
; => nil

(defn argmax [seq]
  (first (reduce (fn [[index element] [arg acc]]
                   (if (> element acc) [(* 1.0 index) element] [arg acc]))
                 [0.0 (first seq)]
                 (map vector (range) seq))))

(defn accuracy [labels predictions]
  (- 1.0 (/ (apply + (map (fn [label pred] (Math/abs (- label pred))) labels predictions))
            (count labels))))

(def training-input (mapv (comp vec (partial take 2)) training-set))
(def training-labels (mapv last training-set))
(def training-predictions (fc-nn-forward @parameters training-input))
(def training-pred-labels (mapv argmax training-predictions))

(accuracy training-labels training-pred-labels)
;=> 0.87575

(def validation-input (mapv (comp vec (partial take 2)) validation-set))
(def validation-labels (mapv last validation-set))
(def validation-predictions (fc-nn-forward @parameters validation-input))
(def validation-pred-labels (mapv argmax validation-predictions))

(accuracy validation-labels validation-pred-labels)
; => 0.894

(def test-input (mapv (comp vec (partial take 2)) test-set))
(def test-labels (mapv last test-set))
(def test-predictions (fc-nn-forward @parameters test-input))
(def test-pred-labels (mapv argmax test-predictions))

(accuracy test-labels test-pred-labels)
; => 0.856
