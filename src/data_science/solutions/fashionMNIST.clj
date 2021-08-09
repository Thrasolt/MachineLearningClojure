(ns data-science.solutions.fashionMNIST
  (:require
    [clojure-csv.core :as csv]
    [clojure.java.io :as io]
    [cortex.experiment.train :as train]
    [cortex.nn.execute :as execute]
    [cortex.nn.layers :as layers]
    ))

(defn train-val-test-split [data [train-quota val-quota]]
  (let [data-size (count data)
        train (* data-size train-quota)
        val (+ train (* data-size val-quota))]
    (mapv (fn [start end] (subvec data start end))
          [0 train val] [train val data-size])))

(def MNIST-training-csv-file-name "resources/FashionMNIST/fashion-mnist_train.csv")
(def MNIST-test-csv-file-name "resources/FashionMNIST/fashion-mnist_test.csv")

(defn read-FashionMNIST-dataset-from-csv
  "Takes cvs file name and reads data."
  [filename]
  (vec (with-open [file (io/reader filename)]
         (doall (map (comp
                       (fn [line] (mapv (fn [num] (Double/parseDouble num)) line))
                       first
                       csv/parse-csv) (rest (line-seq file)))))))

(def MNIST-train-raw (read-FashionMNIST-dataset-from-csv MNIST-training-csv-file-name))
(def MNIST-test-raw (read-FashionMNIST-dataset-from-csv MNIST-test-csv-file-name))

(def MNIST-split (train-val-test-split MNIST-train-raw [0.8 0.2]))

(def training-set (first MNIST-split))
(def validation-set (second MNIST-split))

(def image-width 28)
(def image-height 28)
(def number-of-classes 10)

(def model-description
  [(layers/input image-width image-height 1 :id :input)
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

(def trained-MNIST  (train/train-n model-description MNIST-training-input MNIST-validation-input
                             :batch-size 200
                             :network-filestem "resources/FashionMNIST-model"
                             :epoch-count 10))


(defn argmax [seq]
  (first (reduce (fn [[index element] [arg acc]]
            (if (> element acc) [(* 1.0 index) element] [arg acc]))
          [0.0 (first seq)]
          (map vector (range) seq))))

(def MNIST-predictions (execute/run trained-MNIST MNIST-test-input))
(def MNIST-test-labels (map first MNIST-test-raw))

(def MNIST-prediction-labels (map argmax (map :label MNIST-predictions)))

(def acc (/ (reduce + (map
  (fn [label pred] (if (= label pred) 1.0 0.0))
  MNIST-test-labels MNIST-prediction-labels)) (count MNIST-test-labels)))

(println acc)



