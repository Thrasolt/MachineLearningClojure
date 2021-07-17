(ns data-science.machine-learning-intro)

; Mathematische Objekte und Operationen sind die Grundlage von allem was mit
; Data Science und Machine Learning zu tun hat.
; Wie macht man das in Clojure?

; Vektoren

; Vektoren sind die mathematischen Objekte, die das Fundament der modernen Datascience und
; des Machine Learnings bilden.

; In vielen anderen Sprachen, wie auch der Data Science Sprache schlechthin Python, muss man bereits
; an dieser Stelle anfangen andere Pakete wie Numpy zu verwenden in clojure kriegen wir
; das umsonst.

(def vector1 [0.1 0.2 0.3 0.4 0.5])
(def vector2 [0.6 0.7 0.8 0.9 1.0])

; Die wichtigste Operation, die man braucht, um mit Vectoren im Machine Learning zu arbeiten,
; ist das Skalarprodukt oder im englischen meist Dot Product genannt.

; In clojure können wir das so umsetzten

(defn dot-product [vec1 vec2]
  (reduce + (map * vec1 vec2)))

(dot-product vector1 vector2)
; => 1.3

; Matrizen

; Matrizen gehören ebenfalls zu diesem Fundament

(def matrix1
  [[1 2 3 4 5]
   [6 7 8 9 1]
   [2 3 4 5 6]])

; Eine der wichtigsten Matrizen ist die Identitätsmatrix

(def identity-5x5
  [[1 0 0 0 0]
   [0 1 0 0 0]
   [0 0 1 0 0]
   [0 0 0 1 0]
   [0 0 0 0 1]])

; Der wichtigste Anwendungsfall des Skalarprodukts in Data Science und Machine Learning ist die
; Matrix-Vektor- oder Matrix-Matrix-Multiplikation.

; Matrix-Vektor-Multiplikation könnte man in Clojure so implementieren

(defn matrix-vector-mul [matrix vector]
  (mapv (partial dot-product vector) matrix))

(matrix-vector-mul matrix1 vector1)
; => [5.5 8.5 7.0]
(matrix-vector-mul identity-5x5 vector2)
; => [0.6 0.7 0.8 0.9 1.0]

; Um Matrix-Vektor-Multiplikation unter Benutzung unserer anderer Funktionen zu implementieren,
; muss man zunächst eine transpose Funktion bauen, die eine Matrix transponiert

(defn transpose [matrix]
  (apply mapv vector matrix))

(transpose matrix1)
; => [[1 6 2] [2 7 3] [3 8 4] [4 9 5] [5 1 6]]

(defn matrix-mul [left right]
  (mapv (partial matrix-vector-mul left) (transpose right))
  )

(def matrix2
  [[1 2 3 4 5]
   [6 7 8 9 1]
   [2 3 4 5 6]
   [7 8 9 1 2]
   [3 4 5 6 7]])

(matrix-mul matrix2 identity-5x5)
; => [[1 6 2 7 3] [2 7 3 8 4] [3 8 4 9 5] [4 9 5 1 6] [5 1 6 2 7]]
(matrix-mul identity-5x5 matrix2)
; => [[1 6 2 7 3] [2 7 3 8 4] [3 8 4 9 5] [4 9 5 1 6] [5 1 6 2 7]]

(matrix-mul matrix2 matrix1)
; => [[19 64 28 73 37] [25 85 37 97 49] [31 106 46 121 61] [37 127 55 145 73] [25 85 37 97 49]]

; Was ist das Problem an diesen Implementierungen?

(time (matrix-mul identity-5x5 matrix2))
; "Elapsed time: 0.278691 msecs"
; => [[1 6 2 7 3] [2 7 3 8 4] [3 8 4 9 5] [4 9 5 1 6] [5 1 6 2 7]]

(time (matrix-mul matrix2 matrix1))
; "Elapsed time: 0.2219 msecs"
; => [[19 64 28 73 37] [25 85 37 97 49] [31 106 46 121 61] [37 127 55 145 73] [25 85 37 97 49]]

; Was passiert, wenn wir wirklich große Matrizen nehmen?

(defn random-matrix [rows, columns]
  (vec (repeat rows (vec (repeat columns (rand))))))

(random-matrix 3 2)
; =>
; [[0.2709231708880665 0.2709231708880665]
; [0.2709231708880665 0.2709231708880665]
; [0.2709231708880665 0.2709231708880665]]

(time (def result-pure (matrix-mul (random-matrix 100 300) (random-matrix 300 200))))
; "Elapsed time: 817.241124 msecs"

; Für wirklich große Matrizen, ist diese Berechnungszeit nicht akzeptabel

(require '[uncomplicate.neanderthal [core :refer :all] [native :refer :all]])

(def n-matrix-1 (dge 100 300 (repeat (* 100 300) (rand))))
(def n-matrix-2 (dge 300 200 (repeat (* 300 200) (rand))))

(time (def result-n (mm n-matrix-1 n-matrix-2)))
; "Elapsed time: 0.980913 msecs"

; Man erkennt direkt, dass diese Berechnung mit der neanderthal library geschwindigkeitsmäßig
; in einer ganz anderen Liga spielen als die naive Implementierung

; Diese Geschwindigkeit kann noch erhöht werden, wenn man eine gute Graphikkarte
; hat. Dann kann man die Berechnung auf die Graphikkarte verschieben, was durch deren besondere
; auf parallele Berechnung ausgelegte Architektur zu noch viel schnelleren Berechnungen führt

(use 'clojure.core.matrix)
(use 'clojure.core.matrix.operators)

(def m-matrix-1 (matrix (random-matrix 100 300)))
(def m-matrix-2 (matrix (random-matrix 300 200)))

(time (def result-m (mmul m-matrix-1 m-matrix-2)))
; "Elapsed time: 524.244884 msecs"

; Diese Berechnung zeigt, dass auch die standard Matrix library schneller als die
; naive Implementierung ist aber nicht an des Geschwindigkeitsniveau der
; Neandethaler library herankommt und das bereits ohne Nutzung einer Graphikkarte

; Siehe ml_cortex.clj
