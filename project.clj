(defproject data-science "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [
                 [org.clojure/clojure "1.10.0"]
                 [uncomplicate/neanderthal "0.40.0"]
                 [org.bytedeco/mkl-platform-redist "2020.3-1.5.4"]
                 ]
  :main ^:skip-aot data-science.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
