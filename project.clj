(defproject data-science "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [
                 [org.clojure/clojure "1.10.0"]
                 [uncomplicate/neanderthal "0.40.0"]
                 [org.bytedeco/mkl-platform-redist "2020.3-1.5.4"]
                 [net.mikera/core.matrix "0.62.0"]
                 [thinktopic/cortex "0.9.22"]
                 [thinktopic/experiment "0.9.22"]
                 [org.clojure/tools.cli "0.3.5"]
                 [thinktopic/think.tsne "0.1.1"]
                 ;;If you need cuda 8...
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 ;;If you need cuda 7.5...
                 ;[org.bytedeco.javacpp-presets/cuda "7.5-1.2"]
                 [clojure-csv/clojure-csv "2.0.1"]
                 [incanter "1.9.3"]
                 ]
  :exclusions [[org.jcuda/jcuda-natives :classifier "apple-x86_64"]
               [org.jcuda/jcublas-natives :classifier "apple-x86_64"]]
  :jvm-opts ^:replace [#_"--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"]
  :main ^:skip-aot data-science.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})

