from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster('local').setAppName('spark_process_data')
sc = SparkContext(conf = conf)

text = sc.textFile("CaPUMS5full.csv")

x = text.map(lambda line: map(int, line.split(',')[:5]) + map(int, line.split(',')[6:-2]))

y = text.map(lambda line: int(line.split(',')[5]))

x.saveAsTextFile("training_X.txt")

y.saveAsTextFile("training_Y.txt")

# took 6.432831 s