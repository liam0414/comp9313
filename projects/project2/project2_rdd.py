# written by liam z3278107, refactored
# 22/10/2023 fix sorting
# 25/10/2023 fix floating point precision
from pyspark import SparkContext, SparkConf
import sys
from itertools import combinations  # https://docs.python.org/3/library/itertools.html#module-itertools
from heapq import nsmallest  # https://docs.python.org/3/library/heapq.html
from datetime import datetime

def generate_lists(x):
    return [x[0], [list(sublist) for sublist in combinations(x[1], 3)]]


def parse_date(date_str):
    return datetime.strptime(date_str, "%m/%Y")


class Project2:

    def run(self, inputPath, outputPath, k):
        conf = SparkConf().setAppName("project2_rdd").setMaster("local")
        sc = SparkContext(conf=conf)

        # load rdd and broadcast k to all partitions
        rdd = sc.textFile(inputPath)
        limit = sc.broadcast(k)

        # extract cols as kv pair, for example
        # A,3,12/1/2010  8:26:00 AM,2.5 -> ('1', '1/2010'), 'A'
        rdd_part = rdd.map(lambda x: x.split(','))\
            .map(lambda x: ((x[0], x[3].split(' ', 1)[0].split('/', 1)[1]), x[1]))

        # items are grouped, with id discarded ('1/2010', ['A', 'B', 'C'])
        rdd_grouped = rdd_part.reduceByKey(lambda a, b: a + ',' + b)\
            .map(lambda x: (x[0][1], sorted(x[1].split(','))))

        # calculate the total in M and the result like below after flatMapValue
        # (('1/2010', 3), ['A', 'B', 'C']),
        # (('1/2010', 3), ['A', 'C', 'D']),
        # (('1/2010', 3), ['A', 'B', 'C', 'D'])
        aggregated_items = rdd_grouped.map(lambda x: (x[0], ([x[1]], 1)))\
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))\
            .map(lambda x: ((x[0], x[1][1]), x[1][0]))\
            .flatMapValues(lambda x: x)

        # generating all the 3-item-set, each 3-set has count of 1 in this step
        support_counts = aggregated_items.map(generate_lists)\
            .flatMapValues(lambda x: x)\
            .mapValues(lambda x: ('(' + '|'.join(map(str, x)) + ')', 1))

        # calculate the frequency for each set and reduceByKey
        support_results = support_counts.map(lambda x: ((x[0][0], x[1][0], x[0][1]), x[1][1]))\
            .reduceByKey(lambda a, b: a + b)\
            .map(lambda x: (x[0][0], (x[0][1], x[1] / x[0][2])))
        # sort the results using heapq, first by time (sortBy)
        # then by value in descending order
        # then by item set using alphabetical order and format results
        sorted_support_results = support_results.groupByKey()\
            .sortBy(lambda x: parse_date(x[0]))\
            .mapValues(lambda x: nsmallest(int(limit.value), x, key=lambda y: (-y[1], y[0]))) \
            .flatMapValues(lambda x: x)\
            .map(lambda x: x[0] + ',' + x[1][0] + ',' + str(x[1][1]))

        # # collect all results from different partitions
        sorted_support_results.coalesce(1).saveAsTextFile(outputPath)
        sc.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Wrong arguments")
        sys.exit(-1)
    Project2().run(sys.argv[1], sys.argv[2], sys.argv[3])

