# final submission by z3278107 Liam Chen
# adding comments
import sys
from pyspark import SparkConf, SparkContext
import math

class project3:
    def run(self, inputpath, outputpath, k):
        def calculate_prefix_length(r):
            threshold = float(tau.value)
            # http://www.cse.unsw.edu.au/~lxue/WWW08.pdf lemma 3
            # threshold optimization for prefix length calculation
            p = len(r) - 2 * len(r) * threshold / (1 + threshold) + 1
            # p = len(r) - math.ceil(len(r) * float(tau)) + 1 lecture calculation is not used
            return p if p <= len(r) else len(r)

        def jaccard_similarity(list1, list2):
            # calculation of jaccard similarity
            set1 = set(list1)
            set2 = set(list2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return 1.0 * intersection / union

        def generate_pairs(iterables):
            # only pairs that meet the conditions will be returned
            cp = set()
            for item_x in iterables:
                xr = item_x[3]
                x_id = item_x[0]
                x_date = item_x[1]
                for item_y in iterables:
                    yr = item_y[3]
                    y_id = item_y[0]
                    y_date = item_y[1]
                    # if two records are from same year, they are not evaluated
                    if x_date[-4:] == y_date[-4:] or int(x_id) >= int(y_id):
                        continue
                    res = jaccard_similarity(xr, yr)
                    if res >= float(tau.value):
                        cp.add(((x_id, y_id), res))
            return cp

        conf = SparkConf()
        sc = SparkContext(conf=conf)
        tau = sc.broadcast(k) # broadcast the variable for faster access
        rdd = sc.textFile(inputpath)
        # Stage 1: sort tokens by frequency
        sorted_frequency = rdd.map(lambda x: (x.split(',')[1], 1)) \
            .reduceByKey(lambda a, b: a + b) \
            .sortBy(lambda x: (x[1], x[0])).zipWithIndex().map(lambda x: (x[0][0], x[1]))
        # order map is the sorted frequency of each token, it is buffered in memory
        order_map = sc.broadcast(sorted_frequency.collectAsMap())
        # Stage 2: find 'similar' id pairs
        rdd_grouped = rdd.map(lambda x: x.split(',')) \
            .map(lambda x: ((x[0], x[3].split(' ', 1)[0].split('/', 1)[1]), x[1])) \
            .reduceByKey(lambda a, b: a + ',' + b) \
            .map(lambda x: (x[0], list(set(x[1].split(',')))))
        # order sets by length, within each set, order the items in accordance with order_map, sorted_records
        sorted_rdd = rdd_grouped.sortBy(lambda x: len(x[1])) \
            .map(lambda x: (x[0], sorted(x[1], key=lambda y: order_map.value[y])))
        # calculate prefix length and partition using prefixes
        prefix_length_rdd = sorted_rdd.map(lambda x: (x[0], (calculate_prefix_length(x[1]), x[1])))
        prefix_rdd = prefix_length_rdd.map(lambda x: ((x[0], x[1]), x[1][1][:round(x[1][0])])).flatMapValues(lambda x: x) \
            .map(lambda x: ((x[1]), [x[0][0][0], x[0][0][1], x[0][1][0], x[0][1][1]]))
        # Stage 3: group by and find pairs
        # then jaccard similarities are calculated for each valid pair
        group_prefix_rdd = prefix_rdd.groupByKey() \
            .map(lambda x: (x[0], list(x[1]))).map(lambda x: generate_pairs(x[1])) \
            .flatMap(lambda x: x).distinct().sortBy(lambda x: int(x[0][0])) \
            .map(lambda x: '(' + x[0][0] + ',' + x[0][1] + '):' + str(x[1]))
        # collect the results and output them to path
        group_prefix_rdd.coalesce(1).saveAsTextFile(outputpath)


if __name__ == '__main__':
    project3().run(sys.argv[1], sys.argv[2], sys.argv[3])


