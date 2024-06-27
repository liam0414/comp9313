# written by Liam z3278107 first working version
# 22/10/2023 fix sorting, fix file output
# 25/10/2023 fix sorting on date
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from itertools import combinations
import sys
import heapq


def find_lists(x):
    return [list(sublist) for sublist in combinations(x, 3)]


def takeTopK(values, k):
    topk = []
    heapq.heapify(topk)
    for e in values:
        p = Pair(e[0], e[1])
        heapq.heappush(topk, p)
        if len(topk) > int(k):
            heapq.heappop(topk)
    sorted_topk = sorted(topk, reverse=True)
    output = []
    for e in sorted_topk:
        output.append(e.items + "," + str(e.support_ratio))
    return output


class Pair:
    def __init__(self, items, support_ratio):
        self.items = items
        self.support_ratio = support_ratio

    def __lt__(self, other):
        if self.support_ratio != other.support_ratio:
            return self.support_ratio < other.support_ratio
        else:
            return self.items > other.items


class Project2:
    def run(self, inputPath, outputPath, k):
        spark = SparkSession.builder.master("local").appName("project2_df").getOrCreate()
        # define the schema for dataframe
        schema = StructType([
            StructField("id", IntegerType(), False),
            StructField("des", StringType(), False),
            StructField("quantity", IntegerType(), False),
            StructField("invoice_time", StringType(), False),
            StructField("unit_price", FloatType(), False),
        ])
        df = spark.read.option("delimiter", ",").csv(inputPath, schema=schema)

        # select the relevant columns, A,3,12/1/2010  8:26:00 AM,2.5 -> ('1', '1/2010'), 'A'
        col_df = df.select(df['id'], split(split(df['invoice_time'], " ").getItem(0), "/", 2).getItem(1).alias('date'), df['des'])
        # items are grouped, with id col discarded ('1/2010', ['A', 'B', 'C']), list is sorted
        agg_df = col_df.groupBy("id", "date").agg(collect_list('des').alias("items"))\
            .withColumn("items", sort_array("items")).drop("id")

        # generating all the 3-item-set, each 3-set has count of 1 in this step
        find_lists_udf = udf(find_lists, ArrayType(ArrayType(StringType())))

        # find all the sublists using udf
        lists_df = agg_df.withColumn("three_items", find_lists_udf(agg_df.items)).drop(agg_df['items'])

        # count the total in support calculation
        count_df = lists_df.groupBy("date").agg(count("*").alias("total"))
        total_df = lists_df.join(count_df, "date")

        # flat three items list
        exploded_df = total_df.select("date", explode("three_items").alias("items"), "total")

        # calculate the frequency for each item set and group by date and items
        # |  date|    items|support|total|
        # |1/2010|[A, B, C]|      1|    2|
        result_df = exploded_df.groupBy("date", "items").agg(count("date").alias("support"), first("total").alias("total"))
        support_df = result_df.withColumn("items", concat(lit("("), concat_ws("|", "items"), lit(")")))
        ratio_df = support_df.withColumn("support_ratio", col("support") / col("total")).drop("support", "total")\
             .groupBy("date").agg(collect_list(struct('items', 'support_ratio')).alias('value'))
        
        ratio_df = ratio_df.withColumn("year", split(col("date"), "/")[1].cast("int"))
        ratio_df = ratio_df.withColumn("month", split(col("date"), "/")[0].cast("int"))
        # sorting the results
        top_k_udf = udf(lambda x, k: takeTopK(x, k), ArrayType(StringType()))
        sorted_results = ratio_df.orderBy("year", "month")\
            .withColumn('value', top_k_udf('value', lit(k)).alias("top_k_data"))\
            .select('date', explode('value').alias('sorted_values'))
        result = sorted_results.coalesce(1).select(concat(col("date"), lit(","), col("sorted_values")))
        result.write.text(outputPath)
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Wrong arguments")
        sys.exit(-1)
    Project2().run(sys.argv[1], sys.argv[2], sys.argv[3])

