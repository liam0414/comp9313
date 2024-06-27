# written by Liam Chen z3278107
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.compat import jobconf_from_env

class proj1(MRJob):
    def mapper_init(self):
        self.tmp = {}  # in-mapper combiner

    def mapper(self, _, line):
        station, date, humidity = line.split(',')
        station = station.strip()
        date = date.strip()
        humidity = float(humidity)
        k = station + "#" + date
        self.tmp[k] = self.tmp.get(k, []) + [humidity]

    def mapper_final(self):
        for k, reading_array in self.tmp.items():
            station, date = k.split("#")
            yield f"{station}#{date}", (sum(reading_array), len(reading_array))
            yield f"{station}#9999", (sum(reading_array), len(reading_array))

    def reducer_init(self):
        self.overall_avg = 0.0
        self.tau = float(jobconf_from_env('myjob.settings.tau'))

    def reducer(self, key, values):
        station, date = key.split("#")
        overall_sum = 0.0
        overall_count = 0
        if date == "9999":
            for daily_sum, daily_count in values:
                overall_sum += daily_sum
                overall_count += daily_count
            self.overall_avg = overall_sum / overall_count
        else:
            for daily_sum, daily_count in values:
                overall_sum += daily_sum
                overall_count += daily_count
            gap = abs(overall_sum / overall_count - self.overall_avg)
            if gap > self.tau:
                yield station, f"{date},{gap}"

    SORT_VALUES = True

    JOBCONF = {
        'map.output.key.field.separator': '#',
        'mapreduce.partition.keypartitioner.options': '-k1,1',  # partitioning
        'mapreduce.job.output.key.comparator.class': 'org.apache.hadoop.mapreduce.lib.partition.KeyFieldBasedComparator',
        'mapreduce.partition.keycomparator.options': '-k1,1 -k2,2r'  # secondary sort
    }


if __name__ == '__main__':
    proj1.run()
