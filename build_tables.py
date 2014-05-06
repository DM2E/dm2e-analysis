import csv
import re
import math
import json
import string
import logging
from operator import itemgetter


def clean_string(s):
    s = re.sub("^\?", "", s)
    s = re.sub("^<", "", s)
    s = re.sub(">$", "", s)
    return s


class NumberSeries:

    def __init__(self, key):
        logging.basicConfig(level=logging.DEBUG)
        self.key = clean_string(key)
        self.values = []

    def append(self, number):
        self.values.append(float(number))

    def min(self):
        return min(self.values)

    def max(self):
        return max(self.values)

    def size(self):
        return len(self.values)

    def sum(self):
        return sum(self.values)

    def amean(self):
        return self.sum() / float(self.size())

    def gmean(self):
        logsum = 0
        for val in self.values:
            logsum += math.log(val)
        return math.exp(logsum/self.size())

    def hmean(self):
        return self.size() / sum(1. / val for val in self.values)

    def to_dict(self):
        return {
            '?key': self.key,
            '?max': self.max(),
            '?min': self.min(),
            '?amean': self.amean(),
            '?gmean': self.gmean(),
            '?hmean': self.hmean(),
            '?size': self.size()}


class FrequencyAnalysis:

    def __init__(self, table, label, relate, max_number=20, logarithmic=False, add_to=0):
        self.table = table
        self.label = label
        self.relate = relate
        self.logarithmic = logarithmic
        if max_number > 0:
            self.max_number = max_number
        else:
            self.max_number = 1000
        self.add_to = add_to


class TableBuilder:

    def __init__(self, **kwargs):
        logging.basicConfig(level=logging.DEBUG)
        self.vis_dir = kwargs['vis_dir']
        self.sparql_dir = kwargs['sparql_dir']
        self.analyses_dir = kwargs['analyses_dir']
        self.template_dir = kwargs['template_dir']

        self.prefixes = {}
        self.init_prefixes()

        self.datasets = []
        self.per_dataset_stats = {}
        self.init_per_dataset_stats()
        self.datasets_plus_total = ['total']
        for dataset in self.datasets:
            self.datasets_plus_total.append(dataset)
        # self.count_per_dataset_stats()
        # self.write_per_dataset_stats()

    def init_prefixes(self):
        prefix_file = self.sparql_dir + 'prefixes.rq'
        pat = re.compile("PREFIX ([^:]+:) <([^>]+)>")
        with open(prefix_file, "rb") as rqin:
            for line in rqin:
                x = re.findall(pat, line)
                if len(x) > 0:
                    self.prefixes[x[0][0]] = x[0][1]

    def shorten_url(self, url):
        url = clean_string(url)
        for prefix, long in self.prefixes.iteritems():
            if url.startswith(long):
                url = url.replace(long, prefix)
                break
        return url

    def init_per_dataset_stats(self):
        with open(self.template_dir + 'numbers-per-dataset.tsv', 'rb') as csvin:
            reader = csv.DictReader(csvin)
            for row in reader:
                self.per_dataset_stats[row['?handle']] = row
                self.datasets.append(row['?handle'])

    def count_per_dataset_stats(self):
        for dataset in self.datasets:
            stats = self.per_dataset_stats[dataset]
            # ?nr_stmt_total
            stats['?nr_stmt_total'] = self.get_second_line_as_number(dataset, 'triples-per-dataset')
            # ?nr_diff_subj
            stats['?nr_diff_subj'] = self.count_lines(dataset, 'subjects-per-dataset')
            # ?nr_diff_obj
            stats['?nr_diff_obj'] = self.count_lines(dataset, 'objects-per-dataset')
            # ?nr_diff_pred
            stats['?nr_diff_pred'] = self.count_lines(dataset, 'predicates-per-dataset')
            # ?nr_stmt_poequal
            stats['?nr_stmt_poequal'] = self.count_lines(dataset, 'predicate-object-equal-statements')
            # ?nr_diff_host
            stats['?nr_diff_host'] = self.count_lines(dataset, 'hostnames')
            # ?nr_diff_baseurl
            stats['?nr_diff_baseurl'] = self.count_lines(dataset, 'hostnames')  # TODO
            # ?nr_diff_license
            stats['?nr_diff_license'] = self.count_lines(dataset, 'license')
            # ?nr_diff_types
            stats['?nr_diff_types'] = self.count_lines(dataset, 'types')
            # ?nr_diff_types
            stats['?nr_diff_dctypes'] = self.count_lines(dataset, 'dctypes')
            # ?nr_untyped_subj
            stats['?nr_untyped_subj'] = self.count_lines(dataset, 'untyped')
            # ?nr_stmt_literal
            stats['?nr_stmt_literal'] = self.get_line_and_column(dataset, 'literal-statements', column=1)
            # ?perc_stmt_literal
            stats['?perc_stmt_literal'] = 100 * stats['?nr_stmt_literal'] / float(stats['?nr_stmt_total'])
            # ?perc_stmt_uri
            stats['?perc_stmt_uri'] = 100 - stats['?perc_stmt_literal']
            # ?perc_stmt_poequal
            stats['?perc_stmt_poequal'] = 100 * stats['?nr_stmt_poequal'] / float(stats['?nr_stmt_total'])
            # ?perc_stmt_unique
            stats['?perc_stmt_unique'] = 100 - stats['?perc_stmt_poequal']
            # ?nr_stmt_uri
            stats['?nr_stmt_uri'] = stats['?nr_stmt_total'] - stats['?nr_stmt_literal']
            # ?perc_unique_obj
            stats['?perc_unique_obj'] = 100 * stats['?nr_diff_obj'] / float(stats['?nr_stmt_uri'])
        self.per_dataset_stats['total'] = {}
        for ds, ds_dict in self.per_dataset_stats.iteritems():
            for key_stat in ds_dict.keys():
                if not key_stat in self.per_dataset_stats['total']:
                    self.per_dataset_stats['total'][key_stat] = ds_dict[key_stat]
                else:
                    self.per_dataset_stats['total'][key_stat] += ds_dict[key_stat]

    def get_second_line_as_number(self, dataset, analysis):
        ret = 0
        with open("%s/%s-count-%s.rq.tsv" % (self.analyses_dir, dataset, analysis), 'rb') as tsvin:
            tsvin.readline()
            ret = int(tsvin.readline().strip())
        return ret

    def get_line_and_column(self, dataset, analysis, line=1, column=0):
        ret = 0
        with open("%s/%s-count-%s.rq.tsv" % (self.analyses_dir, dataset, analysis), 'rb') as tsvin:
            for idx, row in enumerate(csv.reader(tsvin, delimiter='\t')):
                if idx == 0:
                    continue
                if idx == line:
                    ret = int(row[column])
                    break
        return ret

    def count_lines(self, dataset, analysis):
        with open("%s/%s-count-%s.rq.tsv" % (self.analyses_dir, dataset, analysis), 'rb') as tsvin:
            total = -1
            for line in tsvin:
                total += 1
        return total

    def write_per_dataset_stats(self):
        field_sequence = []
        dataset_sequence = []
        with open(self.template_dir + 'numbers-per-dataset.tsv', 'rb') as csvin:
            firstline = True
            for line in csvin:
                seq = line.strip().split(',')
                if firstline:
                    field_sequence = seq
                    firstline = False
                else:
                    dataset_sequence.append(seq[0])
        with open("%s/numbers-per-dataset.tsv" % self.analyses_dir, 'wb') as csvout:
            writer = csv.DictWriter(csvout, field_sequence)
            writer.writeheader()
            for dataset in dataset_sequence:
                writer.writerow(self.per_dataset_stats[dataset])

    def calculate_number_series(self, analysis, key_group, key_number):
        ret_tables = {}
        total_series = {}
        for dataset in self.datasets:
            logging.debug("Average '%s' for dataset '%s'" % (analysis, dataset))
            dataset_series = {}
            with open("%s/%s-count-%s.rq.tsv" % (self.analyses_dir, dataset, analysis), 'rb') as tsvin:
                reader = csv.DictReader(tsvin, delimiter='\t')
                for row in reader:
                    if not row[key_group] in dataset_series.keys(): dataset_series[row[key_group]] = NumberSeries(row[key_group])
                    if not row[key_group] in total_series.keys(): total_series[row[key_group]] = NumberSeries(row[key_group])
                    dataset_series[row[key_group]].append(row[key_number])
                    total_series[row[key_group]].append(row[key_number])
                ret_tables[dataset] = []
                for series in dataset_series.itervalues(): ret_tables[dataset].append(series.to_dict())
        ret_tables['total'] = []
        for series in total_series.itervalues(): ret_tables['total'].append(series.to_dict())

        return ret_tables

    def calculate_average(self, analysis, key_group, key_number):
        logging.debug("Average '%s'" % analysis)
        ret_tables = self.calculate_number_series(analysis, key_group, key_number)

        for dataset in ret_tables.keys():
            # fname = self.output_dir + 'average_' + analysis + '_by_' + key_group.replace("?", "") + "_" + dataset + ".rq.tsv"
            fname = "%s/%s_average_%s_by_%s.rq.tsv" % (self.analyses_dir, dataset, analysis, clean_string(key_group))
            with open(fname, "wb") as csvout:
                writer = csv.DictWriter(csvout, "?key ?size ?min ?max ?amean ?gmean ?hmean".split(" "), delimiter="\t")
                writer.writeheader()
                for row in ret_tables[dataset]:
                    writer.writerow(row)
            print(fname)

    def extract_data_from_table(self, analysis, col_select, legend=[], sort_col=0, delimiter='\t'):
        ret = []
        fname = "%s/%s" % (self.analyses_dir, analysis)
        with open(fname, 'rb') as tsvin:
            col_names = []
            col_names = re.compile('[,\t]').split(tsvin.readline().strip())
            for col_select_idx, this_col in enumerate(col_select):
                try:
                    idx = int(float(this_col))
                    # print col_names[idx]
                    col_select[col_select_idx] = col_names[idx]
                except ValueError:
                    pass
        with open(fname, "rb") as tsvin:
            reader = csv.DictReader(tsvin, delimiter=delimiter)
            i = 0
            for row in reader:
                i += 1
                row_arr = []
                for col in col_select:
                    val = row[col]
                    try:
                        val = float(val)
                    except ValueError:
                        val = self.shorten_url(val)
                    row_arr.append(val)
                ret.append(row_arr)
                if 0 == i % 100000:
                    logging.debug("%s: %s" % (analysis, i))
        ret = sorted(ret, key=itemgetter(sort_col))
        if [] == legend:
            legend = col_select
        ret.insert(0, legend)
        return ret

    def run_template(self, out_name, **kwargs):
        tpl_path = self.template_dir + 'gchart.html'
        out_path = self.vis_dir + out_name + '_' + kwargs['vis'] + '.html'
        tpl = string.Template(open(tpl_path, 'rb').read())
        repl = {}
        for tplvar_key, tplvar_val in kwargs.iteritems():
            repl[tplvar_key] = json.dumps(tplvar_val)
        out_str = tpl.substitute(repl)
        with open(out_path, 'wb') as out_file:
            out_file.write(out_str)

    def sum_up_data(self, totals, legend, key_idx=0, sort_idx=1, logarithmic=False, add_to=0):
        merge = {}
        i = 0
        for row in totals:
            i += 1
            key = row[key_idx]
            if row == legend:
                continue
            if key in merge:
                for field_idx, field_val in enumerate(row):
                    if field_idx != key_idx:
                        merge[key][field_idx] += row[field_idx]
            else:
                merge[key] = []
                for x in row:
                    merge[key].append(x)
            if 0 == i % 1:
                pass
                # print i
        if logarithmic:
            for k,v in merge.iteritems():
                if v[sort_idx] > 0:
                    merge[k][sort_idx] = math.log(float(v[sort_idx]))
        if add_to > 0:
            for k,v in merge.iteritems():
                v[sort_idx] += add_to

        ret = sorted(merge.values(), key=itemgetter(sort_idx), reverse=True)
        ret.insert(0, legend)
        return ret

    def relativize_data(self, data, rel, legend, rel_idx=1):
        ret = []
        for row in data:
            if row == legend:
                continue
            new_row = []
            for cell_idx, cell_val in enumerate(row):
                if cell_idx == rel_idx:
                    cell_val /= float(rel)
                    cell_val *= 100
                new_row.append(cell_val)
            ret.append(new_row)
        ret.insert(0, legend)
        return ret

    def visualize_global_values(self, long_running=False):
        logging.debug("Visualizing Global Values ")
        most_frequent_analyses = [
            FrequencyAnalysis('baserurls', 'Base URL', ['?nr_diff_subj', '?nr_stmt_total']),
            FrequencyAnalysis('dctypes', 'DC Types', ['?nr_diff_subj']),
            FrequencyAnalysis('hostnames', 'Hostnames', ['?nr_diff_subj']),
            FrequencyAnalysis('license', 'Licenses', ['?nr_stmt_total']),
            FrequencyAnalysis('literal-statements', 'Literal Statements', ['?nr_stmt_total']),
            FrequencyAnalysis('predicates-per-dataset', 'Predicates', ['?nr_stmt_total']),
            FrequencyAnalysis('predicates-per-dataset-longtail', 'Predicates', ['?nr_stmt_total'], max_number=-1, logarithmic=False, add_to=50000),
            FrequencyAnalysis('types', 'RDF Types', ['?nr_diff_subj']),
            FrequencyAnalysis('types-longtail', 'RDF types', ['?nr_stmt_total'], max_number=-1, logarithmic=False, add_to=50000),
        ]
        # long running
        if long_running:
            # most_frequent_analyses.append(FrequencyAnalysis('untyped', 'Untyped', ['?nr_diff_subj']))
            # most_frequent_analyses.append(FrequencyAnalysis('subjects-per-dataset', 'Subjects', ['?nr_diff_subj']))
            most_frequent_analyses.append(FrequencyAnalysis('objects-per-dataset', 'Objects', ['?nr_diff_obj']))
        for mf_analysis in most_frequent_analyses:
            totals = []
            legend = [mf_analysis.label, 'Frequency (absolute)']
            for dataset in self.datasets:
                logging.debug("Most frequent %s in %s" % (mf_analysis.table, dataset))
                tsv_path = "%s-count-%s.rq.tsv" % (dataset,  mf_analysis.table)
                column_select = [0, '?no']
                logging.debug("Start extracting")
                data = self.extract_data_from_table(tsv_path, column_select, legend=legend, sort_col=1)
                logging.debug("Done extracting")
                logging.debug("Start summing up")
                absol = self.sum_up_data(data, legend, logarithmic=mf_analysis.logarithmic, add_to=mf_analysis.add_to)
                logging.debug("Done summing up")
                self.run_template("%s_most_frequent_%s_absolute" % (dataset, mf_analysis.table), vis='bar', data=absol[0:mf_analysis.max_number], title=mf_analysis.label)
                self.run_template("%s_most_frequent_%s_absolute" % (dataset, mf_analysis.table), vis='column', data=absol[0:mf_analysis.max_number], title=mf_analysis.label)
                self.run_template("%s_most_frequent_%s_absolute" % (dataset, mf_analysis.table), vis='pie', data=absol[0:mf_analysis.max_number], title=mf_analysis.label)
                self.run_template("%s_most_frequent_%s_absolute" % (dataset, mf_analysis.table), vis='hist', data=absol, title=mf_analysis.label)
                for relation in mf_analysis.relate:
                    relat = self.relativize_data(absol, self.per_dataset_stats[dataset][relation], legend)
                    self.run_template("%s_most_frequent_%s_relative_to_%s" % (dataset, mf_analysis.table, clean_string(relation)), vis='pie', data=relat[0:mf_analysis.max_number], title=mf_analysis.label)
                    self.run_template("%s_most_frequent_%s_relative_to_%s" % (dataset, mf_analysis.table, clean_string(relation)), vis='bar', data=relat[0:mf_analysis.max_number], title=mf_analysis.label)
                    self.run_template("%s_most_frequent_%s_relative_to_%s" % (dataset, mf_analysis.table, clean_string(relation)), vis='column', data=relat[0:mf_analysis.max_number], title=mf_analysis.label)
                    self.run_template("%s_most_frequent_%s_relative_to_%s" % (dataset, mf_analysis.table, clean_string(relation)), vis='hist', data=relat, title=mf_analysis.label)
                [totals.append(x) for x in data]
            total_absol = self.sum_up_data(totals, legend, logarithmic=mf_analysis.logarithmic, add_to=mf_analysis.add_to)
            self.run_template("%s_most_frequent_%s_absolute" % ('total', mf_analysis.table), vis='column', data=total_absol[0:mf_analysis.max_number], title=mf_analysis.label)
            self.run_template("%s_most_frequent_%s_absolute" % ('total', mf_analysis.table), vis='bar', data=total_absol[0:mf_analysis.max_number], title=mf_analysis.label)
            self.run_template("%s_most_frequent_%s_absolute" % ('total', mf_analysis.table), vis='pie', data=total_absol[0:mf_analysis.max_number], title=mf_analysis.label)
            self.run_template("%s_most_frequent_%s_absolute" % ('total', mf_analysis.table), vis='hist', data=total_absol, title=mf_analysis.label)
            for relation in mf_analysis.relate:
                relat = self.relativize_data(total_absol, self.per_dataset_stats['total'][relation], legend)
                self.run_template("%s_most_frequent_%s_relative_to_%s" % ('total', mf_analysis.table, clean_string(relation)), vis='pie', data=relat[0:mf_analysis.max_number], title=mf_analysis.label)
                self.run_template("%s_most_frequent_%s_relative_to_%s" % ('total', mf_analysis.table, clean_string(relation)), vis='bar', data=relat[0:mf_analysis.max_number], title=mf_analysis.label)
                self.run_template("%s_most_frequent_%s_relative_to_%s" % ('total', mf_analysis.table, clean_string(relation)), vis='column', data=relat[0:mf_analysis.max_number], title=mf_analysis.label)
                self.run_template("%s_most_frequent_%s_relative_to_%s" % ('total', mf_analysis.table, clean_string(relation)), vis='hist', data=relat, title=mf_analysis.label)

    def visualize_global_sums(self):
        pie_bar_number = {
            "?inst_type": "",
            "?nr_stmt_total": "Number of total statements",
            "?nr_diff_subj": "Number of different subjects",
            "?nr_diff_pred": "Number of different predicates",
            "?nr_diff_obj": "Number of different objects",
            "?nr_stmt_poequal": "Number of P-O-equal statements",
            "?nr_stmt_literal": "Number of literal statements",
            "?nr_diff_host": "Number of different hostnames",
            "?nr_diff_baseurl": "Number of different base URL",
            "?nr_diff_license": "Number of different licenses",
            "?nr_stmt_literal": "Number of literal statements",
            "?nr_diff_types": "Number of different rdf:types",
            "?nr_diff_dctypes": "Number of different dc:types",
            "?nr_untyped_subj": "Number of resources w/o rdf:type",
            "?avg_stmt_per_resource": "Arith. Avg of statements per subject",
            "?perc_stmt_poequal": "Percentage of P-O-equal statements",
            "?perc_stmt_literal": "Percentage of Literal statements",
            "?perc_unique_obj": "Percentage of One-off References ot a URI [TODO]",
        }
        for csv_key, legend_key in pie_bar_number.iteritems():
            data = tb.extract_data_from_table('numbers-per-dataset.tsv', ['?handle', csv_key], legend=['Dataset', legend_key], sort_col=1, delimiter=',')
            tb.run_template('total_' + clean_string(csv_key), vis='bar', data=data, title=legend_key)
            tb.run_template('total_' + clean_string(csv_key), vis='pie', data=data, title=legend_key)
            tb.run_template('total_' + clean_string(csv_key), vis='hist', data=data, title=legend_key)

        stack_bar_number = {
            'Literal vs. Resource Statements': {
                "?perc_stmt_literal": "Literal Stmt",
                "?perc_stmt_uri": "Resource Stmt",
            },
            'Redundant vs. Unique Statements': {
                "?perc_stmt_unique": "Unique Stmt",
                "?perc_stmt_poequal": "P-0-Equals Stmts",
            }
        }
        for label, csv_dict in stack_bar_number.iteritems():
            csv_keys = ['?handle']
            legend_keys = ['Dataset']
            for csv_key, legend_key in csv_dict.iteritems():
                csv_keys.append(csv_key)
                legend_keys.append(legend_key)
            data = tb.extract_data_from_table('numbers-per-dataset.tsv', csv_keys, legend=legend_keys, sort_col=1, delimiter=',')
            tb.run_template('total_' + clean_string(label), vis='stack-bar', data=data, title=label)
            tb.run_template('total_' + clean_string(label), vis='pie', data=data, title=label)

    def visualize_average_table(self, table_name, label, outname):
        # numcol_legend = {
        #     "?size": "Total Number",
        #     "?min": "Minimum",
        #     "?max": "Maximum",
        #     "?amean": "Arith. Mean",
        #     "?gmean": "Geom. Mean",
        #     "?hmean": "Harm. Mean",
        # }
        data_minmax = tb.extract_data_from_table(table_name, ['?key', '?min', '?max'], legend=['Value', 'Min', 'Max'], sort_col=1)
        tb.run_template(outname + '_min_max', vis='bar', data=data_minmax, title='Min/Max for ' + label)

        data_avg = tb.extract_data_from_table(table_name, ['?key', '?amean', '?gmean', '?hmean'], ['Value', 'Arith. Mean', 'Geom. Mean', 'Harm. Mean'])
        tb.run_template(outname + '_avg', vis='bar', data=data_avg, title='Average for ' + label)

    def visualize_averages(self):
        analysis_legend = {
            'average_predicate-object-equal-statements_by_predicate': 'P-O-Equal Statemenets by Predicate',
            'average_statements-per-resource-and-type_by_dctype': 'Statements per Resource by dc:type',
            'average_statements-per-resource-and-type_by_type': 'Statements per Resource by rdf:type',
            # 'average_ranges-per-property_by_range', 'Average Number of rdfs:range per rdfs:range',
        }
        for dataset in self.datasets_plus_total:
            for analysis, legend in analysis_legend.iteritems():
                logging.debug("Dataset %s: %s" % (dataset, analysis))
                tb.visualize_average_table(
                    '%s_%s.rq.tsv' % (dataset, analysis),
                    legend,
                    "%s_%s" % (dataset, analysis))

    def visualize_per_dctype(self):
        ret = []
        dctypes = [
            '<http://onto.dm2e.eu/schemas/dm2e/Page>',
            '<http://purl.org/ontology/bibo/Book>',
            '<http://purl.org/ontology/bibo/Series>',
            '<http://onto.dm2e.eu/schemas/dm2e/Manuscript>',
            '<http://purl.org/ontology/bibo/Journal>',
            '<http://purl.org/spar/fabio/Article>',
            '<http://purl.org/spar/fabio/Article>',
            '<http://purl.org/ontology/bibo/Issue>',
            '<http://onto.dm2e.eu/schemas/dm2e/Paragraph>',
            '<http://purl.org/ontology/bibo/Letter>',
        ]
        header = ['dctype']
        for dataset in self.datasets:
            header.append(dataset)
        ret.append(header)
        for dctype in dctypes:
            row = []
            row.append(self.shorten_url(dctype))
            for dataset in self.datasets:
                absnr = int(self.grep_table_value(dataset + '-count-dctypes', grep_column=0, needle=dctype, return_column=1))
                # if (absnr > 0):
                    # absnr = math.log(absnr)
                row.append(absnr)
            ret.append(row)
        # tb.run_template('00-DC-Types-stack', vis='stack-bar', data=ret, title='Frequency of Dc Types per Dataset')
        tb.run_template('00-DC-Types', vis='bar', data=ret, title='Frequency of Dc Types per Dataset')

    def grep_table_value(self, analysis, grep_column, needle, return_column=1):
        with open("%s/%s.rq.tsv" % (self.analyses_dir, analysis), 'rb') as tsvin:
            ret = 0
            for row in csv.reader(tsvin, delimiter='\t'):
                if row[grep_column] == needle:
                    ret = row[return_column]
                    break
        return ret

    def calculate_long_tail(self, prop_list, analysis):
        per_dataset_long_tail = {'total':{}}
        for dataset in self.datasets:
            per_dataset_long_tail[dataset] = {}
            with open("%s/%s.lst" % (self.template_dir, prop_list), "r") as prop_list_file:
                for prop_line in prop_list_file:
                    prop = prop_line.strip()
                    val = int(self.grep_table_value("%s-count-%s" % (dataset, analysis), 0, prop))
                    per_dataset_long_tail[dataset][prop] = val
                    if not prop in per_dataset_long_tail['total']:
                        per_dataset_long_tail['total'][prop] = 0
                    per_dataset_long_tail['total'][prop] += val
        for dataset, long_tail in per_dataset_long_tail.iteritems():
            with open("%s/%s-count-%s-longtail.rq.tsv" % (self.analyses_dir, dataset, analysis), "w") as tsvout:
                out = csv.writer(tsvout, delimiter='\t')
                out.writerow(["?val", "?no"])
                for prop, val in sorted(long_tail.iteritems(), key=itemgetter(1), reverse=True):
                    out.writerow([prop, val])

    def calculate_good_bad_literals(self):
        per_dataset = {}
        for dataset in self.datasets:
            per_dataset[dataset] = {}
            try:
                with open("%s/%s-count-literal-good-bad.rq.tsv" % (self.analyses_dir, dataset), "r") as tsvin:
                    reader = csv.DictReader(tsvin, delimiter='\t')
                    out = {}
                    for row in reader:
                        out['nr_total'] = float(row['?no'])
                        out['nr_good'] = int(row['?no_good'])
                        out['nr_bad'] = int(row['?no_bad'])
                        out['nr_okay'] = out['nr_total'] - (out['nr_good'] + out['nr_bad'])
                        out['perc_good'] = 100 * out['nr_good'] / out['nr_total']
                        out['perc_bad'] = 100 * out['nr_bad'] / out['nr_total']
                        out['perc_okay'] = 100 * out['nr_okay'] / out['nr_total']
                        per_dataset[dataset] = out
            except IOError:
                pass
        per_dataset['total'] = {}
        for dataset in self.datasets:
            for k, v in per_dataset[dataset].iteritems():
                if not k in per_dataset['total']:
                    per_dataset['total'][k] = 0
                else:
                    per_dataset['total'][k] += v
        per_dataset['total']['perc_good'] = 100 * per_dataset['total']['nr_good'] / per_dataset['total']['nr_total']
        per_dataset['total']['perc_bad'] = 100 * per_dataset['total']['nr_bad'] / per_dataset['total']['nr_total']
        per_dataset['total']['perc_okay'] = 100 * per_dataset['total']['nr_okay'] / per_dataset['total']['nr_total']
        ret = [['Dataset', 'good', 'okay', 'bad']]
        for dataset in self.datasets_plus_total:
            if dataset in per_dataset and len(per_dataset[dataset]) > 0:
                ret.append([dataset, per_dataset[dataset]['perc_good'], per_dataset[dataset]['perc_okay'], per_dataset[dataset]['perc_bad']])
        tb.run_template('00-Literals-Good-Bad', vis='stack-bar', data=ret, title='Frequency of good, okay and bad literal statements')

    def calculate_per_dataset_longtail(self, analysis, val_col='?val', no_col='?no', relativize='?nr_stmt_total'):
        per_value = {}
        nr_value_per_dataset = {}
        for dataset in self.datasets:
            with open("%s/%s-count-%s.rq.tsv" % (self.analyses_dir, dataset, analysis), "r") as tsvin:
                nr_value_per_dataset[dataset] = {}
                reader = csv.DictReader(tsvin, delimiter='\t')
                for row in reader:
                    val = row[val_col]
                    if not val in nr_value_per_dataset[dataset]:
                        nr_value_per_dataset[dataset][val] = 1
                    else:
                        nr_value_per_dataset[dataset][val] += 1
                    no = float(row[no_col])
                    if not val in per_value:
                        per_value[val] = {}
                        for this_dataset in self.datasets:
                            per_value[val][this_dataset] = 0
                    per_value[val][dataset] += no
        header = ['predicate']
        for dataset in self.datasets:
            header.append(dataset)
        ret = []
        ret.append(header)
        # print nr_value_per_dataset
        # this determines the order of the values
        # with open("%s/%s-count-%s-longtail.rq.tsv" % (self.analyses_dir, 'onbcodices', analysis), "r") as tsvin:
        # with open("%s/%s-count-%s.rq.tsv" % (self.analyses_dir, 'uibwab', analysis), "r") as tsvin:
        for val in per_value:
            # reader = csv.DictReader(tsvin, delimiter='\t')
            # already_out = {}
            # for row in reader:
            # val = row[val_col]
            # if val in already_out:
                # continue
            # else:
                # already_out[val] = True
            outrow = [self.shorten_url(val)]
            for dataset in self.datasets:
                print dataset
                out_no = per_value[val][dataset]
                if relativize == 'nr_vals':
                    if not val in nr_value_per_dataset[dataset]:
                        nr_value_per_dataset[dataset][val] = 1
                    out_rel = nr_value_per_dataset[dataset][val]
                    outrow.append(out_no / out_rel)
                elif relativize in self.per_dataset_stats[dataset]:
                    per_value[val][dataset] += 100 * (out_no / self.per_dataset_stats[dataset][relativize])
                else:
                    outrow.append(out_no)
            ret.append(outrow)
        print ret
        self.run_template("total_per_dataset_%s_by_%s_multilongtail" % (analysis, clean_string(val_col)), vis='column', data=ret[0:100], title='foo')

if __name__ == '__main__':
    tb = TableBuilder(
        vis_dir='out/',
        template_dir='tpl/',
        sparql_dir='sparql/',
        analyses_dir='analysis',
    )

    tb.count_per_dataset_stats()
    tb.write_per_dataset_stats()

    # tb.calculate_long_tail('dm2e-properties-adjusted', 'predicates-per-dataset')
    # tb.calculate_long_tail('dm2e-classes-adjusted', 'types')

    # tb.calculate_good_bad_literals()
    tb.calculate_per_dataset_longtail('predicates-per-dataset-longtail', val_col='?val', no_col='?no', relativize='?nr_stmt_total')
    tb.calculate_per_dataset_longtail('statements-per-resource-and-type', val_col='?type', no_col='?no', relativize='nr_vals')
    # tb.calculate_per_dataset_longtail('statements-per-resource-and-type', val_col='?type', no_col='?no', relativize=None)
    # print tb.datasetso

    # tb.calculate_average('predicate-object-equal-statements', '?predicate', '?no')
    # tb.calculate_average('statements-per-resource-and-type', '?dctype', '?no')
    # tb.calculate_average('statements-per-resource-and-type', '?type', '?no')

    # tb.visualize_global_sums()

    # tb.visualize_global_values(long_running=False)

    # tb.visualize_averages()

    # tb.visualize_per_dctype()
