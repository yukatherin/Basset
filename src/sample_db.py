#!/usr/bin/env python
from optparse import OptionParser
import random

################################################################################
# sample_db.py
#
# Sample from an existing database, represented as a BED file and activity
# table.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sample_num> <db_bed> <db_act_file>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='out_prefix', default='peaks', help='Output file prefix [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
    	parser.error('Must provide # of sequences to sample, DB BED, and DB activity table.')
    else:
        sample_num = int(args[0])
        db_bed = args[1]
        db_act_file = args[2]

    # initialize reservoirs
    bed_lines = ['']*sample_num
    table_lines = ['']*sample_num

    # open input files
    bed_in = open(db_bed)
    table_in = open(db_act_file)

    # save table header
    table_header = table_in.readline()

    # fill
    i = 0
    while i < sample_num:
        bed_lines[i] = bed_in.readline()
        table_lines[i] = table_in.readline()
        i += 1

    # sample
    bl = bed_in.readline()
    tl = table_in.readline()
    while bl and tl:
        j = random.randint(0,i+1)
        if j < sample_num:
            bed_lines[j] = bl
            table_lines[j] = tl
        i += 1
        bl = bed_in.readline()
        tl = table_in.readline()

    # close input files
    bed_in.close()
    table_in.close()


    # open output files
    bed_out = open('%s.bed' % options.out_prefix, 'w')
    table_out = open('%s_act.txt' % options.out_prefix, 'w')

    # print table header
    print >> table_out, table_header,

    # print reservoir
    for o in range(sample_num):
        print >> bed_out, bed_lines[o],
        print >> table_out, table_lines[o],

    # close files
    bed_out.close()
    table_out.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
