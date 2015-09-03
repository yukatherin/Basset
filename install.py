#!/usr/bin/env python
from optparse import OptionParser

################################################################################
# name
#
#
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    ############################################################
    # prepare data
    ############################################################
    os.chdir('data')

    # download and arrange available data
    cmd = './get_dnase.sh'
    subprocess.call(cmd, shell=True)

    # preprocess
    cmd = 'preprocess_features.py -y -m 100 -s 600 -o encode_roadmap -c /Users/davidkelley/research/common/data/genomes/hg19/assembly/human.hg19.genome sample_beds.txt'
    subprocess.call(cmd, shell=True)

    # make a FASTA file
    cmd = 'bedtools getfasta -fi $HG19/sequence/hg19.fa -bed encode_roadmap.bed -s -fo encode_roadmap.fa'
    subprocess.call(cmd, shell=True)

    # make an HDF5 file
    cmd = 'seq_hdf5.py -t 0.0197676 -v 0.0204215 encode_roadmap.fa encode_roadmap_acc.txt encode_roadmap'
    subprocess.call(cmd, shell=True)

    os.chdir('..')


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
