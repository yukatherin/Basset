#!/usr/bin/env python
from optparse import OptionParser
import os
import subprocess

################################################################################
# install.py
#
# Download and arrange pre-trained models and data.
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    os.chdir('data')

    ############################################################
    # download pre-trained model
    ############################################################
    os.chdir('models')

    cmd = 'wget https://www.dropbox.com/s/rguytuztemctkf8/pretrained_model.th.gz?dl=0'
    subprocess.call(cmd, shell=True)

    cmd = 'gunzip pretrained_model.th.gz'
    subprocess.call(cmd, shell=True)

    os.chdir('..')


    ############################################################
    # download and prepare public data
    ############################################################
    # download and arrange available data
    cmd = './get_dnase.sh'
    subprocess.call(cmd, shell=True)

    # preprocess
    cmd = 'preprocess_features.py -y -m 200 -s 600 -o encode_roadmap -c /Users/davidkelley/research/common/data/genomes/hg19/assembly/human.hg19.genome sample_beds.txt'
    subprocess.call(cmd, shell=True)

    # make a FASTA file
    cmd = 'bedtools getfasta -fi $HG19/sequence/hg19.fa -bed encode_roadmap.bed -s -fo encode_roadmap.fa'
    subprocess.call(cmd, shell=True)

    # make an HDF5 file
    cmd = 'seq_hdf5.py -c -r -t 71886 -v 70000 encode_roadmap.fa encode_roadmap_act.txt encode_roadmap.h5'
    subprocess.call(cmd, shell=True)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
