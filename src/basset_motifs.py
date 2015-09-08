#!/usr/bin/env python
from optparse import OptionParser
import copy, os, pdb, shutil, subprocess, time

import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
from sklearn import preprocessing

import dna_io

################################################################################
# basset_motifs.py
#
# Collect statistics and make plots to explore the first convolution layer
# of the given model using the given sequences.
################################################################################

weblogo_opts = '-X NO -Y NO --errorbars NO --fineprint "" -c classic'

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <model_file> <test_hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='model_hdf5_file', default=None, help='Pre-computed model output as HDF5.')
    parser.add_option('-o', dest='out_dir', default='.')
    parser.add_option('-m', dest='meme_db', default='~/software/meme_4.10.1/motif_databases/CIS-BP/Homo_sapiens.meme')
    parser.add_option('-s', dest='sample', default=1000, type='int', help='Sample sequences from the test set [Default:%default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide Basset model file and test data in HDF5 format.')
    else:
        model_file = args[0]
        test_hdf5_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # Torch predict
    #################################################################
    if options.model_hdf5_file is None:
        options.model_hdf5_file = '%s/model_out.h5' % options.out_dir
        torch_cmd = 'basset_motifs_predict.lua -sample %d %s %s %s' % (options.sample, model_file, test_hdf5_file, options.model_hdf5_file)
        subprocess.call(torch_cmd, shell=True)


    #################################################################
    # load data
    #################################################################
    # load sequences
    test_hdf5_in = h5py.File(test_hdf5_file, 'r')
    seq_vecs = np.array(test_hdf5_in['test_in'])[:options.sample,:]
    seq_targets = np.array(test_hdf5_in['test_out'])[:options.sample,:]
    try:
        target_names = list(test_hdf5_in['target_names'])
    except KeyError:
        # TEMP TEMP TEMP
        # target_names = [str(x) for x in range(125)]
        target_names = [line.split()[1] for line in open('../data/cell_activity.txt')]
    test_hdf5_in.close()

    # convert to letters
    seqs = dna_io.vecs2dna(seq_vecs)

    # load model output
    model_hdf5_in = h5py.File(options.model_hdf5_file, 'r')
    filter_weights = np.array(model_hdf5_in['weights'])
    filter_outs = np.array(model_hdf5_in['outs'])
    model_hdf5_in.close()

    # store useful variables
    num_filters = filter_weights.shape[0]
    filter_size = filter_weights.shape[2]
    num_seqs = filter_outs.shape[0]
    num_targets = len(target_names)

    filter_names = np.array(['f%d'%fi for fi in range(num_filters)])


    #################################################################
    # global filter plots
    #################################################################
    # plot filter-sequence heatmap
    plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf'%options.out_dir)

    # plot filter-segment heatmap
    plot_filter_seg_heat(filter_outs)
    plot_filter_seg_heat(filter_outs, whiten=False)

    # plot filter-target correlation heatmap
    plot_target_corr(filter_outs, seq_targets, filter_names, target_names, '%s/filter_target_cors.pdf'%options.out_dir)

    #################################################################
    # individual filter plots
    #################################################################
    stats_out = open('%s/filter_stats.txt'%options.out_dir, 'w')

    meme_out = open('%s/filters_meme.txt'%options.out_dir, 'w')
    meme_intro(meme_out, seqs)

    for f in range(num_filters):
        print 'Filter %d' % f

        # plot filter parameters as a heatmap
        plot_filter_heat(filter_weights[f,:,:], '%s/filter%d_heat.pdf' % (options.out_dir,f))

        # print filter parameters
        params_out = open('%s/filter%d.txt' % (options.out_dir,f), 'w')
    	for n in range(4):
    		print >> params_out, '  '.join(['%6.3f' % v for v in filter_weights[f,n,:]])
    	print >> params_out, ''
        params_out.close()

        # collapse to a consensus motif
        print >> stats_out, '%2d %s' % (f, filter_motif(filter_weights[f,:,:]))

        # plot density of filter output scores
        plot_score_density(np.ravel(filter_outs[:,f,:]), stats_out, '%s/filter%d_dens.pdf' % (options.out_dir,f))

        # plot weblogo of high scoring outputs
        plot_filter_logo(filter_outs[:,f,:], filter_size, seqs, '%s/filter%d_logo'%(options.out_dir,f), maxpct_t=0.5)

        # add to the meme motif file
        meme_add(meme_out, '%s/filter%d_logo.fa'%(options.out_dir,f), f)

    stats_out.close()
    meme_out.close()

    # run tomtom
    subprocess.call('tomtom -thresh 0.2 -oc %s/tomtom -png %s/filters_meme.txt %s' % (options.out_dir,options_out_dir,options.meme_db), shell=True)


################################################################################
# meme_add
#
# Print intro material for MEME motif format.
#
# Input
#  meme_out:   open file
#  fasta_file: for learning the background frequencies
################################################################################
def meme_add(meme_out, fasta_file, f):
    nts = {'A':0, 'C':1, 'G':2, 'T':3}
    pwm_counts = []
    nsites = 4 # pseudocounts
    for line in open(fasta_file):
        if line[0] != '>':
            seq = line.rstrip()
            nsites += 1
            if len(pwm_counts) == 0:
                # initialize with the length
                for i in range(len(seq)):
                    pwm_counts.append([1]*4)

            # count
            for i in range(len(seq)):
                pwm_counts[i][nts[seq[i]]] += 1

    if nsites > 4:
        # normalize
        pwm_freqs = []
        for i in range(len(pwm_counts)):
            pwm_freqs.append([pwm_counts[i][j]/float(nsites) for j in range(4)])

        print >> meme_out, 'MOTIF filter%d' % f
        print >> meme_out, 'letter-probability matrix: alength= 4 w= %d nsites= %d' % (len(pwm_freqs), nsites)

        for i in range(len(pwm_freqs)):
            print >> meme_out, '%.4f %.4f %.4f %.4f' % tuple(pwm_freqs[i])
        print >> meme_out, ''


################################################################################
# meme_intro
#
# Print intro material for MEME motif format.
#
# Input
#  meme_out:   open file
#  fasta_file: for learning the background frequencies
################################################################################
def meme_intro(meme_out, seqs):
    nts = {'A':0, 'C':1, 'G':2, 'T':3}

    # count
    nt_counts = [1]*4
    for i in range(len(seqs)):
        for nt in seqs[i]:
            nt_counts[nts[nt]] += 1

    # normalize
    nt_sum = float(sum(nt_counts))
    nt_freqs = [nt_counts[i]/nt_sum for i in range(4)]

    print >> meme_out, 'MEME version 4'
    print >> meme_out, ''
    print >> meme_out, 'ALPHABET= ACGT'
    print >> meme_out, ''
    print >> meme_out, 'Background letter frequencies:'
    print >> meme_out, 'A %.4f C %.4f G %.4f T %.4f' % tuple(nt_freqs)
    print >> meme_out, ''


################################################################################
# plot_target_corr
#
# Plot a clustered heatmap of correlations between filter activations and
# targets.
#
# Input
#  filter_outs:
#  filter_names:
#  target_names:
#  out_pdf:
################################################################################
def plot_target_corr(filter_outs, seq_targets, filter_names, target_names, out_pdf):
    num_seqs = filter_outs.shape[0]
    num_targets = len(target_names)

    filter_outs_mean = filter_outs.mean(axis=2)

    # std is sequence by filter.
    filter_means_std = filter_outs_mean.std(axis=0)
    filter_outs_mean = filter_outs_mean[:,filter_means_std > 0]
    filter_names_live = filter_names[filter_means_std > 0]

    filter_target_cors = np.zeros((len(filter_names_live),num_targets))
    for fi in range(len(filter_names_live)):
        for ti in range(num_targets):
            cor, p = spearmanr(filter_outs_mean[:,fi], seq_targets[:num_seqs,ti])
            filter_target_cors[fi,ti] = cor

    cor_df = pd.DataFrame(filter_target_cors, index=filter_names_live, columns=target_names)

    sns.set(font_scale=0.3)
    plt.figure()
    sns.clustermap(cor_df, cmap='BrBG', center=0, figsize=(8,10))
    plt.savefig(out_pdf)
    plt.close()


################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_seq_heat(filter_outs, out_pdf, whiten=True, drop_dead=True):
    # compute filter output means per sequence
    filter_seqs = filter_outs.mean(axis=2)

    # whiten
    if whiten:
        filter_seqs = preprocessing.scale(filter_seqs)

    # transpose
    filter_seqs = np.transpose(filter_seqs)

    if drop_dead:
        filter_stds = filter_seqs.std(axis=1)
        filter_seqs = filter_seqs[filter_stds > 0]

    # downsample sequences
    seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)

    hmin = np.percentile(filter_seqs[:,seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:,seqs_i], 99.9)

    sns.set(font_scale=0.3)

    plt.figure()
    sns.clustermap(filter_seqs[:,seqs_i], row_cluster=True, col_cluster=True, linewidths=0, xticklabels=False, vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    #out_png = out_pdf[:-2] + 'ng'
    #plt.savefig(out_png, dpi=300)
    plt.close()


################################################################################
# plot_filter_seq_heat
#
# Plot a clustered heatmap of filter activations in sequence segments.
#
# Mean doesn't work well for the smaller segments for some reason, but taking
# the max looks OK. Still, similar motifs don't cluster quite as well as you
# might expect.
#
# Input
#  filter_outs
################################################################################
def plot_filter_seg_heat(filter_outs, whiten=True, drop_dead=True):
    b = filter_outs.shape[0]
    f = filter_outs.shape[1]
    l = filter_outs.shape[2]

    s = 5
    while l/float(s) - (l/s) > 0:
        s += 1
    print '%d segments of length %d' % (s,l/s)

    # split into multiple segments
    filter_outs_seg = np.reshape(filter_outs, (b, f, s, l/s))

    # mean across the segments
    filter_outs_mean = filter_outs_seg.max(axis=3)

    # break each segment into a new instance
    filter_seqs = np.reshape(np.swapaxes(filter_outs_mean, 2, 1), (s*b, f))

    # whiten
    if whiten:
        filter_seqs = preprocessing.scale(filter_seqs)

    # transpose
    filter_seqs = np.transpose(filter_seqs)

    if drop_dead:
        filter_stds = filter_seqs.std(axis=1)
        filter_seqs = filter_seqs[filter_stds > 0]

    # downsample sequences
    seqs_i = np.random.randint(0, filter_seqs.shape[1], 500)

    hmin = np.percentile(filter_seqs[:,seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:,seqs_i], 99.9)

    sns.set(font_scale=0.3)
    if whiten:
        out_pdf = 'filter_segs.pdf'
    else:
        out_pdf = 'filter_segs_raw.pdf'

    plt.figure()
    sns.clustermap(filter_seqs[:,seqs_i], row_cluster=True, col_cluster=True, linewidths=0, xticklabels=False, vmin=hmin, vmax=hmax)
    plt.savefig(out_pdf)
    #out_png = out_pdf[:-2] + 'ng'
    #plt.savefig(out_png, dpi=300)
    plt.close()


################################################################################
# filter_motif
#
# Collapse the filter parameter matrix to a single DNA motif.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def filter_motif(param_matrix):
    nts = 'ACGT'

    motif_list = []
    for v in range(param_matrix.shape[1]):
        max_n = 0
        for n in range(1,4):
            if param_matrix[n,v] > param_matrix[max_n,v]:
                max_n = n

        if param_matrix[max_n,v] > 0:
            motif_list.append(nts[max_n])
        else:
            motif_list.append('N')

    return ''.join(motif_list)


################################################################################
# plot_filter_heat
#
# Plot a heatmap of the filter's parameters.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_heat(param_matrix, out_pdf):
    param_range = abs(param_matrix).max()

    plt.figure(figsize=(param_matrix.shape[1], 4))
    sns.heatmap(param_matrix, cmap='PRGn', linewidths=0.2, vmin=-param_range, vmax=param_range)
    ax = plt.gca()
    ax.set_xticklabels(range(1,param_matrix.shape[1]+1))
    ax.set_yticklabels('TGCA', rotation='horizontal', size=10)
    plt.savefig(out_pdf)
    plt.close()


################################################################################
# plot_filter_logo
#
# Plot a weblogo of the filter's occurrences
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_filter_logo(filter_outs, filter_size, seqs, out_prefix, raw_t=0, maxpct_t=None):
    if maxpct_t:
        all_outs = np.ravel(filter_outs)
        all_outs_mean = all_outs.mean()
        all_outs_norm = all_outs - all_outs_mean
        raw_t = maxpct_t * all_outs_norm.max() + all_outs_mean

    # print fasta file of positive outputs
    filter_fasta_out = open('%s.fa' % out_prefix, 'w')
    filter_count = 0
    for i in range(filter_outs.shape[0]):
        for j in range(filter_outs.shape[1]):
            if filter_outs[i,j] > raw_t:
                kmer = seqs[i][j:j+filter_size]
                print >> filter_fasta_out, '>%d_%d' % (i,j)
                print >> filter_fasta_out, kmer
                filter_count += 1
    filter_fasta_out.close()

    # make weblogo
    if filter_count > 0:
        weblogo_cmd = 'weblogo %s < %s.fa > %s.eps' % (weblogo_opts, out_prefix, out_prefix)
        subprocess.call(weblogo_cmd, shell=True)


################################################################################
# plot_score_density
#
# Plot the score density and print to the stats table.
#
# Input
#  param_matrix: np.array of the filter's parameter matrix
#  out_pdf:
################################################################################
def plot_score_density(f_scores, stats_out, out_pdf):
    sns.set(font_scale=1.3)
    plt.figure()
    sns.distplot(f_scores, kde=False)
    plt.xlabel('ReLU output')
    plt.savefig(out_pdf)
    plt.close()

    zero_pct = len([s for s in f_scores if s > 0]) / float(len(f_scores))

    print >> stats_out, '%.4f  %6.4f' % (zero_pct, np.mean(f_scores))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
