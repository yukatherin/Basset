#!/usr/bin/env python
from optparse import OptionParser
import copy, os, pdb, subprocess, sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

import dna_io
from seq_logo import seq_logo

################################################################################
# basset_heat.py
#
################################################################################

################################################################################
# main
############################s####################################################
def main():
    usage = 'usage: %prog [options] <model_file> <input_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='input_activity_file', help='Optional activity table corresponding to an input FASTA file')
    parser.add_option('-c', dest='cells', default='83', help='Comma-separated list of cell indexes to plot (or -1 for all) [Default: %default]')
    parser.add_option('-l', dest='min_limit', default=0.1, type='float', help='Minimum heatmap limit [Default: %default]')
    parser.add_option('-n', dest='center_nt', default=200, type='int', help='Center nucleotides to mutate and plot in the heatmap [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='heat', help='Output directory [Default: %default]')
    parser.add_option('-s', dest='sample', default=100, type='int', help='Sample sequences from the input HDF5 test set [Default:%default]')
    parser.add_option('-t', dest='torch_out_hdf5', default=None, help='Pre-computed Torch model output as HDF5 [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide Basset model file and input sequences (as a FASTA file or test data in an HDF file')
    else:
        model_file = args[0]
        input_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # parse input file
    #################################################################
    try:
        # load sequences and headers
        seqs = dna_io.fasta2dict(input_file)
        seq_headers = seqs.keys()

        model_input_hdf5 = '%s/model_in.h5'%options.out_dir

        if options.input_activity_file:
            # one hot code
            seqs_1hot, targets = dna_io.load_data_1hot(input_file, options.input_activity_file, mean_norm=False, whiten=False, permute=False, sort=False)

            # read in target names
            target_names = open(options.input_activity_file).readline().strip().split('\t')

            # write as test data to a HDF5 file
            #  (save everything so the user can just input this next time)
            h5f = h5py.File(model_input_hdf5, 'w')
            h5f.create_dataset('target_names', data=target_names)
            h5f.create_dataset('test_in', data=seqs_1hot)
            h5f.create_dataset('test_out', data=targets)
            h5f.create_dataset('test_headers', data=seq_headers)
            h5f.close()

        else:
            # load sequences and headers
            seqs_1hot = dna_io.load_sequences(input_file, permute=False)

            # write as test data to a HDF5 file
            h5f = h5py.File(model_input_hdf5, 'w')
            h5f.create_dataset('test_in', data=seqs_1hot)
            h5f.close()

    except (IOError, KeyError):
        try:
            model_input_hdf5 = input_file

            # load (sampled) test data from HDF5
            hdf5_in = h5py.File(input_file, 'r')
            seqs_1hot = np.array(hdf5_in['test_in'])[:options.sample,:]
            targets = np.array(hdf5_in['test_out'])[:options.sample,:]
            try: # TEMP
                seq_headers = list(hdf5_in['test_headers'])[:options.sample]
                target_names = list(hdf5_in['target_names'])
            except:
                seq_headers = None
                target_names = None
            hdf5_in.close()

            # convert to ACGT sequences
            seqs = dna_io.vecs2dna(seqs_1hot)

        except IOError:
            parser.error('Could not parse input file as FASTA or HDF5.')


    #################################################################
    # Torch predict modifications
    #################################################################
    if options.torch_out_hdf5 is None:
        options.torch_out_hdf5 = '%s/model_out.h5' % options.out_dir
        torch_cmd = 'basset_heat_predict.lua -sample %d -center_nt %d %s %s %s' % (options.sample, options.center_nt, model_file, model_input_hdf5, options.torch_out_hdf5)
        print torch_cmd
        subprocess.call(torch_cmd, shell=True)


    #################################################################
    # load modification predictions
    #################################################################
    hdf5_in = h5py.File(options.torch_out_hdf5, 'r')
    seq_mod_preds = np.array(hdf5_in['seq_mod_preds'])
    hdf5_in.close()

    # trim seqs to match seq_mod_preds length
    seq_len = len(seqs[seq_headers[0]])
    delta_start = 0
    delta_len = seq_mod_preds.shape[2]
    if delta_len < seq_len:
        delta_start = (seq_len - delta_len)/2
        for header in seq_headers:
            seqs[header] = seqs[header][delta_start:delta_start+delta_len]

    # decide which cells to plot
    if options.cells == '-1':
        plot_cells = xrange(scores.shape[1])
    else:
        plot_cells = [int(ci) for ci in options.cells.split(',')]


    #################################################################
    # plot
    #################################################################
    table_out = open('%s/table.txt' % options.out_dir, 'w')

    rdbu = sns.color_palette("RdBu_r", 10)

    nts = 'ACGT'
    for si in range(seq_mod_preds.shape[0]):
        header = seq_headers[si]
        seq = seqs[header]

        # plot some descriptive heatmaps for each individual cell type
        for ci in plot_cells:
            seq_mod_preds_cell = seq_mod_preds[si,:,:,ci]
            real_pred_cell = get_real_pred(seq_mod_preds_cell, seq)

            # compute matrices
            norm_matrix = seq_mod_preds_cell - real_pred_cell
            min_scores = seq_mod_preds_cell.min(axis=0)
            max_scores = seq_mod_preds_cell.max(axis=0)
            minmax_matrix = np.vstack([min_scores - real_pred_cell, max_scores - real_pred_cell])

            # prepare figure
            sns.set(style='white', font_scale=0.5)
            sns.axes_style({'axes.linewidth':1})
            heat_cols = 400
            sad_start = 1
            sad_end = 322
            logo_start = 0
            logo_end = 323
            fig = plt.figure(figsize=(20,3))
            ax_logo = plt.subplot2grid((3,heat_cols), (0,logo_start), colspan=(logo_end-logo_start))
            ax_sad = plt.subplot2grid((3,heat_cols), (1,sad_start), colspan=(sad_end-sad_start))
            ax_heat = plt.subplot2grid((3,heat_cols), (2,0), colspan=heat_cols)

            # print a WebLogo of the sequence
            vlim = max(options.min_limit, abs(minmax_matrix).max())
            seq_heights = 0.25 + 1.75/vlim*(-minmax_matrix[0])
            logo_eps = '%s/%s_c%d_seq.eps' % (options.out_dir, header.replace(':','_'), ci)
            seq_logo(seq, seq_heights, logo_eps)

            # add to figure
            logo_png = '%s.png' % logo_eps[:-4]
            subprocess.call('convert -density 300 %s %s' % (logo_eps, logo_png), shell=True)
            logo = Image.open(logo_png)
            ax_logo.imshow(logo)
            ax_logo.set_axis_off()

            # plot loss and gain SAD scores
            ax_sad.plot(-minmax_matrix[0], c=rdbu[0], label='loss', linewidth=1)
            ax_sad.plot(minmax_matrix[1], c=rdbu[-1], label='gain', linewidth=1)
            ax_sad.legend()
            # ax_sad.grid(True, linestyle=':')
            for axis in ['top','bottom','left','right']:
                ax_sad.spines[axis].set_linewidth(0.5)

            # plot real-normalized scores
            vlim = max(options.min_limit, abs(norm_matrix).max())
            sns.heatmap(norm_matrix, linewidths=0, cmap='RdBu_r', vmin=-vlim, vmax=vlim, xticklabels=False, ax=ax_heat)
            ax_heat.yaxis.set_ticklabels('TGCA', rotation='horizontal') # , size=10)

            # save final figure
            plt.tight_layout()
            plt.savefig('%s/%s_c%d_heat.pdf' % (options.out_dir,header.replace(':','_'), ci), dpi=300)
            plt.close()


        #################################################################
        # print table of nt variability for each cell
        #################################################################
        for ci in range(seq_mod_preds.shape[3]):
            seq_mod_preds_cell = seq_mod_preds[si,:,:,ci]
            real_pred_cell = get_real_pred(seq_mod_preds_cell, seq)

            min_scores = seq_mod_preds_cell.min(axis=0)
            max_scores = seq_mod_preds_cell.max(axis=0)

            loss_matrix = real_pred_cell - seq_mod_preds_cell.min(axis=0)
            gain_matrix = seq_mod_preds_cell.max(axis=0) - real_pred_cell

            for pos in range(seq_mod_preds_cell.shape[1]):
                cols = [header, delta_start+pos, ci, loss_matrix[pos], gain_matrix[pos]]
                print >> table_out, '\t'.join([str(c) for c in cols])

    table_out.close()


def get_real_pred(seq_mod_preds, seq):
    si = 0
    while seq[si] == 'N':
        si += 1

    if seq[si] == 'A':
        real_pred = seq_mod_preds[0,si]
    elif seq[si] == 'C':
        real_pred = seq_mod_preds[1,si]
    elif seq[si] == 'G':
        real_pred = seq_mod_preds[2,si]
    else:
        real_pred = seq_mod_preds[3,si]

    return real_pred


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
