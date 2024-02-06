import os
import sys
import glob


def pdbs_to_pdbqts(pdb_dir, pdbqt_dir, dataset):

    for root, dirs, files in os.walk(pdb_dir):
        for file in files:
            if file.endswith('.pdb'):
                name = os.path.splitext(file)[0]
                # name = os.path.splitext(os.path.basename(file))[0]
                outfile = os.path.join(pdbqt_dir, name + '.pdbqt')
                pdb_to_pdbqt(os.path.join(root,file), outfile, dataset)
                print('Wrote converted file to {}'.format(outfile))


def pdb_to_pdbqt(pdb_file, pdbqt_file, dataset):
    if dataset == 'crossdocked':
        os.system('prepare_receptor4.py -r {} -o {}'.format(pdb_file, pdbqt_file))
    elif dataset == 'bindingmoad':
        os.system('prepare_receptor4.py -r {} -o {} -A checkhydrogens -e'.format(pdb_file, pdbqt_file))
    else:
        raise NotImplementedError
    return pdbqt_file


if __name__ == '__main__':
    pdbs_to_pdbqts('./data/test_data_1k', './data/test_data_1k/test_pdbqt', 'crossdocked')