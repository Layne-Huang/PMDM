if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('exp_name', type=str)
    parser.add_argument('--dataset', type=str, default='crossdock')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()

    dataset_info = get_dataset_info(args.dataset, False)
    path = './logs/Final_crossdock_pocket_revisedCA_velEGNN_no_fix_2022_12_16__13_05_02/generalized_batch_final_0_25_result_2023_05_07__17_41_27/samples_all.pkl'
    # path = args.path
    print(os.path.dirname(path))
    save_mol_result_path = os.path.join(os.path.dirname(path), 'mol_results.pkl')
    if os.path.exists(save_mol_result_path):
        with open(save_mol_result_path, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    