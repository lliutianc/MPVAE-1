
def retrieve_target_label_idx(args, target_label):
    from data import load_data

    if len(target_label) > 1 and isinstance(target_label, str) == False:
        raise NotImplementedError(
            'cannot handle multiple target labels yet...')

    np.random.seed(args.seed)
    _, _, labels, _, _ = load_data(
        args.dataset, args.mode, True, 'onehot')

    label_type, count = np.unique(labels, axis=0, return_counts=True)
    count_sort_idx = np.argsort(-count)
    label_type = label_type[count_sort_idx]
    for idx, lab in enumerate(label_type):
        lab_str = ''.join(lab.astype(int).astype(str))
        if lab_str == target_label:
            return idx

    return None