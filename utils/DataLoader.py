from torch.utils.data import Dataset, DataLoader
from bert_score import score as bert_sim_score
import numpy as np
import random
import pandas as pd
import torch


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False,
                             num_workers=2)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, ref_src_node_ids: np.ndarray = None, ref_dst_node_ids: np.ndarray = None, ref_node_interact_times: np.ndarray = None, ref_edge_ids: np.ndarray = None, ref_on_src_side: np.ndarray = None):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)

        # Used for link recommendation task
        self.ref_src_node_ids = np.zeros_like(src_node_ids) if ref_src_node_ids is None else ref_src_node_ids
        self.ref_dst_node_ids = np.zeros_like(dst_node_ids) if ref_dst_node_ids is None else ref_dst_node_ids
        self.ref_node_interact_times = np.zeros_like(node_interact_times) if ref_node_interact_times is None else ref_node_interact_times
        self.ref_edge_ids = np.zeros_like(edge_ids) if ref_edge_ids is None else ref_edge_ids
        self.ref_on_src_side = np.zeros_like(labels) if ref_on_src_side is None else ref_on_src_side #Set to True if connected to target edge's src node

    def calc_merged(self, data: 'Data'):
        merged_src_node_ids = np.concatenate((self.src_node_ids, data.src_node_ids))
        merged_dst_node_ids = np.concatenate((self.dst_node_ids, data.dst_node_ids))
        merged_node_interact_times = np.concatenate((self.node_interact_times, data.node_interact_times))
        merged_edge_ids = np.concatenate((self.edge_ids, data.edge_ids))
        merged_labels = np.concatenate((self.labels, data.labels))
        merged = Data(merged_src_node_ids, merged_dst_node_ids, merged_node_interact_times, merged_edge_ids, merged_labels)
        merged.ref_src_node_ids = np.concatenate((self.ref_src_node_ids, data.ref_src_node_ids))
        merged.ref_dst_node_ids = np.concatenate((self.ref_dst_node_ids, data.ref_dst_node_ids))
        merged.ref_node_interact_times = np.concatenate((self.ref_node_interact_times, data.ref_node_interact_times))
        merged.ref_edge_ids = np.concatenate((self.ref_edge_ids, data.ref_edge_ids))
        merged.ref_on_src_side = np.concatenate((self.ref_on_src_side, data.ref_on_src_side))
        return merged

    # Computes up to n labels. Returns true if all labels computed
    def update_labels(self, n, edge_texts):
        neg_ones = self.labels == -1
        indices = np.where(neg_ones)[0]
        mask = np.zeros_like(self.labels, dtype=bool)
        if len(indices) == 0:
            return True

        ret = False
        n_positions = min(n, len(indices))
        if len(indices) <= n:
            ret = True
        mask[indices[:n_positions]] = True

        # Compute next n labels
        tgt_texts = edge_texts.set_index('i').loc[self.edge_ids[mask], 'text'].tolist()
        ref_texts = edge_texts.set_index('i').loc[self.ref_edge_ids[mask], 'text'].tolist()
        tgt_texts = [str(txt) for txt in tgt_texts]
        ref_texts = [str(txt) for txt in ref_texts]
        _, _, bertScore = bert_sim_score(tgt_texts, ref_texts, model_type="distilbert-base-uncased")
        self.labels[mask] = bertScore

        return ret

    # For link recommendation task, calculated out of place
    def calc_ref_edges(self, ref_data: 'Data', num_ref_edges_per_side, edge_texts, take_most_recent = False):
        # Want reference/to pred edge to be in diff pools (one from training data before time t, one from after training data from time t)
        # For each target edge, we want to find candidate historical edges (src_n_h, dst_n_h, time_h) from ref_data that share at least one node with the target edge
        # Does not calculate labels

        new_arr_shape = (self.src_node_ids.shape[0], num_ref_edges_per_side * 2)
        new_src_node_ids, new_dst_node_ids, new_node_interact_times, new_edge_ids, new_labels = np.zeros(new_arr_shape), np.zeros(new_arr_shape), np.zeros(new_arr_shape), np.zeros(new_arr_shape), np.zeros(new_arr_shape)
        opt_mask = np.zeros(new_arr_shape, dtype = bool)
        new_ref_src_node_ids, new_ref_dst_node_ids, new_ref_node_interact_times, new_ref_edge_ids, new_ref_on_src_side = np.zeros(new_arr_shape), np.zeros(new_arr_shape), np.zeros(new_arr_shape), np.zeros(new_arr_shape), np.zeros(new_arr_shape)

        batch_size = 8
        for batch_start in range(0, self.src_node_ids.shape[0], batch_size):
            if batch_start % 10000 == 0:
                print(batch_start)
            batch_end = min(batch_start + batch_size, self.src_node_ids.shape[0])
            batch_src_ids = self.src_node_ids[batch_start:batch_end]
            batch_dst_ids = self.dst_node_ids[batch_start:batch_end]

            # Figure out valid reference edges: batch this as it is the expensive part
            batch_src_ids_expanded = batch_src_ids[:, np.newaxis]  # Shape: (batch_size, 1)
            batch_dst_ids_expanded = batch_dst_ids[:, np.newaxis]
            same_src_mask = ((ref_data.src_node_ids == batch_src_ids_expanded) |
                            (ref_data.dst_node_ids == batch_src_ids_expanded))  # Shape: (batch_size, num_ref_edges), latter part after the or is needed for non-bipartite graphs
            same_dest_mask = ((ref_data.src_node_ids == batch_dst_ids_expanded) |
                            (ref_data.dst_node_ids == batch_dst_ids_expanded))

            for i in range(batch_end - batch_start):
                num_used_refs = 0
                ref_mask = np.zeros_like(same_src_mask[0], dtype=bool)
                ref_info = None
                for mask, side in [(same_src_mask[i],'src'), (same_dest_mask[i],'dest')]:
                    num_available_ref_edges = np.sum(mask)

                    if num_available_ref_edges == 0:
                        continue

                    # Select reference edges for this sample
                    ref_indices = np.where(mask)[0]
                    selected_refs = np.random.choice(
                        ref_indices,
                        min(num_ref_edges_per_side, num_available_ref_edges),
                        replace = False
                    )
                    if(take_most_recent):
                        selected_refs = np.sort(ref_indices)[-min(num_ref_edges_per_side, num_available_ref_edges):]
                    ref_mask[selected_refs] = True
                    if(side == 'src'):
                        ref_info = np.repeat(True, selected_refs.shape[0])
                    else:
                        num_used_refs = np.sum(ref_mask)
                        if ref_info is None:
                            ref_info = np.repeat(True, selected_refs.shape[0])
                        else:
                            ref_info = np.concatenate((ref_info, np.repeat(False, num_used_refs - ref_info.shape[0])))

                if num_used_refs == 0:
                    continue

                new_src_node_ids[batch_start + i, :num_used_refs] = np.repeat(self.src_node_ids[batch_start + i], num_used_refs)
                new_dst_node_ids[batch_start + i, :num_used_refs] = np.repeat(self.dst_node_ids[batch_start + i], num_used_refs)
                new_node_interact_times[batch_start + i, :num_used_refs] = np.repeat(self.node_interact_times[batch_start + i], num_used_refs)
                new_edge_ids[batch_start + i, :num_used_refs] = np.repeat(self.edge_ids[batch_start + i], num_used_refs)
                new_labels[batch_start + i, :num_used_refs] = np.repeat(self.labels[batch_start + i], num_used_refs)
                new_ref_src_node_ids[batch_start + i, :num_used_refs] = ref_data.ref_src_node_ids[ref_mask]
                new_ref_dst_node_ids[batch_start + i, :num_used_refs] = ref_data.dst_node_ids[ref_mask]
                new_ref_node_interact_times[batch_start + i, :num_used_refs] = ref_data.node_interact_times[ref_mask]
                new_ref_edge_ids[batch_start + i, :num_used_refs] = ref_data.edge_ids[ref_mask]
                new_ref_on_src_side[batch_start + i, :num_used_refs] = ref_info
                opt_mask[batch_start + i, :num_used_refs] = np.repeat(True, num_used_refs)

        # Flatten and apply mask
        opt_mask = opt_mask.flatten()
        new_src_node_ids = new_src_node_ids.flatten()[opt_mask]
        new_dst_node_ids = new_dst_node_ids.flatten()[opt_mask]
        new_node_interact_times = new_node_interact_times.flatten()[opt_mask]
        new_edge_ids = new_edge_ids.flatten()[opt_mask]
        new_labels = new_labels.flatten()[opt_mask]
        new_ref_src_node_ids = new_ref_src_node_ids.flatten()[opt_mask]
        new_ref_dst_node_ids = new_ref_dst_node_ids.flatten()[opt_mask]
        new_ref_node_interact_times = new_ref_node_interact_times.flatten()[opt_mask]
        new_ref_edge_ids = new_ref_edge_ids.flatten()[opt_mask]
        new_ref_on_src_side = new_ref_on_src_side.flatten()[opt_mask]

        return Data(new_src_node_ids, new_dst_node_ids, new_node_interact_times, new_edge_ids, new_labels, new_ref_src_node_ids, new_ref_dst_node_ids, new_ref_node_interact_times, new_ref_edge_ids, new_ref_on_src_side)

    def apply_mask(self, mask: np.ndarray):
        self.src_node_ids = self.src_node_ids[mask]
        self.dst_node_ids = self.dst_node_ids[mask]
        self.node_interact_times = self.node_interact_times[mask]
        self.edge_ids = self.edge_ids[mask]
        self.labels = self.labels[mask]

        self.num_interactions = len(self.src_node_ids)
        self.unique_node_ids = set(self.src_node_ids) | set(self.dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)

        self.ref_src_node_ids = self.ref_src_node_ids[mask]
        self.ref_dst_node_ids = self.ref_dst_node_ids[mask]
        self.ref_node_interact_times = self.ref_node_interact_times[mask]
        self.ref_edge_ids = self.ref_edge_ids[mask]

        self.ref_on_src_side = self.ref_on_src_side[mask]

    # in place
    # factor should be a value between 0 and 1
    def reduce_ref_count(self, factor):

        """
        # Previous: approximate taking one sample per side
        # Mask for ref edges connected on source side
        src_mask = np.zeros(len(self.edge_ids), dtype=bool)
        _, first_idx = np.unique(self.edge_ids, return_index=True)
        src_mask[first_idx] = True

        # Mask for ref edges connected on dest side
        dest_mask = np.zeros(len(self.edge_ids), dtype=bool)
        _, idx = np.unique(self.edge_ids[::-1], return_index=True)
        last_idx = len(self.edge_ids) - 1 - idx
        dest_mask[last_idx] = True

        mask = np.logical_or(src_mask, dest_mask)
        """

        n_true = int(self.src_node_ids.size * factor)

        mask = np.zeros_like(self.src_node_ids, dtype=bool)

        true_indices = np.random.choice(self.src_node_ids.size, size=n_true, replace=False)
        mask[true_indices] = True

        self.apply_mask(mask)

    def correct_typing(self):
        self.src_node_ids = self.src_node_ids.astype('int')
        self.dst_node_ids = self.dst_node_ids.astype('int')
        self.node_interact_times = self.node_interact_times.astype('int')
        self.edge_ids = self.edge_ids.astype('int')
        self.ref_src_node_ids = self.ref_src_node_ids.astype('int')
        self.ref_dst_node_ids = self.ref_dst_node_ids.astype('int')
        self.ref_node_interact_times = self.ref_node_interact_times.astype('int')
        self.ref_edge_ids = self.ref_edge_ids.astype('int')

        self.ref_on_src_side = self.ref_on_src_side.astype('bool')
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def normalize_labels(self, min, max):
        #self.labels = ((self.labels - mean) / std)
        self.labels = (self.labels - min) / (float)(max - min)

    def unnormalize_labels(self,  min, max):
        #self.labels = self.labels * std + mean
        self.labels = (self.labels * (float)(max - min)) + min

    def copy(self):
        return Data(self.src_node_ids, self.dst_node_ids, self.node_interact_times, self.edge_ids, self.labels, self.ref_src_node_ids, self.ref_dst_node_ids, self.ref_node_interact_times, self.ref_edge_ids, self.ref_on_src_side)

    def remove_info(self, data_to_remove):
        # Removes invalid data, used to properly set up valid train data neighbors
        mask = ~np.isin(self.edge_ids, data_to_remove.edge_ids)
        self.apply_mask(mask)


def get_link_recomendation_data(dataset_name: str, val_ratio: float, test_ratio: float, edge_texts: np.ndarray, args):

    random.seed(0)

    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, _ = get_link_prediction_data(dataset_name, val_ratio, test_ratio, args)

    # Used to set up node samplers
    full_data_edges = Data(full_data.src_node_ids, full_data.dst_node_ids, full_data.node_interact_times, full_data.edge_ids, full_data.labels)
    train_data_edges = Data(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times, train_data.edge_ids, train_data.labels)

    # Split training data into reference and prediction
    train_ref_cutoff_time = np.percentile(train_data.node_interact_times, 70, axis=0)
    train_ref_mask = train_data.node_interact_times < train_ref_cutoff_time
    train_data_ref = Data(train_data.src_node_ids[train_ref_mask], train_data.dst_node_ids[train_ref_mask], train_data.node_interact_times[train_ref_mask], train_data.edge_ids[train_ref_mask], train_data.labels[train_ref_mask])
    train_pred_mask = train_data.node_interact_times >= train_ref_cutoff_time
    train_data_pred = Data(train_data.src_node_ids[train_pred_mask], train_data.dst_node_ids[train_pred_mask], train_data.node_interact_times[train_pred_mask], train_data.edge_ids[train_pred_mask], train_data.labels[train_pred_mask])

    # Calculate ref data for train/test/val splits
    val_data_ref = train_data
    test_data_ref = train_data.calc_merged(val_data)

    train_data_pred = train_data_pred.calc_ref_edges(train_data_ref, 3, edge_texts)
    val_data = val_data.calc_ref_edges(val_data_ref, 1, edge_texts)
    new_node_val_data = new_node_val_data.calc_ref_edges(val_data_ref, 1, edge_texts)
    test_data = test_data.calc_ref_edges(test_data_ref, 1, edge_texts)
    new_node_test_data = new_node_test_data.calc_ref_edges(test_data_ref, 1, edge_texts)

    # Recompile full data
    full_data = train_data_pred.calc_merged(val_data).calc_merged(new_node_val_data)
    full_data = full_data.calc_merged(test_data).calc_merged(new_node_test_data)

    return node_raw_features, edge_raw_features, full_data, train_data_pred, val_data, test_data, new_node_val_data, new_node_test_data, train_data_edges, full_data_edges


def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float, args):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('../DyLink_Datasets/{}/edge_list.csv'.format(dataset_name, dataset_name))
    node_num = max(graph_df['u'].max(), graph_df['i'].max()) + 1
    rel_num = graph_df['r'].max() + 1
    if graph_df['label'].min() != 0:
        graph_df.label -= 1
    if 'GDELT' in dataset_name:
        graph_df.ts = graph_df.ts//15
    cat_num = graph_df['label'].max() + 1

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 768
    if args.use_feature == 'Bert':
        print('get pretrained features')
        edge_raw_features = np.load('../DyLink_Datasets/{}/r_feat.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('../DyLink_Datasets/{}/e_feat.npy'.format(dataset_name, dataset_name))
        assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
        assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
        # padding the features of edges and nodes to the same dimension (172 for all the datasets)
        if node_raw_features.shape[1] < NODE_FEAT_DIM:
            node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
            node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
        if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
            edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
            edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

        assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'
    else:
        if 'JODIE' in args.model_name:
            edge_raw_features = np.zeros((rel_num, EDGE_FEAT_DIM))
            node_raw_features = np.zeros((node_num, EDGE_FEAT_DIM))
        elif 'DyRep' in args.model_name:
            edge_raw_features = np.random.randn(rel_num, EDGE_FEAT_DIM)
            node_raw_features = np.random.randn(node_num, EDGE_FEAT_DIM)
        elif 'GraphMixer' in args.model_name:
            #edge_raw_features = np.zeros((rel_num, EDGE_FEAT_DIM))
            #node_raw_features = np.zeros((node_num, EDGE_FEAT_DIM))
            edge_raw_features = np.random.randn(rel_num, EDGE_FEAT_DIM)
            node_raw_features = np.random.randn(node_num, EDGE_FEAT_DIM)
        else:
            edge_raw_features = np.random.randn(rel_num, EDGE_FEAT_DIM)
            node_raw_features = np.random.randn(node_num, EDGE_FEAT_DIM)
            #edge_raw_features = np.zeros((rel_num, EDGE_FEAT_DIM))
            #node_raw_features = np.zeros((node_num, EDGE_FEAT_DIM))

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.r.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(list(test_node_set), int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, cat_num


def get_edge_classification_data(dataset_name: str, val_ratio: float, test_ratio: float, args):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('../DyLink_Datasets/{}/edge_list.csv'.format(dataset_name, dataset_name))
    node_num = max(graph_df['u'].max(), graph_df['i'].max()) + 1
    rel_num = graph_df['r'].max()
    if graph_df['label'].min() != 0:
        graph_df.label -= 1
    if 'GDELT' in dataset_name:
        graph_df.ts = graph_df.ts//15
    cat_num = graph_df['label'].max() + 1
    # rel_num = cat_num

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 768
    if args.use_feature == 'Bert':
        print('get pretrained features')
        edge_raw_features = np.load('../DyLink_Datasets/{}/r_feat.npy'.format(dataset_name, dataset_name))
        node_raw_features = np.load('../DyLink_Datasets/{}/e_feat.npy'.format(dataset_name, dataset_name))
        if edge_raw_features.shape[0] >= rel_num:
            edge_raw_features = np.random.randn(rel_num, EDGE_FEAT_DIM)
        assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
        assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
        # padding the features of edges and nodes to the same dimension (172 for all the datasets)
        if node_raw_features.shape[1] < NODE_FEAT_DIM:
            node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
            node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
        if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
            edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
            edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

        assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'
    else:
        #edge_raw_features = np.zeros((rel_num, EDGE_FEAT_DIM))
        #node_raw_features = np.zeros((node_num, EDGE_FEAT_DIM))
        edge_raw_features = np.random.randn(rel_num, EDGE_FEAT_DIM)
        node_raw_features = np.random.randn(node_num, EDGE_FEAT_DIM)


    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.r.values.astype(np.longlong) # graph_df.label.values
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # compute nodes which appear at test time
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # sample nodes which we keep as new nodes (to test inductiveness), so then we have to remove all their edges from training
    new_test_node_set = set(random.sample(list(test_node_set), int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # new edges with new nodes in the val and test set (for inductive evaluation)
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, cat_num