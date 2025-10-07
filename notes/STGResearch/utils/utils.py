import pickle
from .adj_mx_norm import *


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(file_path: str, adj_type: str):
    try:
        _, _, adj_mx = load_pickle(file_path)
    except ValueError:
        adj_mx = load_pickle(file_path)

    if adj_type == 'scalap':
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == 'normlap':
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == 'symnadj':
        adj = [calculate_symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]
    elif adj_type == 'transition':
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == 'doubletransition':
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == 'identity':
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'original':
        adj = [adj_mx]
    else:
        raise ValueError('Undefined adjacency matrix type.')
    return adj, adj_mx