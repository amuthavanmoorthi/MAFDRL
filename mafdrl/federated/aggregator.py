import copy

def fedavg(list_of_state_dict_lists):
    """
    list_of_state_dict_lists: [ [sd_actor_0, sd_actor_1, ... sd_actor_{U-1}] from client 0,
                                [sd_actor_0, sd_actor_1, ...] from client 1,
                                ...
                              ]
    returns averaged_state_dict_list for all U actors.
    """
    U = len(list_of_state_dict_lists[0])
    num_clients = len(list_of_state_dict_lists)
    out = []
    for u in range(U):
        avg = copy.deepcopy(list_of_state_dict_lists[0][u])
        for k in avg.keys():
            for c in range(1, num_clients):
                avg[k] += list_of_state_dict_lists[c][u][k]
            avg[k] /= num_clients
        out.append(avg)
    return out
