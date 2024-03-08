from .MDM_pocket_coor_shared import MDM_full_pocket_coor_shared


def get_model(config):
    if config.network == 'MDM_full_pocket_coor_shared':
        return MDM_full_pocket_coor_shared(config)

    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
