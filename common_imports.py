# system imports

# third party imports

# local imports

# define
tab_length = 4
tab = " "*tab_length


def old_print_wrapper(*args, **kwargs):
    """ wrapper for turning all old prints on/off"""

    # delayed default argument
    if 'suppress_print' not in kwargs:
        kwargs['suppress_print'] = True

    if not kwargs['suppress_print']:
        del kwargs['suppress_print']  # remove `suppress_print` flag
        print(*args, **kwargs)


""" These are the indices used to label the h and t's in the generated latex"""
summation_indices = 'ijklmnopqr'
unlinked_indices = 'zyxwvuts'


""" These are the indices used to label the h and z's in the generated latex"""
z_summation_indices = 'klmno'
z_unlinked_indices = 'yxwvuts'
