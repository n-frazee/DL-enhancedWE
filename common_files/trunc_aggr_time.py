# trunc_aggr_time.py
# 
# Calculate aggregate simulation time from a WESTPA simulation
# and provide advice on which iteration to truncate to for equal aggr. simul. time.
# Automatic truncating option requires the newest WESTPA commits (>=2022.05).
#
# Modify the `if __name__ == "__main__"` block at the end to your liking.
# Uncomment the last few lines to `w_truncate` automatically.
# To run the whole thing, execute `python trunc_aggr_time.py`.
#
# By Jeremy Leung
#
# Last Modified: July 10, 2023


import h5py
import numpy


def calc_aggr_sim_time(h5file='west.h5', tau=10, units='ps', intolist=False):
    '''Calculates aggregate simulation time. The `intolist` option outputs the an array of cummulative sum of counts.'''
    count = 0
    with h5py.File(h5file,'r') as f:
        if intolist:
            accul_list = numpy.zeros(len(f['summary']))
        for idx in range(len(f['summary'][:])):
            if f['summary'][idx,'walltime'] > 0:
                count += f['summary']['n_particles'][idx]
                if intolist:
                    accul_list[idx] = count

    print(f'{h5file} Aggregate: {count} segments; Total of {count*tau} {units} at {tau} {units}/seg.')

    if intolist:
        return numpy.trim_zeros(accul_list)
    else:
        return count


def calc_clock_time(h5file='west.h5'):
    '''Calculates the clock time (in terms of CPU and Wallclock time) from a west.h5 file.'''
    cpu_time = 0
    wall_time = 0
    with h5py.File(h5file,'r') as f:
        for idx in range(len(f['summary'][:])):
            wall_time += f['summary']['walltime'][idx]
            cpu_time += f['summary']['cputime'][idx]

    print(f'Wall Clock: {wall_time:.3f} secs / {wall_time/60:.3f} mins / {wall_time/60/60:.3f} hours\n\
CPU Clock: {cpu_time:.3f} secs / {cpu_time/60:.3f} mins / {cpu_time/60/60:.3f} hours')
   
    return cpu_time, wall_time


def equalize_aggr_sim_time(h5files, nsegs=None, tau=10, units='ps'):
    '''Given a list of h5file paths (h5files), run calc_aggr_sim_time() and advice what iteration to truncate to so all
       simulations have the same aggregate simulation time.'''
    list_of_counts = []
    for h5file in h5files:
        list_of_counts.append(calc_aggr_sim_time(h5file, tau=tau, units=units, intolist=True))
    
    array_of_counts = numpy.asarray(list_of_counts, dtype=object)

    if nsegs:
        assert numpy.all([j[-1] for j in array_of_counts] >= nsegs), f'Given nseg={nseg} is longer than the shortest simulation length.'

    last_frame_count = [i[-1] for i in array_of_counts]
    min_length = nsegs or min(last_frame_count)

    # min_length_where is of the form (idx in h5files, iteration index)
    if nsegs:
        min_length_where = (0, numpy.where(array_of_counts[0] >= nsegs)[0][0])
    else:
        argmin = numpy.argmin(last_frame_count)
        min_length_where = (argmin, len(array_of_counts[argmin]))

    good_list = []

    for idx, h5file in enumerate(h5files):
        if idx == min_length_where[0]:
            good_iter = min_length_where[1]
        else:
            good_iter = numpy.where(array_of_counts[idx] >= min_length)[0][0]
        
        print(f'{h5file}: `w_truncate` with N={good_iter+1}')
        good_list.append(good_iter)
    
    return good_list


def gen_patterns(pattern, nrange):
    '''Function to easily generate a list of filenames based on patterns. `pattern` should be a format string and `nrange` a list
       of values to format into `pattern`.'''
    return_list = []
    for val in nrange:
        return_list.append(pattern.format(val))

    return return_list


def strip_string(string):
    '''Convenient function of stripping the `.h5` from a string.'''
    return string.rsplit('.h5', maxsplit=1)[0] 


def run_w_trace_automatically(h5file, n_iter):
    '''Function to automatically run `w_trace`. Requires WESTPA >= v2022.05'''
    import argparse
    from unittest import mock
    from os.path import exists
    from shutil import copyfile

    try:
        from westpa.cli.core.w_truncate import entry_point
    except ImportError:
        raise ImportError('Unable to import WESTPA. Update/Install WESTPA >= v2022.05 or run `w_truncate` yourself.')

    new_h5file = strip_string(h5file) + f'_{n_iter-1}.h5'
    if not exists(new_h5file):
        copyfile(h5file, new_h5file)
    try:
        with mock.patch(
            target='argparse.ArgumentParser.parse_args',
            return_value=argparse.Namespace(rcfile='west.cfg', verbosity=None, n_iter=n_iter, we_h5filename=new_h5file)
        ):
            entry_point()
    except (OSError, ImportError):
        raise ImportError('Unable to complete. Update your WESTPA (to >= v2022.05) or run the `w_truncate` yourself.')


if __name__ == "__main__":
    # If you need to run any of these on your own.
    calc_aggr_sim_time('west.h5', tau=10)
    calc_clock_time('west.h5')

    # This generates a list ['diala_history_80_{5,10,20...}_multi/multi.h5']
    #input_list = gen_patterns('diala_history_80_{}_multi/multi.h5', list(range(5,26,5)))
    #input_list.extend(['diala_custom_fix_heavy_multi/multi.h5', 'diala_plain_80_multi/multi.h5'])
    #truncate_list = equalize_aggr_sim_time(input_list, nsegs=None, tau=10, units='ps')

    # Automatically w_truncating the files. This will automatically duplicate your h5 files.
    #print('Automatically w_truncating the files.')
    #for (h5file, n_iter) in zip(input_list, truncate_list):
    #    print(f'Truncating {h5file} with N={n_iter+1}...')
    #    run_w_trace_automatically(h5file, n_iter+1)

