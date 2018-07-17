###### Generate the Mapping between the Tasks Sets and the Stimuli ######

# Libraries
import cython
import itertools
import numpy

# Function 

def Get_TaskSet_Stimulus_Mapping(state_num, action_num, codec_type='extended'):
    
## Returns the mapping between hidden state and state/action mapping
#
# The returned A_map is of size 'hidden states' x state_num, specifying for
# each hidden state/state combination the correct action
#
# Codec_type specifies the kind of mapping to be used. If not given, it
# defaults to 'extended'.
#
# Possible codec's: Full Not implemented
#
# - extended: no 2 states can be mapped to the same action. This is taken
#   into account by first picking a sequence id from the number of
#   available actions, and then picking the action that corresponds to this
#   sequence id. For state n, the number of possible actions is
#   action_num - n + 1.
#
# - full: full mapping from states to actions, allowing different states to
#   promote the same action. The encoding is based on first
#   (action-num)-ary little-endian encoding. That is, we first iterate over
#   all actions for state 1 before incrementing actions for state 2, and so
#   on. This results in the number of hidden states to be
#   action_num^state_num.

    assert (codec_type == 'extended')
    
    if codec_type == 'extended':
        # Number of hidden states
        assert (state_num <= action_num);
        K = numpy.prod(numpy.arange(action_num+1)[-state_num:]);

        # Mapping from state to action
        A_map = numpy.zeros([K, state_num], dtype=numpy.intc);
        i = 0;
        for indices in itertools.permutations(numpy.arange(action_num),state_num):
            A_map[i,:] = numpy.intc(indices)
            i += 1
    return A_map