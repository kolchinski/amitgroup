
import numpy as np

def zeropad(data, padwidth):
    data = np.asarray(data)
    shape = data.shape
    if isinstance(padwidth, int):
        padwidth = (padwidth,)*len(shape) 
        
    padded_shape = map(lambda ix: ix[1]+padwidth[ix[0]]*2, enumerate(shape))
    new_data = np.zeros(padded_shape, dtype=data.dtype)
    new_data[ [slice(w, -w) if w > 0 else slice(None) for w in padwidth] ] = data 
    return new_data

def zeropad_to_shape(data, shape):
    """Zero-pads an array to a certain shape"""
    new_data = np.zeros(shape)
    print [slice(shape[i] - data.shape[i]//2, shape[i] - data.shape[i]//2 + data.shape[i]) for i in xrange(len(shape))]
    new_data[ [slice(shape[i]//2 - data.shape[i]//2, shape[i]//2 - data.shape[i]//2 + data.shape[i]) for i in xrange(len(shape))] ] = data
    return new_data

def border_value_pad(data, padwidth):
    data = np.asarray(data)
    shape = data.shape
    if isinstance(padwidth, int):
        padwidth = (padwidth,)*len(shape) 
        
    padded_shape = map(lambda ix: ix[1]+padwidth[ix[0]]*2, enumerate(shape))
    new_data = np.empty(padded_shape, dtype=data.dtype)
    new_data[ [slice(w, -w) if w > 0 else slice(None) for w in padwidth] ] = data
    
    for i, pw in enumerate(padwidth):
        if pw > 0:
            selection = [slice(None)] * data.ndim
            selection2 = [slice(None)] * data.ndim
            
            # Lower boundary
            selection[i] = slice(0, pw)
            selection2[i] = slice(pw, pw+1)
            new_data[tuple(selection)] = new_data[tuple(selection2)]
            
            # Upper boundary
            selection[i] = slice(-pw, None)
            selection2[i] = slice(-pw-1, -pw)
            new_data[tuple(selection)] = new_data[tuple(selection2)]
            
     
    return new_data
    
