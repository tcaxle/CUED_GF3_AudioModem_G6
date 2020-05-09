import numpy,random,operator,math

# compute hamming distance of two bit sequences
def hamming(s1,s2):
    return sum(map(operator.xor,s1,s2))

# xor together all the bits in an integer
def xorbits(n):
    result = 0
    while n > 0:
        result ^= (n & 1)
        n >>= 1
    return result

def expected_parity(from_state,to_state,k,glist):
    # x[n] comes from to_state
    # x[n-1] ... x[n-k-1] comes from from_state
    x = ((to_state >> (k-2)) << (k-1)) + from_state
    return [xorbits(g & x) for g in glist]

def convolutional_encoder(bits,k,glist):
    result = []
    state = 0
    for b in bits:
        state = (b << (k-1)) + (state >> 1)
        for g in glist:
            result.append(xorbits(state & g))
    return numpy.array(result)


#MIT example of convolutional encoder
#http://web.mit.edu/6.02/www/s2009/handouts/labs/lab5.shtml