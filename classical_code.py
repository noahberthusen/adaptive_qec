import numpy as np
class ClassicalCode:
    """
    - n is the number of bits
    - m is the number of checks
    - bit_nbhd is a size n list of lists. bit_nbhd[no_bit] is the list
    of checks which involve the bit number no_bit.
    - check_nbhd is a size m list of lists. check_nbhd[no_check] is the list
    of bits which are involved in the check number no_check.
    """
    def __init__(self, n, m, dv, dc, bit_nbhd, check_nbhd):
        self.n = n
        self.m = m
        # self.bit_nbhd = [list(set(nbhd)) for nbhd in bit_nbhd]
        # self.check_nbhd = [list(set(nbhd)) for nbhd in check_nbhd]
        self.bit_nbhd = bit_nbhd # bit neighborhood
        self.check_nbhd = check_nbhd # check neighborhood
        self.dv = dv
        self.dc = dc


def read_code(f_name: str):
    """
    Reads in classical code f_name and returns a ClassicalCode object representing the code
    """

    with open(f_name, 'r') as f:
        n = int(f.readline().split(',')[1])
        m = int(f.readline().split(',')[1])
        dv = int(f.readline().split(',')[1])
        dc = int(f.readline().split(',')[1])
        f.readline()
        bit_nbhd = []
        for i in range(n):
            nbhd = [int(c) for c in f.readline().strip(',\n').split(',')]
            bit_nbhd.append(nbhd)

        f.readline()

        check_nbhd = []
        for i in range(m):
            nbhd = [int(c) for c in f.readline().strip(',\n').split(',')]
            check_nbhd.append(nbhd)

    return ClassicalCode(n, m, dv, dc, bit_nbhd, check_nbhd)

def write_code(f_name: str, code: ClassicalCode):
    """
    Creates a code txt file from a ClassicalCode object
    """

    with open(f_name, 'w') as f:
        f.write(f'n,{code.n}\n')
        f.write(f'm,{code.m}\n')
        f.write(f'dv,{code.dv}\n')
        f.write(f'dc,{code.dc}\n')
        f.write('bit_nbhd\n')
        for bit_nbhd in code.bit_nbhd:
            for check in bit_nbhd:
                f.write(str(check))
                f.write(',')
            f.write('\n')
        f.write(f'check_nbhd\n')
        for check_nbhd in code.check_nbhd:
            for bit in check_nbhd:
                f.write(str(bit))
                f.write(',')
            f.write('\n')

if __name__ == "__main__":
    ccode = read_code("./prebuilt_code/ssf_masked/ccode/16_12_3_4.code")
    H = np.zeros((ccode.m, ccode.n))
    for i in range(ccode.m):
        for j in range(ccode.n):
            if (j in ccode.check_nbhd[i]):
                H[i][j] = 1
    print(H)