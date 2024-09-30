import os.path

column_names_for_simulation = "c,n,k,p_phys,p_meas,no_test,mean,variance"
first_line = column_names_for_simulation + '\n'

class Result:
    def __init__(self,c,n,k,p_phys,p_meas,no_test,mean,variance):
        if not (type(c) == int or type(n) == int or type(k) == int or type(no_test) == int or\
                type(p_phys) == float or type(p_meas) == float or type(mean) == float or type(variance) == float):
            raise NameError('Bad result format')
        self.c = c
        self.n = n
        self.k = k
        self.p_phys = p_phys
        self.p_meas = p_meas
        self.no_test = no_test
        self.mean = mean
        self.variance = variance

def res_to_line(r):
    line = str(r.c) + ',' + str(r.n) + ',' + str(r.k) + "," +\
           str(r.p_phys) + ',' + str(r.p_meas) + "," + str(r.no_test) + ',' + str(r.mean) + ',' + str(r.variance) + '\n'
    return line

def line_to_res(line):
    tmp = line.strip('\n').split(',')
    r = Result(int(tmp[0]),int(tmp[1]),int(tmp[2]),
               float(tmp[3]),float(tmp[4]),int(tmp[5]),float(tmp[6]),float(tmp[7]))
    return r

# Creates a file whose lines are stored in lines_list
def create_file(file_name, lines_list):
    lines_list.sort()
    file = open(file_name, 'w')
    file.write(first_line)
    for line in lines_list:
        file.write(line)
    file.close()


# r1 and r2 are objects of the class Result
# This function tries to combine the results r1 and r2.
def combine_res(r1,r2):
    if (r1.c == r2.c and r1.n == r2.n and r1.k == r2.k and\
        r1.p_phys == r2.p_phys and r1.p_meas == r2.p_meas):
        no_test = r1.no_test + r2.no_test
        new_mean = r1.mean + ((r2.mean - r1.mean) / no_test)
        new_variance = r1.variance + ((r2.mean - r1.mean) * (r2.mean - new_mean))
        return Result(r1.c,r1.n,r1.k,
                      r1.p_phys,r1.p_meas,no_test,new_mean,new_variance)
    return None


############# To store the results during simulations #############
# This function adds the result r in the list 'res_list'
# In place function
def add_new_res(res_list, r):
    done = False
    i = 0
    while not done and i < len(res_list):
        r2 = res_list[i]
        r_new = combine_res(r2,r)
        if r_new == None:
            r_new = r2
        else:
            done = True
        res_list[i] = r_new
        i = i+1
    if not done:
        res_list.append(r)


# This function adds the result r in the file 'file_name'
# We create a tmp file and then rename it. If we had modified directly the file 'file_name' then we lose the result when there is Ctrl-C or a timeout during this function.
def save_new_res(file_name, new_res_list):
    if not os.path.exists(file_name):
        create_file(file_name,[])
    tmp_file_name = file_name + ".tmp"
    file = open(file_name, 'r')
    if file.readline() != first_line:
        file.close()
        raise NameError('Bad file format')

    res_list = [line_to_res(line) for line in file]
    file.close()
    for r in new_res_list:
        add_new_res(res_list, r)
    new_lines_list = [res_to_line(r) for r in res_list]
    create_file(tmp_file_name,new_lines_list)
    os.replace(tmp_file_name,file_name)
