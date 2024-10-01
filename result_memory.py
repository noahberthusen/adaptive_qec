import os.path

column_names_for_simulation = "concat,adapt,n,k,p_phys,p_meas,no_test,no_success,p_log"
first_line = column_names_for_simulation + '\n'

class Result:
    def __init__(self,concat,adapt,n,k,p_phys,p_meas,no_test,no_success):
        if not (type(concat) == int or type(adapt) == int or type(n) == int or type(k) == int or type(no_test) == int or\
                type(p_phys) == float or type(p_meas) == float or type(no_success) == int):
            raise NameError('Bad result format')
        self.concat = concat
        self.adapt = adapt
        self.n = n
        self.k = k
        self.p_phys = p_phys
        self.p_meas = p_meas
        self.no_test = no_test
        self.no_success = no_success

def res_to_line(r):
    p_log = r.no_success / r.no_test
    line = str(r.concat) + ',' + str(r.adapt) + ',' + str(r.n) + ',' + str(r.k) + ',' +\
           str(r.p_phys) + ',' + str(r.p_meas) + ',' + str(r.no_test) + ',' + str(r.no_success) + ',' + str(p_log) + '\n'
    return line

def line_to_res(line):
    tmp = line.strip('\n').split(',')
    r = Result(int(tmp[0]),int(tmp[1]),int(tmp[2]),int(tmp[3]),
               float(tmp[4]),float(tmp[5]),int(tmp[6]),int(tmp[7]))
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
    if (r1.concat == r2.concat and r1.adapt == r2.adapt and r1.n == r2.n and r1.k == r2.k and\
        r1.p_phys == r2.p_phys and r1.p_meas == r2.p_meas):
        no_test = r1.no_test + r2.no_test
        no_success = r1.no_success + r2.no_success
        return Result(r1.concat,r1.adapt,r1.n,r1.k,
                      r1.p_phys,r1.p_meas,no_test,no_success)
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
