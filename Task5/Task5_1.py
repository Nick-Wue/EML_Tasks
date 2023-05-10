import math
def forward_f(i_x, i_y, i_z):
    l_a = i_y + i_z
    l_b = l_a * i_x
    
    return l_b

def backward_f(i_x, i_y, i_z):
    l_a = i_y + i_z 

    l_dbda = i_x
    l_dbdx = l_a

    l_dady = 1
    l_dadz = 1

    l_dbdy = l_dbda * l_dady
    l_dbdz = l_dbda * l_dadz

    return [l_dbdx, l_dbdy, l_dbdz]

def forward_g(i_w0, i_w1, i_w2, i_x0, i_x1):
    l_a = i_w0 * i_x0
    l_b = i_w1 * i_x1
    l_c = l_b + i_w2
    l_d = l_a + l_c
    l_e = - math.exp(l_d)
    l_f = 1 + l_e
    l_g = 1 / l_f

    return l_g

def backward_g(i_w0, i_w1, i_w2, i_x0, i_x1):

    l_a = i_w0 * i_x0
    l_b = i_w1 * i_x1
    l_c = l_b + i_w2
    l_d = l_a + l_c
    l_e = - math.exp(l_d)
    l_f = 1 + l_e
    l_g = 1 / l_f


    l_dgdf = - 1 / l_f ** 2    
    l_dfde = 1
    l_dedd = -math.exp(l_d)
    l_ddda = 1
    l_dddc = 1
    l_dcdb = 1
    l_dcdw2 = 1
    l_dbdw1 = i_x1
    l_dbdx1 = i_w1
    l_dadw0 = i_x0
    l_dadx0 = i_w0

    l_dgdw0 = l_dgdf * l_dfde * l_dedd * l_ddda * l_dadw0
    l_dgdw1 = l_dgdf * l_dfde * l_dedd * l_dddc * l_dcdb * l_dbdw1
    l_dgdw2 = l_dgdf * l_dfde * l_dedd * l_dddc * l_dcdw2
    l_dgdx0 = l_dgdf * l_dfde * l_dedd * l_ddda * l_dadx0
    l_dgdx1 = l_dgdf * l_dfde * l_dedd * l_dddc * l_dcdb * l_dbdx1

    return l_dgdw0, l_dgdw1, l_dgdw2, l_dgdx0, l_dgdx1

def forward_h (i_x, i_y):
    l_a = i_x * i_y
    l_b = math.sin(l_a)
    l_c = i_x + i_y
    l_d = math.cos(l_c)
    l_e = l_b + l_d
    l_f = i_x - i_y
    l_g = math.exp(l_f)
    l_h = l_e / l_g

    return l_h


def backward_h (i_x, i_y):
    l_a = i_x * i_y
    l_b = math.sin(l_a)
    l_c = i_x + i_y
    l_d = math.cos(l_c)
    l_e = l_b + l_d
    l_f = i_x - i_y
    l_g = math.exp(l_f)
    l_h = l_e / l_g


    l_dhde = 1 / l_g
    l_dhdg = - l_e / (l_g ** 2)
    l_dgdf = math.exp(l_f)
    l_dfdx = 1
    l_dfdy = -1
    l_dedb = 1
    l_dedd = 1
    l_dddc = - math.sin(l_c)
    l_dcdx = 1
    l_dcdy = 1
    l_dbda = math.cos(l_a)
    l_dadx = i_y
    l_dady = i_x

    l_dhdx = (l_dhde * l_dedb * l_dbda * l_dadx) + (l_dhde *l_dedd * l_dddc * l_dcdx) + (l_dhdg * l_dgdf * l_dfdx)
    l_dhdy = (l_dhde * l_dedb * l_dbda * l_dady) + (l_dhde * l_dedd * l_dddc * l_dcdy) + (l_dhdg* l_dgdf * l_dfdy)
    return l_dhdx, l_dhdy


