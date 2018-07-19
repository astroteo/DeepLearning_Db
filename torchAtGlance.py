from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################## torch basics
# about types:
a = torch.tensor([1,2,4])
print("a = torch.tensor([1,2,4]) ")
print(type(a))
print(a.shape)

b = torch.tensor([1,2,4])
print(type(b))
print(b.shape)

# product
c = a*b
print(type(c))
print(c)

# matrix (scalar product)
c_m = torch.empty(1,1).type(torch.LongTensor)
b = b.view(3,1)# torch_tensor.view ( dim1,dim2 ) -> assigns the correct dimension
a = a.view(1,3)
print(b.shape)
print(type(b))

torch.mm(a,b,out = c_m)

print(c_m.shape)
print(type(c_m))
print(c_m)

c_m_python_number = c_m.item() # this returns the tensor as a standard python numbe
                               #!!Only one element tensors can be converted to Python scalars !!
print(c_m_python_number)
print(type(c_m_python_number))



# create an empty matrix: random initialized weights
m_empty = torch.empty(5, 3)
print(m_empty)# decleared as tensor.

# create random matrix of size (n,m)
m_rand = torch.rand(6,6);
print(m_rand, type(m_rand))# decleared as tensor

# list -> torch.tensor:
t_ = torch.tensor([[1,2],[3,4]])
print(t_,type(t_))#decleared as tensor

# operations
#-> formulation with torch_element.add()
t_add = t_.add(t_)

y = torch.tensor([[1,2],[3,4]]) # when decleared as numerical,  pytorch assign it to type = LongTensor
x = torch.tensor([[1,2],[3,4]]).type(torch.LongTensor)

# tensors behaves naturally just like lists/np.array => toooop


#-> formulation with torch.add(torch_element_1, torch_element_2, out = pre_decleared_empty_torch_element)
result = torch.empty(2,2).type(torch.LongTensor)# when decleared as empty, rand, etc. ,pytorch assign it to type = FloatTensor.
torch.add(x, y, out = result)
print(result)

#-> matrix times vector {no numpy array as intermediate}
A = torch.tensor([[1,2,5],[1,24,6],[7,0,8]])
x = torch.tensor([[1],[2],[4]])# hard to reshape => use numpy instead !!
y = torch.empty(3,1).type(torch.LongTensor);
torch.mm(A,x,out = y)
print(y)

# Linear algebra.
# -> set requires_grad = True to  stick a gradient on such tensor.
W = torch.tensor([[12,3],[4,5]],requires_grad = True)
x_ = torch.mm(W,x[:2])
print(x_)

x_m = x_.type(torch.FloatTensor).mean();# mean and others works only with FloatTensor
x_m = torch.tensor(x_m, requires_grad = True)
print(x_m)

#AutoGrad

#-> tensors as functionals are directly defined as variables,
x_t = torch.tensor([1,2,4]).type(torch.FloatTensor)
print(x_t)
print(x_t.requires_grad)# by default the gradient flag is set to False.

x_t.requires_grad_(True)# setting it to True manually requires_grad_
print(x_t.requires_grad)

#-> make a differentiation on a derived variable
y_t =  x_t**3
print(y_t.requires_grad)# MAKE-SENSE: any variable deriving from ANY differentiable variable is differentiable too
print(y_t)

# call 'backward()'  ! allowed only on a scalar variable !
y_sum = y_t.sum()
y_sum_bkwd = y_sum.backward()# allowed only on scalars
print(y_sum_bkwd)



#-> no mean
print(y_sum.grad_fn)

#-> call .grad()
print(x_t.grad)# ~> dy_t/dx_t = [3*x_t(1)^2,3*x_t(1)^2,3*x_t(1)^2 ]

########################################### simple NN
#-> class estending the torch.nn.model
class Model(nn.Module):
    def __init__(self):
        torch.nn.model.__init__()
        self.conv1 = nn.Conv2d(1, 20,  5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self,x):
        x = F.relu( self.conv1(x) )
        return F.relu( self.conv2(x) )

#-> define a function initializing the weights:
def init_weights(m):
        print(m)
        if type(m) == nn.Linear:
            m.weight.data.fill_(1.0)
            print(m.weight)

net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)# .apply => wrapping the function of init_weights
