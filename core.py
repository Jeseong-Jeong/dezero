import numpy as np

class Variable:
	def __init__(self, data):
		self.data = data
		self.grad = None
		self.creator = None

	def set_creator(self, func):
		self.creator = func

	def backward(self):
		f = self.creator
		if f is not None:
			x = f.input
			x.grad = f.backward(self.grad)
			x.backward() # reculsive

class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		output = Variable(y)
		output.set_creator(self) # 출력 변수에 창조자를 설정한다.
		self.input = input 
		self.output = output # 출력도 저장한다.
		return output
	
	def forward(self, x):
		raise NotImplementedError()
		
	def backward(self, gy):
		raise NotImplementedError()

class Square(Function):
	def forward(self, x):
		y = x ** 2
		return y
	
	def backward(self, gy):
		x = self.input.data
		gx = 2 * x * gy
		return gx

class Exp(Function):
	def forward(self, x):
		y = np.exp(x)
		return y
	
	def backward(self, gy):
		x = self.input.data
		gx = np.exp(x) * gy
		return gx

if __name__ == '__main__':
	A = Square()
	B = Exp()
	C = Square()

	x = Variable(np.array(0.5))
	a = A(x)
	b = B(a)
	y = C(b)

	assert y.creator == C
	assert y.creator.input == b
	assert y.creator.input.creator == B
	assert y.creator.input.creator.input == a
	assert y.creator.input.creator.input.creator == A
	assert y.creator.input.creator.input.creator.input == x

	# 역전파 도전!
	y.grad = np.array(1.0)

	C = y.creator
	b = C.input
	b.grad = C.backward(y.grad)

	B = b.creator
	a = B.input
	a.grad = B.backward(b.grad)

	A = a.creator
	x = A.input
	x.grad = A.backward(a.grad)

	print(x.grad)

	# 재귀로 자동화!
	y.grad = np.array(1.0)
	y.backward()
	print(x.grad)