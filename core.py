import numpy as np

class Variable:
	def __init__(self, data):
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')

		self.data = data
		self.grad = None
		self.creator = None

	def set_creator(self, func):
		self.creator = func

	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = [self.creator]
		while funcs:
			f = funcs.pop()
			gys = [output.grad for output in f.outputs]
			gxs = f.backward(*gys)
			if not isinstance(gxs, tuple):
				gxs = (gxs,)

			for x, gx in zip(f.inputs, gxs):
				x.grad = gx

				if x.creator is not None:
					funcs.append(x.creator)

class Function:
	def __call__(self, *inputs):
		xs = [x.data for x in inputs]
		ys = self.forward(*xs)
		if not isinstance(ys, tuple):
			ys = (ys, )
		outputs = [Variable(as_array(y)) for y in ys]

		for output in outputs:
			output.set_creator(self) # 출력 변수에 창조자를 설정한다.
		self.inputs = inputs 
		self.outputs = outputs # 출력도 저장한다.
		return outputs if len(outputs) > 1 else outputs[0]
	
	def forward(self, xs):
		raise NotImplementedError()
		
	def backward(self, gys):
		raise NotImplementedError()

def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x

def numerical_diff(f, x, eps=1e-4):
	x0 = Variable(x.data - eps)
	x1 = Variable(x.data + eps)
	y0 = f(x0)
	y1 = f(x1)
	return (y1.data - y0.data) / (2 * eps)

class Square(Function):
	def forward(self, x):
		y = x ** 2
		return y
	
	def backward(self, gy):
		x = self.inputs[0].data
		gx = 2 * x * gy
		return gx

class Exp(Function):
	def forward(self, x):
		y = np.exp(x)
		return y
	
	def backward(self, gy):
		x = self.inputs[0].data
		gx = np.exp(x) * gy
		return gx

class Add(Function):
	def forward(self, x0, x1):
		y = x0 + x1
		return y

	def backward(self, gy):
		return gy, gy
				
class Multiply(Function):
	def forward(self, x0, x1):
		y = x0 * x1
		return y

	def backward(self, gy):
		x1 = self.inputs[0].data
		x2 = self.inputs[1].data
		return x2 * gy, x1 * gy

class Subtract(Function):
	def forward(self, x0, x1):
		y = x0 - x1
		return y

	def backward(self, gy):
		return gy, -gy

class Divide(Function):
	def forward(self, x0, x1):
		if x1 == 0:
			raise ValueError("Cannot divide by zero")
		y = x0 / x1
		return y

	def backward(self, gy):
		x0 = self.inputs[0].data
		x1 = self.inputs[0].data
		gx0 = gy / x1
		gx1 = - x0 / x1 ** 2 * gy
		return gx0, gx1

def square(x):
	return Square()(x)

def exp(x):
	return Exp()(x)

def add(x0, x1):
	return Add()(x0, x1)

def multiply(x0, x1):
	return Multiply()(x0, x1)

def subtract(x0, x1):
	return Subtract()(x0, x1)

def divide(x0, x1):
	return Divide()(x0, x1)

if __name__ == '__main__':
	x0 = Variable(np.array(2.0))
	x1 = Variable(np.array(3.0))
	x2 = Variable(np.array(4.0))
	y = subtract(multiply(square(x0), square(x1)), divide(exp(x0), x2))
	y.backward()
	print(y.data)
	print(x0.grad)
	print(x1.grad)
	print(x2.grad)

