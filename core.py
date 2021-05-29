import numpy as np
import weakref
from memory_profiler import profile
import contextlib

class Variable:
	def __init__(self, data):
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError(f'{type(data)}은(는) 지원하지 않습니다.')

		self.data = data
		self.grad = None
		self.creator = None
		self.generation = 0

	def set_creator(self, func):
		self.creator = func
		self.generation = func.generation + 1

	def backward(self, retain_grad=False):
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = []
		seen_set = set()

		def add_func(f):
			if f not in seen_set:
				funcs.append(f)
				seen_set.add(f)
				funcs.sort(key=lambda x: x.generation)

		add_func(self.creator)
		
		while funcs:
			f = funcs.pop()
			gys = [output().grad for output in f.outputs]
			gxs = f.backward(*gys)
			if not isinstance(gxs, tuple):
				gxs = (gxs,)

			for x, gx in zip(f.inputs, gxs):
				if x.grad == None:
					x.grad = gx
				else:
					x.grad = x.grad + gx

				if x.creator is not None:
					add_func(x.creator)

			if not retain_grad:
				for y in f.outputs:
					y().grad = None

	def cleargrad(self):
		self.grad = None

class Function:
	def __call__(self, *inputs):
		xs = [x.data for x in inputs]
		ys = self.forward(*xs)
		if not isinstance(ys, tuple):
			ys = (ys, )
		outputs = [Variable(as_array(y)) for y in ys]

		if Config.enable_backprop:
			self.generation = max([x.generation for x in inputs])
			for output in outputs:
				output.set_creator(self) # 출력 변수에 창조자를 설정한다.
			self.inputs = inputs 
			self.outputs = [weakref.ref(output) for output in outputs]
		
		return outputs if len(outputs) > 1 else outputs[0]
	
	def forward(self, xs):
		raise NotImplementedError()
		
	def backward(self, gys):
		raise NotImplementedError()

class Config:
	enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
	old_value = getattr(Config, name)
	setattr(Config, name, value)
	try:
		yield
	finally:
		setattr(Config, name, old_value)

def no_grad():
	return using_config('enable_backprop', False)

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
	x0 = Variable (np.array(1.0))
	x1 = Variable(np.array (1.0))
	
	with no_grad():
		t = add(x0, x1)
		y = add(x0, t)
		print(y.data)
	
	t = add(x0, x1)
	y = add(x0, t)
	y.backward()
	print (y.grad , t.grad)
	print (x0.grad , x1.grad)
