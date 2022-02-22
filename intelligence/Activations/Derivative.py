

h = 0.00001

def derivitave(func, val):
	top = func(val + h) - func(val)
	bottom = h 
	return top / bottom

