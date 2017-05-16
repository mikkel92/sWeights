

def GenData(Dim='1D'):

	import numpy as np

	Npoints1 = 5000
	Npoints2 = 10000
	Npoints3 = 10000

	# Non default data generators
	def oscdist(x, A, omega, phi) :
	    if (abs(A) <= 1.0) : return 1.0 + A*np.cos(omega*x + phi)
	    else               : return 0.0

	def GetAngle2( A, omega, phi ) :
	    if (abs(A) > 1.0) :
	        print("ERROR: A not in defined range: A = %6.2f" % (A))
	        return -999
	    x = -1.0 + 2.0 * np.random.rand()
	    y = (1.0+abs(A)) * np.random.rand()
	    while (y > oscdist(x, A, omega, phi)) :
	        x = -1.0 + 2.0 * np.random.rand()
	        y = (1.0+abs(A)) * np.random.rand()
	    return x

	np.random.seed(1)
	if Dim == '1D':
		a = []
		for i in range(0,Npoints1):
			a.append(np.random.randn()*0.2+0.8)
		for i in range(0,Npoints2):
			a.append(np.random.exponential(0.5, ))

		b = []
		for i in range(0,Npoints1):
			b.append(np.random.randn()*0.4+0.0)
		for i in range(0,Npoints2):
			b.append(np.random.randn()*0.6-1.0)

		e = np.column_stack((a,b))

	if Dim == '2D':
		a = []
		for i in range(0,Npoints1):
			a.append(np.random.randn()*0.2+0.8)
		for i in range(0,Npoints2):
			a.append(np.random.exponential(0.5, ))

		b = []
		for i in range(0,Npoints1):
			b.append(np.random.randn()*0.4+0.0)
		for i in range(0,Npoints2):
			b.append(np.random.randn()*0.6-1.0)

		c = []
		for i in range(0,Npoints1):
			c.append(GetAngle2(0.9, 12.0, 1.0))
		for i in range(0,Npoints2):
			c.append(GetAngle2( 0.8, 17.0, 0.5 ))

		e = np.column_stack((a,b,c))

	if Dim == '3D':
		a = []
		for i in range(0,Npoints1):
			a.append(np.random.randn()*0.2+0.8)
		for i in range(0,Npoints2):
			a.append(np.random.exponential(0.5, ))
		for i in range(0,Npoints3):
			a.append(np.random.uniform()*5.0)

		b = []
		for i in range(0,Npoints1):
			b.append(np.random.randn()*0.4+0.0)
		for i in range(0,Npoints2):
			b.append(np.random.randn()*0.6-1.0)
		for i in range(0,Npoints3):
			b.append(np.random.randn()*2.0+1.0)
			
		c = []
		for i in range(0,Npoints1):
			c.append(GetAngle2(0.9, 12.0, 1.0))
		for i in range(0,Npoints2):
			c.append(GetAngle2( 0.8, 17.0, 0.5 ))
		for i in range(0,Npoints3):
			c.append(np.random.uniform()*2.0-1.0)

		d = []
		for i in range(0,Npoints1):
			d.append(np.random.poisson(14.0,))
		for i in range(0,Npoints2):
			d.append(np.random.binomial(20,0.5,))
		for i in range(0,Npoints3):
			d.append(np.random.poisson(6.0,))

		e = np.column_stack((a,b,c,d))

	np.savetxt('GenData.txt', e, fmt='%8.4f')

	return(e)

