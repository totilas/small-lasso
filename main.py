from numpy.linalg import inv
import numpy as np

from matplotlib import pyplot as plt

np.random.seed(0)



def lasso(v, X,y):
	# we have X omega - y + v = 0
	w, _, _, _ = np.linalg.lstsq(X, y-v, rcond=None)

	return .5 * v.T @ v + lambda_n * np.fabs(w).sum()

def dual(v, y):
	return .5 * v.T @ v - v.T @ y




def centering_step(Q,p,A,b,t,v0,eps):
	"""the Newton method to solve the centering step given the inputs (Q, p, A, b), the
	barrier method parameter t, initial variable v 0 and a target precision epsilon. 
	Use a backtracking line search with appropriate parameters.

	Outputs :sequence of variables iterates (v i ) i=1,...,n_eps , where n_eps is
			 the number of iterations to obtain the epsilon precision. 
	"""

	maxiter = 100
	m = len(Q[0])

	alpha = .5 # alpha in [.01, .3]
	beta = .7 # in [0,1]

	z = np.zeros(2*d)

	def phi(x,t):
		return t * (x.T @ Q @ x + p.T @ x) - (np.log(np.maximum(z, b - A @ x))).sum()

	def grad_phi(x, t):
		sol = t * (Q + Q.T) @ x +t * p
		for i in range(m):
			sol += (1/(b[i] - A[i].T @ x)) * A[i] 
		return sol

	def hess_phi(x,t):
		sol = (Q + Q.T) * t
		for i in range(m):
			sol += (1/(b[i] - A[i].T @ x)**2) * np.outer(A[i], A[i])
		return sol

	

	nb_it = 0

	x = v0
	v = [x]

	while nb_it < maxiter:

		nb_it +=1
		#Compute the Newton step and decrement
		grad = grad_phi(x,t) 
		delta_x = - inv(hess_phi(x,t)) @ grad
		lambda_s = - grad.T @ delta_x 

		# backtracking line search
		t_b = 1
		current_phi = phi(x,t)
		new_x = x + t_b * delta_x

		# check if x_new in the feasible place
		while any(b - A @ new_x) <= 0:
			t_b *= beta
			new_x = x + t_b * delta_x

		while phi(new_x,t) > current_phi + alpha * t_b * lambda_s:
			#print(phi(new_x,t), current_phi, alpha, t_b,lambda_s)
			t_b *= beta
			new_x = x + t_b * delta_x
			# check if x_new in the feasible place
			while any(b - A @ new_x) <= 0:
				t_b *= beta
				new_x = x + t_b * delta_x 

		# update v
		#print(lambda_s)
		x = new_x
		v.append(x)

		#Stopping criterion
		if lambda_s  / 2 < eps:
			break

	return v

def barr_method(Q,p,A,b,v0,eps, mu, t0):
	"""
	barrier method to solve QP using precedent function given the data inputs (Q, p, A, b),
	a feasible point v 0 , a precision criterion epsilon. 
	Outputs: the sequence of variables iterates (v i ) i=1,...,n_epsilon 
	"""
	# number of constraints : dim of A
	m = d
	
	v_list = [v0]
	t = t0

	# Stopping criterion to have an espilon approx
	while m/t>=eps:
		# centering step:
		v_new = centering_step(Q,p,A,b,t,v_list[-1],eps)
		#v_list.append(v_new[-1]) # with only important steps
		v_list += v_new # with all value
		# Increase t
		t *= mu

	return v_list





"""
Test your function on randomly generated matrices X and observations y with λ = 10.
 . Repeat for different values of the barrier method parameter
μ = 2, 15, 50, 100, . . . and check the impact on w. What would be an appropriate choice
for μ ?
"""

colors=["goldenrod","orange","red", "magenta", "violet", "darkblue", "blue", "sky blue", "turquoise", "pale green", "grass green", "army green" ]

n = 10
d = 200
lambda_n = 10
Q =.5 * np.eye(n)
p = np.random.rand(n)
y = - p
X = np.random.rand(n,d)
A = np.concatenate((X.T, -X.T), axis=0)

v0 = np.zeros(n)
eps = 1e-6
b = np.array([lambda_n]  *  (2*d))


mus = [2, 3, 5, 8, 10, 12,15, 17, 20, 50, 70, 110]

f = plt.figure()
accuracy = []

for i, mu in enumerate(mus):
	t0 = 1 #little at first
	v = barr_method(Q,p,A,b, v0, eps, mu, t0)
	# Plot precision criterion and gap f (v t ) − f ∗ in semilog scale (using the best value found for f as a surrogate for f ∗ )
	opt_value = lasso(v[-1],X,  y)
	gap = np.array([lasso(vv,X, y) - opt_value for vv in v]) 
	x_ax = [ i for i in range(len(v))]
	accuracy.append(opt_value)
	ax = plt.plot(x_ax, gap, label=str(mu), color="xkcd:"+colors[i])

plt.yscale('log')
plt.title("Convergence")
plt.xlabel("steps")
plt.ylabel("gap")
plt.legend()



n = 5
d = 300
lambda_n = 10
Q =.5 * np.eye(n)
p = np.random.rand(n)
y = - p
X = np.random.rand(n,d)
A = np.concatenate((X.T, -X.T), axis=0)

v0 = np.zeros(n)
eps = 1e-6
b = np.array([lambda_n]  *  (2*d))

f = plt.figure()
accuracy2 = []

for i, mu in enumerate(mus):
	t0 = 1 #little at first
	v = barr_method(Q,p,A,b, v0, eps, mu, t0)
	# Plot precision criterion and gap f (v t ) − f ∗ in semilog scale (using the best value found for f as a surrogate for f ∗ )
	opt_value = lasso(v[-1],X,  y)
	gap = np.array([lasso(vv,X, y) - opt_value for vv in v]) 
	x_ax = [ i for i in range(len(v))]
	accuracy2.append(opt_value)
	ax = plt.plot(x_ax, gap, label=str(mu), color="xkcd:"+colors[i])

plt.yscale('log')
plt.title("Convergence 2")
plt.xlabel("steps")
plt.ylabel("gap")
plt.legend()



plt.figure()
plt.plot(mus, accuracy)
plt.title("Evolution of w with mu")

plt.figure()
plt.plot(mus, accuracy2)

plt.title("Evolution of w with mu")
plt.show()