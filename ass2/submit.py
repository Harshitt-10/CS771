import numpy as np
import sklearn
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of numpy or sklearn e.g. np.random.randint for random initialization or sklearn.metrics.accuracy_score to calculate accuracy of a learnt model
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py


################################################################
# Helper functions (mirror the ones used in the eval harness)
################################################################

def apuf_features(X):
	# Standard {0,1} -> linear-model feature map for an n-bit arbiter PUF
	return np.flip(np.cumprod(np.flip(1 - 2 * X, axis=1), axis=1), axis=1)


def insert(X, z, midpos):
	# Insert per-example latent bit z (shape (n,)) at column midpos of X
	return np.concatenate([X[:, :midpos], z[:, np.newaxis], X[:, midpos:]], axis=1)


def _predict(X, w, b):
	pred = np.zeros((len(X),))
	score = X.dot(w) + b
	pred[score > 0] = 1
	return pred


def _acc(y, pred):
	return np.average(y == pred)


def _fit_linear(Phi, y, C=1.0, tol=1e-2, max_iter=1000):
	# LinearSVC with squared_hinge is used throughout as a fast convex
	# surrogate for the logistic-loss (w,b)/(u,a) sub-problems that show
	# up at each alternating-optimization step.
	clf = LinearSVC(C=C, loss='squared_hinge', tol=tol, max_iter=max_iter)
	clf.fit(Phi, y)
	return clf.coef_[0], clf.intercept_[0]


def _log_sigmoid(x):
	# numerically stable log(sigmoid(x))
	return -np.logaddexp(0, -x)


MIDPOS = 8


################################################################
# Part 2: my_latent
################################################################

################################
# Non Editable Region Starting #
################################
def my_latent( X, y ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to find optimal combination of 17-bit model and latent variables
	# You may perform repeated initialization internally yourself to avoid unluckiness
	# However, the evaluation code will itself run repetitions to take the best result

	n, d = X.shape
	rng = np.random.default_rng()

	Phi0 = apuf_features(insert(X, np.zeros(n), MIDPOS))
	Phi1 = apuf_features(insert(X, np.ones(n), MIDPOS))
	ysign = 2 * y - 1

	n_restarts = 6
	n_iters = 15

	best_w = best_b = best_z = None
	best_acc = -1.0

	for r in range(n_restarts):
		z = rng.integers(0, 2, size=n).astype(float)
		prev_acc = -1.0
		w = b = None
		acc = 0.0
		for it in range(n_iters):
			Phi17 = apuf_features(insert(X, z, MIDPOS))
			w, b = _fit_linear(Phi17, y)

			# z-step: for each example independently pick whichever
			# hidden-bit value gives higher margin toward the correct label
			s0 = Phi0.dot(w) + b
			s1 = Phi1.dot(w) + b
			z_new = (s1 * ysign > s0 * ysign).astype(float)

			if np.array_equal(z_new, z):
				z = z_new
				break
			z = z_new

			Phi17b = apuf_features(insert(X, z, MIDPOS))
			w, b = _fit_linear(Phi17b, y)
			acc = _acc(y, _predict(Phi17b, w, b))
			if abs(acc - prev_acc) < 1e-9:
				break
			prev_acc = acc

		Phi17f = apuf_features(insert(X, z, MIDPOS))
		w, b = _fit_linear(Phi17f, y, tol=1e-3, max_iter=3000)
		acc = _acc(y, _predict(Phi17f, w, b))

		if acc > best_acc:
			best_acc = acc
			best_w, best_b, best_z = w, b, z.copy()

		if best_acc >= 0.999:
			break

	return best_w, best_b, best_z.astype(bool)


################################################################
# Part 4: my_latent_updated
################################################################

################################
# Non Editable Region Starting #
################################
def my_latent_updated( X, y ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to find the optimal 17-bit and 16-bit models
	# You may perform repeated initialization internally yourself to avoid unluckiness
	# However, the evaluation code will itself run repetitions to take the best result

	n, d = X.shape
	rng = np.random.default_rng()

	PhiX = apuf_features(X)
	Phi0 = apuf_features(insert(X, np.zeros(n), MIDPOS))
	Phi1 = apuf_features(insert(X, np.ones(n), MIDPOS))
	ysign = 2 * y - 1

	n_restarts = 40
	n_iters = 25
	check_iter = 4
	check_thresh = 0.85

	best_w = best_b = best_u = best_a = None
	best_acc = -1.0

	for r in range(n_restarts):
		z = rng.integers(0, 2, size=n).astype(float)
		prev_acc = -1.0
		w = b = u = a = None
		acc = 0.0
		for it in range(n_iters):
			Phi17 = apuf_features(insert(X, z, MIDPOS))
			w, b = _fit_linear(Phi17, y)
			u, a = _fit_linear(PhiX, z)

			s_wb0 = Phi0.dot(w) + b
			s_wb1 = Phi1.dot(w) + b
			s_ua = PhiX.dot(u) + a

			# z-step: weigh BOTH the response-consistency term (w,b)
			# and the latent-prior term (u,a), as required once the
			# flat 0.5 prior is replaced by the hidden 16-bit PUF model
			resp0 = _log_sigmoid(ysign * s_wb0)
			resp1 = _log_sigmoid(ysign * s_wb1)
			prior0 = _log_sigmoid(-1 * s_ua)
			prior1 = _log_sigmoid(1 * s_ua)

			z_new = ((resp1 + prior1) > (resp0 + prior0)).astype(float)

			Phi17c = apuf_features(insert(X, z_new, MIDPOS))
			w, b = _fit_linear(Phi17c, y)
			acc = _acc(y, _predict(Phi17c, w, b))

			# abandon early if this restart is clearly stuck in a bad basin
			if it == check_iter and acc < check_thresh:
				break

			if np.array_equal(z_new, z) and abs(acc - prev_acc) < 1e-9:
				z = z_new
				break
			z = z_new
			prev_acc = acc

		if acc < check_thresh:
			continue

		u, a = _fit_linear(PhiX, z, tol=1e-3, max_iter=3000)

		if acc > best_acc:
			best_acc = acc
			best_w, best_b, best_u, best_a = w, b, u, a

		if best_acc > 0.995:
			break

	return best_w, best_b, best_u, best_a
