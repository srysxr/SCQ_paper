def cp_vec(X_score_vec, cal_score_vec):
    x = (np.sum(cal_score_vec[:, None] <= X_score_vec, axis=0) + 1) / (1 + len(cal_score_vec))
    return x

def WCP(u_vec, tu_vec, alpha):
    def Q(t):
        Q = (1 + np.sum((tu_vec < u_vec) & (tu_vec <= t))) / max(np.sum((u_vec < tu_vec) & (u_vec <= t)), 1)
        return Q
    tau = -1
    for i in np.append(u_vec, tu_vec):
        if Q(i) <= alpha:
            if i >= tau:
                tau = i
    R_vec = (u_vec < tu_vec) & (u_vec <= tau)
    return R_vec

def pis(x, y, s=None, lamda=0.1, h=50, eps = 0.0001):

    m = len(x)
    p_est = np.zeros(m)

    if s is None:
        s = np.arange(1, m + 1)  # auxiliary variable
        for i in range(m):
            kht = stats.norm.pdf(loc=0, scale=h, x=s-(i + 1))  # (i+1) because `s` starts from 1
            p_est[i] = 1 - (np.sum(kht[x >= lamda]) + np.sum(kht[y >= lamda])) / (2 * (1 - lamda) * np.sum(kht))
            # p_est[i] = 1 - (np.sum(kht[y >= lamda])) / ((1 - lamda) * np.sum(kht))

    p_est[p_est <= 0] = eps
    p_est[p_est >= 0.5] = 0.5 - eps
    return p_est

def storey_pi(cp_vector,lamda):
    m = len(cp_vector)
    pi = (1+np.sum(cp_vector>=lamda)) / (m*(1-lamda))
    return pi

def estimate_pi_hat(S, p, p_tilde, lamda, eps = 0.0001):
    W = (S[:, None] == S[None, :]).astype(int)

    I1 = (p > lamda).astype(int)
    I2 = (p_tilde > lamda).astype(int)

    numerator = np.sum(W * (I1 + I2), axis=1)
    denominator = 2 * (1 - lamda) * np.sum(W, axis=1)

    denominator = np.where(denominator == 0, 1e-10, denominator)

    pi_hat = 1 - numerator / denominator
    pi_hat[pi_hat <= 0] = eps
    pi_hat[pi_hat >= 0.5] = 0.5-eps
    return pi_hat

