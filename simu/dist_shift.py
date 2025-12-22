def generate_data(m1, pi1, m2,pi2, ntrain, ncal, n1_count, p, a,d):
    m = m1 + m2
    pi = 0.01 * np.ones(m)
    pi[200:201+h] = pi1
    pi[600:601+h] = pi1
    pi[1000:1001+h] = pi2
    pi[1400:1401+h] = pi2
    theta = np.random.binomial(n=1, p=pi, size=m).reshape(m, 1)
    F01 = np.random.randn(m1, p)
    F11 = np.random.normal(a, 1, size=(m1, p))
    F02 = np.random.randn(m2, p)
    F12 = np.random.normal(-2, 0.5, size=(m2, p))

    X_test = np.zeros((m,p))
    X_test[0:m1,] = (1 - theta[0:m1,]) * F01 + theta[0:m1,] * F11
    X_test[m1:,] = (1 - theta[m1:,]) * F02 + theta[m1:,] * F12

    X_train = np.random.randn(ntrain, p)
    X_cal = np.random.randn(ncal, p)
    X_mirror = np.random.randn(m, p)
    Y1 = np.random.normal(d, 1, size=(n1_count, p))
    Y2 = np.random.normal(-2, 0.5, size=(n1_count, p))
    Y = np.vstack((Y1, Y2))
    return X_train, X_cal, X_test, X_mirror, Y, theta,pi
def calculate_fdp_ntp(theta, R):
    return sum((1 - theta) * R) / max(sum(R), 1), sum(theta * R) / sum(theta)
def process_single_rep(i, j):

    X_train, X_cal, X_test, X_mirror, Y, theta, pi = generate_data(
        m1, pi1, m2, pi2, ntrain, ncal, n1_count, p, a,d_vec1[i]
    )
    theta = theta.reshape((m,))

    sam = np.concatenate([X_train, Y], axis=0)
    label = np.concatenate([[0] * X_train.shape[0], [1] * Y.shape[0]])

    fdp_results = {}
    ntp_results = {}

    clfo = OneClassSVM(kernel='rbf')
    clfo.fit(X_train)
    test_scoreo = clfo.decision_function(X_test)
    mirror_scoreo = clfo.decision_function(X_mirror)
    cal_scoreo = clfo.decision_function(X_cal)
    cp_testo = cp_vec(test_scoreo, cal_scoreo)
    cp_mirroro = cp_vec(mirror_scoreo, cal_scoreo)
    X_Cal = np.vstack((X_cal, X_mirror))
    Cal_scoreo = clfo.decision_function(X_Cal)
    Cp_testo = cp_vec(test_scoreo, Cal_scoreo)
    hat_pi_0 = storey_pi(Cp_testo, lamda=0.5)
    R_CPBH = multipletests(Cp_testo, alpha/hat_pi_0, method='fdr_bh')[0]
    fdp_results['CPSR'], ntp_results['CPSR'] = calculate_fdp_ntp(theta, R_CPBH)

    lamdao = bh_func(np.minimum(cp_testo, cp_mirroro), BHT)[1]
    pi_hato = pis(cp_testo, cp_mirroro, lamda=lamdao, h=bdw)
    w_hato = pi_hato / (1 / 2 - pi_hato)
    u_veco = cp_testo / w_hato
    tu_veco = cp_mirroro / w_hato
    R_SCQO = WCP(u_veco, tu_veco, alpha)
    fdp_results['SCQSR'], ntp_results['SCQSR'] = calculate_fdp_ntp(theta, R_SCQO)

    clfb = RandomForestClassifier(max_depth=10)
    clfb = clfb.fit(sam, label)

    mirror_score = clfb.predict_proba(X_mirror)[:, 0]
    test_score = clfb.predict_proba(X_test)[:, 0]
    cal_score = clfb.predict_proba(X_cal)[:, 0]

    cp_test = cp_vec(test_score, cal_score)
    cp_mirror = cp_vec(mirror_score, cal_score)
    lamda = bh_func(np.minimum(cp_test, cp_mirror), BHT)[1]
    pi_hat = pis(cp_test, cp_mirror, lamda=lamda, h=bdw)
    w_hat = pi_hat / (1 / 2 - pi_hat)
    u_vec = cp_test / w_hat
    tu_vec = cp_mirror / w_hat
    R_SCQB = WCP(u_vec, tu_vec, alpha)
    fdp_results['SCQRF'], ntp_results['SCQRF'] = calculate_fdp_ntp(theta, R_SCQB)

    np.random.shuffle(Y)
    Y_train = Y[:int(len(Y) / 2), :]
    Y_cal = Y[int(len(Y) / 2):, :]

    clf0 = OneClassSVM(kernel='rbf')
    clf0.fit(X_train)
    clf1 = OneClassSVM(kernel='rbf')
    clf1.fit(Y_train)

    test_score0 = clf0.decision_function(X_test)
    mirror_score0 = clf0.decision_function(X_mirror)
    cal_score0 = clf0.decision_function(X_cal)

    test_score1 = clf1.decision_function(X_test)
    mirror_score1 = clf1.decision_function(X_mirror)
    cal_score1 = clf1.decision_function(Y_cal)

    if np.median(cal_score1) < np.median(clf1.decision_function(X_cal)):
        test_score1 = -test_score1
        mirror_score1 = -mirror_score1
        cal_score1 = -cal_score1

    u0_vec = cp_vec(test_score0, cal_score0)
    u1_vec = cp_vec(test_score1, cal_score1)
    tu0_vec = cp_vec(mirror_score0, cal_score0)
    tu1_vec = cp_vec(mirror_score1, cal_score1)

    u_icp = u0_vec / u1_vec
    tu_icp = tu0_vec / tu1_vec

    cp_icp = cp_vec(u_icp, tu_icp)
    hat_pi_0_icp = storey_pi(cp_icp, lamda=0.5)
    R_ICP = multipletests(cp_icp, alpha/hat_pi_0_icp, method='fdr_bh')[0]
    fdp_results['ICP'], ntp_results['ICP'] = calculate_fdp_ntp(theta, R_ICP)

    print(f'rep {j + 1} for variable {i + 1}')
    return i, j, fdp_results, ntp_results



