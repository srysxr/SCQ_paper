def generate_data(m1, pi1, m2, pi2, ntrain, ncal, n1_count, p, a):
    m = m1+m2
    pi1 = m ** (-beta)
    pi2 = pi1 * (rpi)

    h = int(100/3000*m)
    pi = 0.01 * np.ones(m)
    pi[int(2/30*m):int(2/30*m)+1 + h] = pi1
    pi[int(6/30*m):int(6/30*m)+1 + h] = pi1
    pi[int(10/30*m):int(10/30*m)+1 + h] = pi2
    pi[int(14/30*m):int(14/30*m)+1 + h] = pi2
    theta = np.random.binomial(n=1, p=pi, size=m).reshape(m, 1)

    F01 = np.random.randn(m1, p)
    F11 = np.random.normal(a, 1, size=(m1, p))
    F02 = np.random.randn(m2, p)
    F12 = np.random.normal(a, 1, size=(m2, p))

    X_test = np.zeros((m, p))
    X_test[0:m1, ] = (1 - theta[0:m1, ]) * F01 + theta[0:m1, ] * F11
    X_test[m1:, ] = (1 - theta[m1:, ]) * F02 + theta[m1:, ] * F12

    X_train = np.random.randn(ntrain, p)
    X_cal = np.random.randn(ncal, p)
    X_mirror = np.random.randn(m, p)
    Y1 = np.random.normal(a, 1, size=(int(n1_count), p))
    Y2 = np.random.normal(a, 1, size=(int(n1_count), p))
    Y = np.concatenate((Y1, Y2))

    return X_train, X_cal, X_test, X_mirror, Y, theta, pi
def compute_scores(clf, X_train, X_cal, X_test, X_mirror):
    clfo = clf.fit(X_train)
    cal_scoreo = clfo.score_samples(X_cal)
    test_scoreo = clfo.score_samples(X_test)
    mirror_scoreo = clfo.score_samples(X_mirror)

    scores = {
        'cp_testo': cp_vec(test_scoreo, cal_scoreo),
        'cp_mirroro': cp_vec(mirror_scoreo, cal_scoreo),
    }

    return scores
def compute_scores2(clf, X_train, X_cal, X_test, X_mirror, Y):
    sam = np.concatenate([X_train, Y], axis=0)
    label = np.concatenate([[0] * X_train.shape[0], [1] * Y.shape[0]])
    clfb = clf.fit(sam, label)
    cal_scoreb = clfb.predict_proba(X_cal)[:, 0]
    test_scoreb = clfb.predict_proba(X_test)[:, 0]
    mirror_scoreb = clfb.predict_proba(X_mirror)[:, 0]

    scores = {
        'cp_testb': cp_vec(test_scoreb, cal_scoreb),
        'cp_mirrorb': cp_vec(mirror_scoreb, cal_scoreb),
    }

    return scores
def calculate_fdp_ntp(theta, R):
    return sum((1 - theta) * R) / max(sum(R), 1), sum(theta * R) / sum(theta)
def process_single_rep(i, j):

    X_train, X_cal, X_test, X_mirror, Y, theta, pi = generate_data(
        int(m_vec[i]), pi1, int(m_vec[i]), pi2, ntrain, ncal, n1_count, p, (r * ((np.log(int(m_vec[i])*2))**(k)) ) ** (1/2)
    )
    m = 2*int(m_vec[i])
    theta = theta.reshape((m,))

    fdp_results = {}
    ntp_results = {}

    clfo = OneClassSVM()
    clfo.fit(X_train)
    test_scoreo = clfo.score_samples(X_test)
    mirror_scoreo = clfo.score_samples(X_mirror)
    cal_scoreo = clfo.score_samples(X_cal)
    cp_testo = cp_vec(test_scoreo, cal_scoreo)
    cp_mirroro = cp_vec(mirror_scoreo, cal_scoreo)

    lamdao = bh_func(np.minimum(cp_testo, cp_mirroro), BHT)[1]
    pi_hato = pis(cp_testo, cp_mirroro, lamda=lamdao, h=bdw)
    w_hato = pi_hato / (1 / 2 - pi_hato)
    u_veco = cp_testo / w_hato
    tu_veco = cp_mirroro / w_hato
    R_SCQO = WCP(u_veco, tu_veco, alpha)
    fdp_results['SCQSP'], ntp_results['SCQSP'] = calculate_fdp_ntp(theta, R_SCQO)

    X_mix = np.vstack((X_cal, X_mirror, X_test))
    X_null = X_train

    mix_kd = KernelDensity().fit(X_mix)
    null_kd = KernelDensity().fit(X_null)

    pro_test = np.exp(null_kd.score_samples(X_test)) / np.exp(mix_kd.score_samples(X_test))
    pro_cal = np.exp(null_kd.score_samples(X_mirror)) / np.exp(mix_kd.score_samples(X_mirror))
    scal = np.exp(null_kd.score_samples(X_cal)) / np.exp(mix_kd.score_samples(X_cal))

    ptest = cp_vec(pro_test, scal)
    pmirror = cp_vec(pro_cal, scal)

    lamdao = bh_func(np.minimum(ptest, pmirror), BHT)[1]
    pi_hato = pis(ptest, pmirror, lamda=lamdao, h=bdw)
    w_hato = pi_hato / (1 / 2 - pi_hato)
    u_veco = ptest / w_hato
    tu_veco = pmirror / w_hato
    R_SCQO = WCP(u_veco, tu_veco, alpha)
    fdp_results['SCQPUKDE'], ntp_results['SCQPUKDE'] = calculate_fdp_ntp(theta, R_SCQO)

    x = X_test
    xnull = np.vstack((X_train, X_cal))
    proc = AdaDetectERM(scoring_fn=RandomForestClassifier(max_depth=10),
                        split_size=(len(X_train)) / (len(xnull)))
    proc.fit(x, alpha, xnull)
    s_test1 = proc.test_statistics
    s_cal1 = proc.null_statistics
    cptest_scq_pu = cp_vec(s_test1, s_cal1)
    x = X_mirror
    xnull = np.vstack((X_train, X_cal))
    proc = AdaDetectERM(scoring_fn=RandomForestClassifier(max_depth=10),
                        split_size=(len(X_train)) / (len(xnull)))
    proc.fit(x, alpha, xnull)
    s_test2 = proc.test_statistics
    s_cal2 = proc.null_statistics
    cpmirror_scq_pu = cp_vec(s_test2, s_cal2)

    lamdao = bh_func(np.minimum(cptest_scq_pu, cpmirror_scq_pu), BHT)[1]
    pi_hato_purf = pis(cptest_scq_pu, cpmirror_scq_pu, lamda=lamdao, h=bdw)
    w_hato = pi_hato_purf / (1 / 2 - pi_hato_purf)
    u_vec_scq_pu = cptest_scq_pu / w_hato
    tu_vec_scq_pu = cpmirror_scq_pu / w_hato
    R_SCQPURF = WCP(u_vec_scq_pu, tu_vec_scq_pu, alpha)
    fdp_results['SCQPURF'], ntp_results['SCQPURF'] = calculate_fdp_ntp(theta, R_SCQPURF) 

    print(f'rep {j + 1} for variable {i + 1}')
    return i, j, fdp_results, ntp_results
