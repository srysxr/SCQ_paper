def generate_data(m1, pi1, m2, pi2, ntrain, ncal, n1_count, p, a):
    m = m1 + m2
    pi = 0.01 * np.ones(m)
    pi[200:201 + h] = pi1
    pi[600:601 + h] = pi1
    pi[1000:1001 + h] = pi2
    pi[1400:1401 + h] = pi2
    theta = np.random.binomial(n=1, p=pi, size=m).reshape(m, 1)

    F01 = np.random.randn(m1, p)
    F11 = np.random.normal(a, 1, size=(m1, p))
    F02 = np.random.randn(m2, p)
    F12 = np.random.normal(-2, 0.5, size=(m2, p))

    X_test = np.zeros((m, p))
    X_test[0:m1, ] = (1 - theta[0:m1, ]) * F01 + theta[0:m1, ] * F11
    X_test[m1:, ] = (1 - theta[m1:, ]) * F02 + theta[m1:, ] * F12

    X_train = np.random.randn(ntrain, p)
    X_cal = np.random.randn(ncal, p)
    X_mirror = np.random.randn(m, p)
    Y1 = np.random.normal(a, 1, size=(int(n1_count), p))
    Y2 = np.random.normal(-2, 0.5, size=(int(n1_count), p))
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
def PTAMS(u_vec,tu_vec,cp_mins, bht):
    lamdas = bh_func(cp_mins, bht)[1]
    tp = np.random.binomial(1, 0.5, sum(cp_mins > lamdas))

    u_min = np.minimum(u_vec, tu_vec)
    u_max = np.maximum(u_vec, tu_vec)
    u = np.zeros(m)
    tu = np.zeros(m)
    tu[cp_mins > lamdas] = u_max[cp_mins > lamdas] * tp + u_min[cp_mins > lamdas] * (1 - tp)
    tu[cp_mins <= lamdas] = np.maximum(u_vec[cp_mins <= lamdas], tu_vec[cp_mins <= lamdas])
    u[cp_mins > lamdas] = u_max[cp_mins > lamdas] * (1 - tp) + u_min[cp_mins > lamdas] * tp
    u[cp_mins <= lamdas] = np.minimum(u_vec[cp_mins <= lamdas], tu_vec[cp_mins <= lamdas])
    sf = sum(WCP(u, tu, alpha))
    R = WCP(u_vec, tu_vec, alpha)
    return sf, R, sum(R)

def calculate_fdp_ntp(theta, R):
    return sum((1 - theta) * R) / max(sum(R), 1), sum(theta * R) / sum(theta)
def process_single_rep(i, j):

    X_train, X_cal, X_test, X_mirror, Y, theta, pi = generate_data(
        m1, pi1, m2, pi2, ntrain, ncal, n1_count, p, mu_vec1[i]
    )
    theta = theta.reshape((m,))

    fdp_results = {}
    ntp_results = {}

    clfo = OneClassSVM(kernel='rbf')
    clfo.fit(X_train)
    test_scoreo = clfo.score_samples(X_test)
    mirror_scoreo = clfo.score_samples(X_mirror)
    cal_scoreo = clfo.score_samples(X_cal)
    cp_testo = cp_vec(test_scoreo, cal_scoreo)
    cp_mirroro = cp_vec(mirror_scoreo, cal_scoreo)
    X_Cal = np.vstack((X_cal, X_mirror))

    lamdao = bh_func(np.minimum(cp_testo, cp_mirroro), BHT)[1]
    pi_hato = pis(cp_testo, cp_mirroro, lamda=lamdao, h=bdw)
    w_hato = pi_hato / (1 / 2 - pi_hato)
    u_veco = cp_testo / w_hato
    tu_veco = cp_mirroro / w_hato
    R_SCQO = WCP(u_veco, tu_veco, alpha)
    fdp_results['SCQSP'], ntp_results['SCQSP'] = calculate_fdp_ntp(theta, R_SCQO)

    L1 = np.vstack((X_mirror, X_test))
    L0 = np.vstack((X_cal, X_train))
    sam = np.concatenate([L0, L1], axis=0)
    label = np.concatenate([[0] * L0.shape[0], [1] * L1.shape[0]])
    clf = RandomForestClassifier(max_depth=10)
    clf.fit(sam, label)

    s_test = clf.predict_proba(X_test)[:, 0]
    s_cal = clf.predict_proba(X_mirror)[:, 0]


    p_d = cp_vec(s_test, s_cal)
    hat_pi_0 = storey_pi(p_d, lamda=0.5)
    R_AD = multipletests(p_d, alpha=alpha/hat_pi_0, method='fdr_bh')[0]
    fdp_results['ADRF'], ntp_results['ADRF'] = calculate_fdp_ntp(theta, R_AD)

    X_mix = np.vstack((X_mirror, X_test))
    X_null = np.vstack((X_train, X_cal))

    mix_kd = KernelDensity().fit(X_mix)
    null_kd = KernelDensity().fit(X_null)

    pro_test = np.exp(null_kd.score_samples(X_test)) / np.exp(mix_kd.score_samples(X_test))
    pro_cal = np.exp(null_kd.score_samples(X_mirror)) / np.exp(mix_kd.score_samples(X_mirror))

    p_AD = cp_vec(pro_test, pro_cal)
    hat_pi_0 = storey_pi(p_AD, lamda=0.5)
    R_AD = multipletests(p_AD, alpha=alpha/hat_pi_0, method='fdr_bh')[0]
    fdp_results['ADKDE'], ntp_results['ADKDE'] = calculate_fdp_ntp(theta, R_AD)

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

    X_mix = np.vstack((X_cal, X_mirror, X_test))
    X_null = X_train

    mix_kd = KernelDensity().fit(X_mix)
    null_kd = KernelDensity().fit(X_null)

    pro_test = np.exp(null_kd.score_samples(X_test)) / np.exp(mix_kd.score_samples(X_test))
    pro_cal = np.exp(null_kd.score_samples(X_mirror)) / np.exp(mix_kd.score_samples(X_mirror))
    scal = np.exp(null_kd.score_samples(X_cal)) / np.exp(mix_kd.score_samples(X_cal))

    cptest_scq_kde = cp_vec(pro_test, scal)
    cpmirror_scq_kde = cp_vec(pro_cal, scal)

    lamdao = bh_func(np.minimum(cptest_scq_kde, cpmirror_scq_kde), BHT)[1]
    pi_hato_kde = pis(cptest_scq_kde, cpmirror_scq_kde, lamda=lamdao, h=bdw)
    w_hato = pi_hato_kde / (1 / 2 - pi_hato_kde)
    u_vec_scq_kde = cptest_scq_kde / w_hato
    tu_vec_scq_kde = cpmirror_scq_kde / w_hato
    R_SCQO = WCP(u_vec_scq_kde, tu_vec_scq_kde, alpha)
    fdp_results['SCQPUKDE'], ntp_results['SCQPUKDE'] = calculate_fdp_ntp(theta, R_SCQO)

    r_test = pro_test
    r_mirror = pro_cal
    pi_hato = pi_hato_kde
    clfdr_test = np.minimum(0.9999, (1 - pi_hato) * r_test)
    clfdr_mirror = np.minimum(0.9999, (1 - pi_hato) * r_mirror)
    RR_test = (0.5 - pi_hato) * clfdr_test / ((1 - pi_hato) * (1 - clfdr_test))
    RR_mirror = (0.5 - pi_hato) * clfdr_mirror / ((1 - pi_hato) * (1 - clfdr_mirror))
    R_CL = WCP(RR_test, RR_mirror, alpha)
    fdp_results['CL_KD'], ntp_results['CL_KD'] = calculate_fdp_ntp(theta, R_CL)

    r_test = s_test
    r_mirror = s_cal
    pi_hato = pi_hato_purf
    pi_hato = np.minimum(0.4999, pi_hato)
    clfdr_test = np.minimum(0.9999, (1 - pi_hato) * r_test)
    clfdr_mirror = np.minimum(0.9999, (1 - pi_hato) * r_mirror)
    R_test = (0.5 - pi_hato) * clfdr_test / ((1 - pi_hato) * (1 - clfdr_test))
    R_mirror = (0.5 - pi_hato) * clfdr_mirror / ((1 - pi_hato) * (1 - clfdr_mirror))
    R_CL = WCP(R_test, R_mirror, alpha)
    fdp_results['CL'], ntp_results['CL'] = calculate_fdp_ntp(theta, R_CL)

    AMS_result = {}
    AMS_result['SCQ-OCC'] = PTAMS(u_veco,tu_veco,np.minimum(cp_testo,cp_mirroro),bht)
    AMS_result['SCQ-PU'] = PTAMS(u_vec_scq_pu,tu_vec_scq_pu,np.minimum(cptest_scq_pu,cpmirror_scq_pu),bht)
    AMS_result['SCQ-KDE'] = PTAMS(u_vec_scq_kde,tu_vec_scq_kde,np.minimum(cptest_scq_kde,cpmirror_scq_kde),bht)
    sf_max_idx = np.argmax([res[0] for res in AMS_result.values()])
    model_select = list(AMS_result.keys())[sf_max_idx]
    R_PTAMS = list(AMS_result.values())[sf_max_idx][1]
    print(model_select, "sf:", [res[0] for res in AMS_result.values()])
    print('ntp:', [res[2] for res in AMS_result.values()])

    fdp_results['SCQAMS'], ntp_results['SCQAMS'] = calculate_fdp_ntp(theta, R_PTAMS)

    print(f'rep {j + 1} for variable {i + 1}')
    return i, j, fdp_results, ntp_results

