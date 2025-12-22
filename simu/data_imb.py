def generate_data(m1, pi1, m2,pi2, ntrain, ncal, n1_count, p, a,b):
    m = m1 + m2
    pi = 0.01 * np.ones(m)

    pi[200:201+h] = pi1
    pi[600:601+h] = pi1
    pi[1000:1001+h] = pi2
    pi[1400:1401+h] = pi2
    theta = np.random.binomial(n=1, p=pi, size=m).reshape(m, 1)

    F01 = np.random.randn(m1,p) + np.random.uniform(-b, b, size=(m1, p))
    F11 = np.sqrt(a) * np.random.randn(m1,p) + np.random.uniform(-b, b, size=(m1, p))
    F02 = np.random.randn(m2,p) + np.random.uniform(-b, b, size=(m2, p))
    F12 = np.sqrt(5*a) * np.random.randn(m2,p) + np.random.uniform(-b, b, size=(m2, p))

    X_test = np.zeros((m,p))
    X_test[0:m1,] = (1 - theta[0:m1,]) * F01 + theta[0:m1,] * F11
    X_test[m1:,] = (1 - theta[m1:,]) * F02 + theta[m1:,] * F12
    X_train = np.random.randn(ntrain, p) + np.random.uniform(-b, b, size=(ntrain, p))
    X_cal = np.random.randn(ncal, p) + np.random.uniform(-b, b, size=(ncal, p))
    X_mirror = np.random.randn(m, p) + np.random.uniform(-b, b, size=(m, p))
    Y1 = np.sqrt(a) * np.random.randn(int(n1_count), p) + np.random.uniform(-b, b, size=(int(n1_count), p))
    Y2 = np.sqrt(5*a) * np.random.randn(int(n1_count), p) + np.random.uniform(-b, b, size=(int(n1_count), p))
    Y = np.concatenate((Y1, Y2))

    return X_train, X_cal, X_test, X_mirror, Y, theta,pi


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
    cal_scoreb = clfb.predict_proba(X_cal)[:,0]
    test_scoreb = clfb.predict_proba(X_test)[:,0]
    mirror_scoreb = clfb.predict_proba(X_mirror)[:,0]

    scores = {
        'cp_testb': cp_vec(test_scoreb, cal_scoreb),
        'cp_mirrorb': cp_vec(mirror_scoreb, cal_scoreb),
    }

    return scores
def calculate_fdp_ntp(theta, R):
    return sum((1 - theta) * R) / max(sum(R), 1), sum(theta * R) / sum(theta)

def process_model_parallel(X_train, X_cal, X_test, X_mirror, BHT, theta, n_jobs=-1):

    model_scores = {}

    models = {
        'OneClassSVM_rbf': OneClassSVM(kernel='sigmoid'),
        'LOF': OneClassSVM(kernel='poly'),
    }

    m = X_test.shape[0]
    alpha = 0.05

    def process_single_model(model_name, model):
        clf = model
        clf.fit(X_train)

        test_score = clf.score_samples(X_test)
        mirror_score = clf.score_samples(X_mirror)
        cal_score = clf.score_samples(X_cal)

        cp_test = cp_vec(test_score, cal_score)
        cp_mirror = cp_vec(mirror_score, cal_score)
        cp_mins = np.minimum(cp_test, cp_mirror)

        lamdas = bh_func(cp_mins, bht)[1]
        indicator = cp_mins > lamdas
        tp = np.random.binomial(1, 0.5, size=np.sum(indicator))

        lamda = bh_func(cp_mins, BHT)[1]
        pi_hat = pis(cp_test, cp_mirror, lamda=lamda, h=bdw)
        w_hat = pi_hat / (1 / 2 - pi_hat)

        u_vec = cp_test / w_hat
        tu_vec = cp_mirror / w_hat

        u_min = np.minimum(u_vec, tu_vec)
        u_max = np.maximum(u_vec, tu_vec)

        u = u_min.copy()
        tu = tu_vec.copy()

        idx = np.where(indicator)[0]
        if len(idx) > 0:
            u[idx] = u_max[idx] * (1 - tp) + u_min[idx] * tp
            tu[idx] = u_max[idx] * tp + u_min[idx] * (1 - tp)

        WCP_value, R = WCP(u, tu, alpha), WCP(u_vec, tu_vec, alpha)
        sf = np.sum(WCP_value)
        ntp = np.sum(theta * R) / np.sum(theta)

        print(f"Finished model: {model_name}")
        return model_name, (sf, R, ntp)

    results = Parallel(n_jobs=n_jobs)(delayed(process_single_model)(name, model) for name, model in models.items())

    for name, score in results:
        model_scores[name] = score

    return model_scores
def process_model_two_parallel(X_train, X_cal, X_test, X_mirror, Y, BHT, theta, n_jobs=-1):

    model_scores = {}

    models = {
        'KNN': KNeighborsClassifier(),
        'MLP': MLPClassifier(max_iter=500),
    }

    m = X_test.shape[0]
    bdw = 50
    alpha = 0.05

    def process_single_model(model_name, clf):
        scores = compute_scores2(clf, X_train, X_cal, X_test, X_mirror, Y)

        cp_test = scores['cp_testb']
        cp_mirror = scores['cp_mirrorb']

        cp_mins = np.minimum(cp_test, cp_mirror)
        lamdas = bh_func(cp_mins, bht)[1]
        indicator = cp_mins > lamdas
        tp = np.random.binomial(1, 0.5, size=np.sum(indicator))

        lamda = bh_func(cp_mins, BHT)[1]
        pi_hat = pis(cp_test, cp_mirror, lamda=lamda, h=bdw)
        w_hat = pi_hat / (1 / 2 - pi_hat)

        u_vec = cp_test / w_hat
        tu_vec = cp_mirror / w_hat

        u_min = np.minimum(u_vec, tu_vec)
        u_max = np.maximum(u_vec, tu_vec)

        u = u_min.copy()
        tu = tu_vec.copy()

        idx = np.where(indicator)[0]
        if len(idx) > 0:
            u[idx] = u_max[idx] * (1 - tp) + u_min[idx] * tp
            tu[idx] = u_max[idx] * tp + u_min[idx] * (1 - tp)

        WCP_value, R = WCP(u, tu, alpha), WCP(u_vec, tu_vec, alpha)
        sf = np.sum(WCP_value)
        ntp = np.sum(theta * R) / np.sum(theta)

        print(f"Finished model: {model_name}")
        return model_name, (sf, R, ntp)

    results = Parallel(n_jobs=n_jobs)(delayed(process_single_model)(name, model) for name, model in models.items())

    for name, score in results:
        model_scores[name] = score

    return model_scores
def process_single_rep(i, j):

    X_train, X_cal, X_test, X_mirror, Y, theta, pi = generate_data(
        m1, pi1, m2, pi2, ntrain, ncal, int(n1_vec[i]), p, a,b
    )
    theta = theta.reshape((m,))

    fdp_results = {}
    ntp_results = {}


    resultso = process_model_parallel(X_train, X_cal, X_test, X_mirror, BHT, theta)
    sfo_max_idx = np.argmax([res[0] for res in resultso.values()])
    resultsb = process_model_two_parallel(X_train, X_cal, X_test, X_mirror, Y, BHT, theta)
    sfb_max_idx = np.argmax([res[0] for res in resultsb.values()])

    if np.max([res[0] for res in resultso.values()]) >= np.max([res[0] for res in resultsb.values()]):
        R_PGRAMS = list(resultso.values())[sfo_max_idx][1]
    else:
        R_PGRAMS = list(resultsb.values())[sfb_max_idx][1]

    fdp_results['11'], ntp_results['11'] = calculate_fdp_ntp(theta, list(resultso.values())[0][1])
    fdp_results['21'], ntp_results['21'] = calculate_fdp_ntp(theta, list(resultso.values())[1][1])
    fdp_results['1'], ntp_results['1'] = calculate_fdp_ntp(theta, list(resultsb.values())[0][1])
    fdp_results['2'], ntp_results['2'] = calculate_fdp_ntp(theta, list(resultsb.values())[1][1])

    fdp_results['AMS'], ntp_results['AMS'] = calculate_fdp_ntp(theta, R_PGRAMS)


    print(f'rep {j + 1} for variable {i + 1}')
    return i, j, fdp_results, ntp_results
