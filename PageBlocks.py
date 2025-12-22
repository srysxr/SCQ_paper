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
def process_model(X_train, X_cal, X_test, X_mirror, BHT, alpha, S_vec, theta):
    m = len(X_test)
    model_scores = {}
    models = {
        'LocalOutlierFactor': LocalOutlierFactor(novelty=True),
        'GaussianMixture':GaussianMixture()
    }

    for model_name, model in models.items():
        clf = model

        clf.fit(X_train)
        test_score = clf.score_samples(X_test)
        mirror_score = clf.score_samples(X_mirror)
        cal_score = clf.score_samples(X_cal)

        cp_test = cp_vec(test_score, cal_score)
        cp_mirror = cp_vec(mirror_score, cal_score)

        cp_mins = np.minimum(cp_test, cp_mirror)
        lamdas = bh_func(cp_mins, bht)[1]
        tp = np.random.binomial(1, 0.5, sum(cp_mins > lamdas))

        lamda = bh_func(np.minimum(cp_test, cp_mirror), BHT)[1]
        pi_hat = estimate_pi_hat(S_vec, cp_test, cp_mirror, lamda)
        w_hat = pi_hat / (1/2 - pi_hat)
        u_vec = np.minimum(1, cp_test / w_hat)
        tu_vec = np.minimum(1, cp_mirror / w_hat)

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

        ntp = sum(theta * R) / sum(theta)
        fdr = sum((1 - theta) * R) / max(sum(R), 1)

        model_scores[model_name] = (sf, R, ntp, fdr)

    return model_scores
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
def process_model_two(X_train, X_cal, X_test, X_mirror, Y, BHT, alpha, S_vec, theta):
    m = len(X_test)
    model_scores = {}
    models = {
        'KNN': KNeighborsClassifier(),
        'MLP': MLPClassifier(max_iter=500)
    }
    for model_name, model in models.items():
        clf = model
        scores = compute_scores2(clf, X_train, X_cal, X_test, X_mirror, Y)

        cp_test = scores['cp_testb']
        cp_mirror = scores['cp_mirrorb']

        cp_mins = np.minimum(cp_test, cp_mirror)
        lamdas = bh_func(cp_mins, bht)[1]
        tp = np.random.binomial(1, 0.5, sum(cp_mins > lamdas))

        lamda = bh_func(np.minimum(cp_test, cp_mirror), BHT)[1]

        pi_hat = estimate_pi_hat(S_vec, cp_test, cp_mirror, lamda)
        w_hat = pi_hat / (1/2 - pi_hat)

        u_vec = np.minimum(1, cp_test / w_hat)
        tu_vec = np.minimum(1, cp_mirror / w_hat)

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

        ntp = sum(theta * R) / sum(theta)
        fdr = sum((1 - theta) * R) / max(sum(R), 1)

        model_scores[model_name] = (sf, R, ntp, fdr)

    return model_scores


def process_single_rep(i, j):
        fdp_results = {}
        ntp_results = {}

        np.random.shuffle(X_in)
        np.random.shuffle(XX_out)

        n_in = np.shape(X_in)[0]
        n_out = np.shape(XX_out)[0]

        X_out_train = XX_out[-int(n_out_test_vec[i]):]
        X_out_train = np.delete(X_out_train, [-1, -2], axis=1)
        X_out = XX_out[:150]

        n_go1 = 120
        n_go2 = 25
        n_go3 = 5
        n_gi1 = int(n_in_test/3)
        n_gi2 = int(n_in_test/3)
        n_gi3 = int(n_in_test/3)


        X_test_in = X_in[:n_in_test, :]
        n_test = X_test_in.shape[0] + X_out.shape[0]
        X_null = X_in[n_in_test:, :]
        X_null = np.delete(X_null, [-1, -2], axis=1)
        n_null = X_null.shape[0]
        X_mirror = X_null[:n_test, :]
        X_train = X_null[n_test:(n_test + int((n_null - n_test) / 2)), :]
        X_cal = X_null[(n_test + int((n_null - n_test) / 2)):, :]

        X_test1 = np.vstack((X_out[:n_go1, :],
                             X_test_in[:n_gi1, :]))
        X_test2 = np.vstack((X_out[n_go1:(n_go1 + n_go2), :],
                             X_test_in[n_gi1:(n_gi1 + n_gi2), :]))
        X_test3 = np.vstack((X_out[(n_go1 + n_go2):, :],
                             X_test_in[(n_gi1 + n_gi2):, :]))

        np.random.shuffle(X_test1)
        np.random.shuffle(X_test2)
        np.random.shuffle(X_test3)

        X_test = np.vstack((X_test1, X_test2, X_test3))

        len1 = X_test1.shape[0]
        len2 = X_test2.shape[0]
        len3 = X_test3.shape[0]

        S_vec = np.concatenate([
            np.full(len1, 1),
            np.full(len2, 2),
            np.full(len3, 3)
        ])

        theta = np.zeros(n_test)
        theta[X_test[:, -1] == b'yes'] = 1

        X_test = np.delete(X_test, [-1, -2], axis=1)


        resultso = process_model(X_train, X_cal, X_test, X_mirror, BHT, alpha, S_vec, theta)
        resultsb = process_model_two(X_train, X_cal, X_test, X_mirror, X_out_train, BHT, alpha, S_vec, theta)

        sf_max_idxo = np.argmax([res[0] for res in resultso.values()])
        sf_max_idxb = np.argmax([res[0] for res in resultsb.values()])

        if list(resultso.values())[sf_max_idxo][0] >= list(resultsb.values())[sf_max_idxb][0]:
            ntp_results['AMS'] = sum(list(resultso.values())[sf_max_idxo][1] * theta)
            fdp_results['AMS'] = list(resultso.values())[sf_max_idxo][-1]
        else:
            ntp_results['AMS'] = sum(list(resultsb.values())[sf_max_idxb][1] * theta)
            fdp_results['AMS'] = list(resultsb.values())[sf_max_idxb][-1]

        ntp_results['C1'] = sum(list(resultso.values())[0][1] * theta)
        ntp_results['C2'] = sum(list(resultso.values())[1][1] * theta)
        ntp_results['C3'] = sum(list(resultsb.values())[0][1] * theta)
        ntp_results['C4'] = sum(list(resultsb.values())[1][1] * theta)

        fdp_results['C1'] = list(resultso.values())[0][-1]
        fdp_results['C2'] = list(resultso.values())[1][-1]
        fdp_results['C3'] = list(resultsb.values())[0][-1]
        fdp_results['C4'] = list(resultsb.values())[1][-1]

        print(f'rep {j + 1} for variable {i + 1}')
        return i, j, fdp_results, ntp_results
