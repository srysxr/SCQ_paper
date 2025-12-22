models_o = {
        'OneClassSVM_rbf': OneClassSVM(kernel='rbf'),
        'LOF': LocalOutlierFactor(novelty=True),
    }
models_b = {
        'QDA': QuadraticDiscriminantAnalysis(),
        'RF': RandomForestClassifier(),
    }

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
    Y = np.concatenate((Y1, Y2))

    return X_train, X_cal, X_test, X_mirror, Y, theta,pi
def calculate_fdp_ntp(theta, R):
    return sum((1 - theta) * R) / max(sum(R), 1), sum(theta * R) / sum(theta)
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
def process_model_parallel(X_train, X_cal, X_test, X_mirror, BHT, theta, n_jobs=-1):
    model_scores = {}

    models = {
        'OneClassSVM_rbf': OneClassSVM(kernel='rbf'),
        'LOF': LocalOutlierFactor(novelty=True),
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
        'QDA': QuadraticDiscriminantAnalysis(),
        'RF': RandomForestClassifier(),
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
def clone_model(model):
    from sklearn.base import clone
    return clone(model)
def prefit_models(X_train, Y_train, Y, models_o, models_b):

    occ_on_X = {name: clone_model(m).fit(X_train) for name, m in models_o.items()}
    occ_on_Y = {name: clone_model(m).fit(Y_train) for name, m in models_o.items()}
    sam = np.concatenate([X_train, Y_train], axis=0)
    label = np.concatenate([[0]*len(X_train), [1]*len(Y_train)])
    bic_on_XY = {name: clone_model(m).fit(sam, label) for name, m in models_b.items()}
    sam = np.concatenate([X_train, Y], axis=0)
    label = np.concatenate([[0] * len(X_train), [1] * len(Y)])
    bic_on_XYY = {name: clone_model(m).fit(sam, label) for name, m in models_b.items()}

    return occ_on_X, occ_on_Y, bic_on_XY, bic_on_XYY
def cp_fast(X_scores, cal_scores_sorted):
    idx = np.searchsorted(cal_scores_sorted, X_scores, side='right')
    return (idx + 1) / (len(cal_scores_sorted) + 1)
def cp_fast_scalar(x, cal_scores_sorted):
    return (np.searchsorted(cal_scores_sorted, x, side='right') + 1) / (len(cal_scores_sorted) + 1)
def u0_self_from_ranks(v):
    r = pd.Series(v).rank(method='max').to_numpy()
    return r / (len(v) + 1)
def insert_median_from_sorted(sorted_arr, val):
    n = len(sorted_arr)
    pos = np.searchsorted(sorted_arr, val, side='right')
    new_n = n + 1
    mid = new_n // 2

    if new_n % 2 == 1:
        if pos == mid:
            return val
        elif pos < mid:
            return sorted_arr[mid - 1]
        else:
            return sorted_arr[mid]
    else:
        left = mid - 1
        right = mid
        if pos <= left:
            return 0.5 * (sorted_arr[left - 1] + sorted_arr[left]) if left - 1 >= 0 else 0.5 * (sorted_arr[left] + val)
        elif pos >= right + 1:
            return 0.5 * (sorted_arr[left] + sorted_arr[right])
        else:
            if pos == right:
                return 0.5 * (sorted_arr[left] + val)
            else:
                return 0.5 * (val + sorted_arr[right - 1])
def median_after_insert_sorted(sorted_arr, val):
    n = len(sorted_arr)
    pos = np.searchsorted(sorted_arr, val, side='right')
    new = n + 1
    if new % 2 == 1:
        mid = new // 2
        if pos == mid:
            return val
        elif pos < mid:
            return sorted_arr[mid - 1]
        else:
            return sorted_arr[mid]
    else:
        left_idx = new // 2 - 1
        right_idx = new // 2
        def kth_elem(k):
            if k < pos:
                return sorted_arr[k]
            elif k == pos:
                return val
            else:
                return sorted_arr[k - 1]
        a = kth_elem(left_idx)
        b = kth_elem(right_idx)
        return 0.5 * (a + b)
def mdiff_occ_on_Y_fast(model_fitted_on_Y, X_cal_scores_on_Y_sorted, Y_cal_scores_on_Y, x_score):
    med_cal0 = median_after_insert_sorted(X_cal_scores_on_Y_sorted, x_score)
    med_y = np.median(Y_cal_scores_on_Y)
    return abs(med_cal0 - med_y)
def mdiff_occ_on_X_fast(sorted_X_cal_scores, Y_cal_scores_on_X, x_score):
    med_cal0 = median_after_insert_sorted(sorted_X_cal_scores, x_score)
    med_y = np.median(Y_cal_scores_on_X)
    return abs(med_cal0 - med_y)
def mdiff_bic_fast(sorted_X_cal_proba0, Y_cal_proba0, x_proba0):
    med_cal0 = median_after_insert_sorted(sorted_X_cal_proba0, x_proba0)
    med_y = np.median(Y_cal_proba0)
    return abs(med_cal0 - med_y)
def process_single_rep(i, j, inner_n_jobs=1):
    t_start_rep = time.perf_counter()

    X_train, X_cal, X_test, X_mirror, Y, theta, pi = generate_data(
        m1, pi1, m2, pi2, ntrain, ncal, int(n1_vec[i]), p, a, d
    )
    m = len(X_test)
    theta = theta.reshape((m,))

    X_cal = np.concatenate((X_cal, X_mirror), axis=0)

    np.random.shuffle(Y)
    Y_train = Y[:int(len(Y) / 2), :]
    Y_cal = Y[int(len(Y) / 2):, :]

    cp_fdp = {}
    cp_ntp = {}
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
    occ_on_X, occ_on_Y, bic_on_XY, bic_on_XYY = prefit_models(X_train, Y_train, Y, models_o, models_b)

    C1_scores_Ycal = {}
    C1_scores_Xcal_sorted = {}
    C1_scores_Xtest = {}
    for name, mdl in occ_on_Y.items():
        s_ycal = mdl.decision_function(Y_cal)
        s_xcal = mdl.decision_function(X_cal)
        s_xtest = mdl.decision_function(X_test)
        C1_scores_Ycal[name] = s_ycal
        C1_scores_Xcal_sorted[name] = np.sort(s_xcal)
        C1_scores_Xtest[name] = s_xtest

    C0_scores_Xcal_sorted = {}
    C0_scores_Ycal = {}
    C0_scores_Xtest = {}
    for name, mdl in occ_on_X.items():
        s_xcal = mdl.decision_function(X_cal)
        s_ycal = mdl.decision_function(Y_cal)
        s_xtest = mdl.decision_function(X_test)
        C0_scores_Xcal_sorted[name] = np.sort(s_xcal)
        C0_scores_Ycal[name] = s_ycal
        C0_scores_Xtest[name] = s_xtest

    BIC_proba_Xcal_sorted = {}
    BIC_proba_Ycal = {}
    BIC_proba_Xtest = {}
    for name, mdl in bic_on_XY.items():
        p_cal = mdl.predict_proba(X_cal)[:, 0]
        p_ycal = mdl.predict_proba(Y_cal)[:, 0]
        p_xtest = mdl.predict_proba(X_test)[:, 0]
        BIC_proba_Xcal_sorted[name] = np.sort(p_cal)
        BIC_proba_Ycal[name] = p_ycal
        BIC_proba_Xtest[name] = p_xtest

    BICYY_proba_Xcal = {}
    BICYY_proba_Xtest = {}
    for name, mdl in bic_on_XYY.items():
        p_cal = mdl.predict_proba(X_cal)[:, 0]
        p_xtest = mdl.predict_proba(X_test)[:, 0]
        BICYY_proba_Xcal[name] = p_cal
        BICYY_proba_Xtest[name] = p_xtest

    for name in list(models_o.keys()):
        cp_score = cp_vec(C0_scores_Xtest[name], C0_scores_Xcal_sorted[name])
        hat_pi_0 = storey_pi(cp_score, lamda=0.5)
        R_cp = multipletests(cp_score, alpha/hat_pi_0, method='fdr_bh')[0]
        cp_fdp[name], cp_ntp[name] = calculate_fdp_ntp(theta, R_cp)
    for name in list(models_b.keys()):
        cp_score = cp_vec(BICYY_proba_Xtest[name], BICYY_proba_Xcal[name])
        hat_pi_0 = storey_pi(cp_score, lamda=0.5)
        R_cp = multipletests(cp_score, alpha/hat_pi_0, method='fdr_bh')[0]
        cp_fdp[name], cp_ntp[name] = calculate_fdp_ntp(theta, R_cp)


    names_o = list(models_o.keys())
    names_b = list(models_b.keys())

    def process_one_test(l):
        Xtest = X_test[l]

        md_list = []
        for name in names_o:
            x_score = C1_scores_Xtest[name][l]
            md = mdiff_occ_on_Y_fast(occ_on_Y[name], C1_scores_Xcal_sorted[name], C1_scores_Ycal[name], x_score)
            md_list.append(md)
        name_C1 = names_o[int(np.argmax(md_list))]
        C1 = occ_on_Y[name_C1]

        md_list_o = []
        for name in names_o:
            x_score = C0_scores_Xtest[name][l]
            md = mdiff_occ_on_X_fast(C0_scores_Xcal_sorted[name], C0_scores_Ycal[name], x_score)
            md_list_o.append(md)
        idx_o = int(np.argmax(md_list_o))
        best_o_name = names_o[idx_o]
        best_o_md = md_list_o[idx_o]

        md_list_b = []
        for name in names_b:
            x_proba0 = BIC_proba_Xtest[name][l]
            md = mdiff_bic_fast(BIC_proba_Xcal_sorted[name], BIC_proba_Ycal[name], x_proba0)
            md_list_b.append(md)
        idx_b = int(np.argmax(md_list_b))
        best_b_name = names_b[idx_b]
        best_b_md = md_list_b[idx_b]

        if best_o_md >= best_b_md:
            s0_Xcal = C0_scores_Xcal_sorted[best_o_name]
            s0_x = C0_scores_Xtest[best_o_name][l]
            v0 = np.concatenate([s0_Xcal, [s0_x]])
            u0cal0_x = u0_self_from_ranks(v0)
        else:
            s0_Xcal = BIC_proba_Xcal_sorted[best_b_name]
            s0_x = BIC_proba_Xtest[best_b_name][l]
            v0 = np.concatenate([s0_Xcal, [s0_x]])
            u0cal0_x = u0_self_from_ranks(v0)

        s1cal1 = C1_scores_Ycal[name_C1]
        s1cal1_sorted = np.sort(s1cal1)
        s1_Xcal = None
        s1_x = C1_scores_Xtest[name_C1][l]
        u1_Xcal = cp_fast(C1_scores_Xcal_sorted[name_C1], s1cal1_sorted)
        u1_x = cp_fast_scalar(s1_x, s1cal1_sorted)
        u1cal0_x = np.concatenate([u1_Xcal, [u1_x]])

        rcal0_x =  u0cal0_x / u1cal0_x
        rx = rcal0_x[-1]
        rcal0 = rcal0_x[:-1]
        pval = float(cp_fast_scalar(rx, np.sort(rcal0)))
        return pval

    if inner_n_jobs == 1:
        u_list = [process_one_test(l) for l in range(m)]
    else:
        u_list = Parallel(n_jobs=inner_n_jobs)(
            delayed(process_one_test)(l) for l in range(m)
        )

    u_vec = np.asarray(u_list)
    hat_pi_0 = storey_pi(u_vec, lamda=0.5)
    R_ICP_AMS = multipletests(u_vec, alpha/hat_pi_0, method='fdr_bh')[0]
    cp_fdp['ICP-AMS'], cp_ntp['ICP-AMS'] = calculate_fdp_ntp(theta, R_ICP_AMS)

    t_end_rep = time.perf_counter()
    elapsed = t_end_rep - t_start_rep

    print(f"rep ({i},{j}) done, time: {elapsed:.3f}s")
    return i, j, cp_fdp, cp_ntp, fdp_results, ntp_results, elapsed

