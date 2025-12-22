
def PTAMS(u_vec,tu_vec,cp_mins, bht):
    m = len(u_vec)
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
def process_model(X_train, X_cal, X_test, X_mirror, BHT, theta, m, S_vec):
    model_scores = {}
    models = {
        'OneClassSVM_poly': OneClassSVM(kernel='poly'),
        'OneClassSVM_rbf': OneClassSVM(kernel='rbf'),
         'OneClassSVM_sigmoid': OneClassSVM(kernel='sigmoid'),
         'IsolationForest': IsolationForest(),
        'LocalOutlierFactor': LocalOutlierFactor(novelty=True),
        'GaussianMixture':GaussianMixture(n_components=2, covariance_type="full")
    }

    for model_name, model in models.items():
        clf = model

        clf.fit(X_train)
        test_score = clf.score_samples(X_test)
        mirror_score = clf.score_samples(X_mirror)
        cal_score = clf.score_samples(X_cal)

        cp_test = cp_vec(test_score, cal_score)
        cp_mirror = cp_vec(mirror_score, cal_score)

        cp_mins = np.minimum(cp_test,cp_mirror)
        lamdas = bh_func(cp_mins, bht)[1]
        tp = np.random.binomial(1, 0.5, sum(cp_mins > lamdas))

        lamda = bh_func(np.minimum(cp_test, cp_mirror), BHT)[1]
        pi_hat = estimate_pi_hat(S_vec, cp_test, cp_mirror, lamda=lamda)
        w_hat = pi_hat / (1 / 2 - pi_hat)
        u_vec = cp_test / w_hat
        tu_vec = cp_mirror / w_hat

        u_min = np.minimum(u_vec, tu_vec)
        u_max = np.maximum(u_vec, tu_vec)

        u = np.zeros(m)
        tu = np.zeros(m)

        tu[cp_mins > lamdas] = u_max[cp_mins > lamdas] * tp + u_min[cp_mins > lamdas] * (1 - tp)
        tu[cp_mins <= lamdas] = np.maximum(u_vec[cp_mins <= lamdas], tu_vec[cp_mins <= lamdas])

        u[cp_mins > lamdas] = u_max[cp_mins > lamdas] * (1 - tp) + u_min[cp_mins > lamdas] * tp
        u[cp_mins <= lamdas] = np.minimum(u_vec[cp_mins <= lamdas], tu_vec[cp_mins <= lamdas])
        sf = sum(WCP(u, tu, alpha))


        model_scores[model_name] = sf

    sf_max_idx = np.argmax([res for res in model_scores.values()])
    model_select = list(models.values())[sf_max_idx]

    return model_select
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
def process_model_two(X_train, X_cal, X_test, X_mirror, Y,BHT, theta, m):
    model_scores = {}

    models = {
        'RF': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVC': SVC(probability=True),
        'NB': GaussianNB(),
        'QDA': QuadraticDiscriminantAnalysis(),
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

        lamda = bh_func(np.minimum(cp_test,cp_mirror), BHT)[1]

        pi_hat = pis(cp_test, cp_mirror, lamda=lamda, h=bdw)
        w_hat = pi_hat / (1 / 2 - pi_hat)

        u_vec = cp_test / w_hat
        tu_vec = cp_mirror / w_hat

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

        model_scores[model_name] = (sf, R, ntp)

    return model_scores
def calculate_fdp_ntp(theta, R):
    return sum((1 - theta) * R) / max(sum(R), 1), sum(theta * R)
def add_hour_side_info(df, timestamp_col=' Timestamp', label_col=' Label'):

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

    df['hour_of_day'] = df[timestamp_col].dt.hour

    hourly_stats = df.groupby(['hour_of_day', label_col]).size().unstack(fill_value=0)


    return df, df['hour_of_day'].to_numpy()
def data_sample(inliers, outliers, n_inlier, n_test_inliers=2500):

    test_inliers = inliers.sample(n=n_test_inliers).sort_index()
    test_set = pd.concat([test_inliers, outliers]).sort_index()
    test_set = test_set.sample(frac=1).sort_index()
    theta = (test_set[label_col]).to_numpy()
    time_col = ' Timestamp'
    test_set, S_vec = add_hour_side_info(test_set, timestamp_col= time_col, label_col=' Label')
    X_test = test_set.drop(columns=[label_col, time_col]).to_numpy()
    m = len(X_test)
    n_train_cal = n_inlier - m
    train_inliers = inliers.drop(test_inliers.index).sample(n=n_train_cal + m).sort_index()
    X_train = train_inliers.iloc[:int(n_train_cal/2)]
    X_cal = train_inliers.iloc[int(n_train_cal/2):int(n_train_cal)]
    X_mirror = train_inliers.iloc[-m:]
    feature_cols = list(
        set(X_train.columns) & set(X_cal.columns) &
        set(X_mirror.columns) & set(test_set.columns)
    )
    feature_cols = [col for col in feature_cols if col not in [label_col, time_col]]
    feature_cols = sorted(feature_cols)

    X_train = X_train[feature_cols].to_numpy()
    X_cal = X_cal[feature_cols].to_numpy()
    X_mirror = X_mirror[feature_cols].to_numpy()
    X_test = test_set[feature_cols].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_cal = scaler.transform(X_cal)
    X_mirror = scaler.transform(X_mirror)
    X_test = scaler.transform(X_test)

    return X_train, X_cal, X_mirror, X_test, theta, S_vec
def process_single_rep(i,j):
    X_train, X_cal, X_mirror, X_test, theta, S_vec = data_sample(inliers, test_outliers, n_inlier,n_test_inliers= int(n_test_inliers_vec1[i]))
    m = len(X_test)
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
    Cal_scoreo = clfo.score_samples(X_Cal)
    Cp_testo = cp_vec(test_scoreo, Cal_scoreo)
    hat_pi_0 = storey_pi(Cp_testo, lamda=0.5)
    R_CPBH = multipletests(Cp_testo, alpha/hat_pi_0, method='fdr_bh')[0]
    fdp_results['CPBHSP'], ntp_results['CPBHSP'] = calculate_fdp_ntp(theta, R_CPBH)

    lamdao = bh_func(np.minimum(cp_testo, cp_mirroro), BHT)[1]
    pi_hato = estimate_pi_hat(S_vec, cp_testo, cp_mirroro, lamda=lamdao)
    w_hato = pi_hato / (1 / 2 - pi_hato)
    u_veco = np.minimum(1, cp_testo / w_hato)
    tu_veco = np.minimum(1, cp_mirroro / w_hato)
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
    pi_hato_kde = estimate_pi_hat(S_vec, ptest, pmirror, lamda=lamdao)
    w_hato = pi_hato_kde / (1 / 2 - pi_hato_kde)
    u_vec_scq_kde = ptest / w_hato
    tu_vec_scq_kde = pmirror / w_hato
    R_SCQO = WCP(u_vec_scq_kde, tu_vec_scq_kde, alpha)
    fdp_results['SCQPUKDE'], ntp_results['SCQPUKDE'] = calculate_fdp_ntp(theta, R_SCQO)

    x = X_test
    xnull = np.vstack((X_train, X_cal))
    proc = AdaDetectERM(scoring_fn=RandomForestClassifier(max_depth=10),
                        split_size=(len(X_train)) / (len(xnull)))
    proc.fit(x, alpha, xnull)
    s_test1 = proc.test_statistics
    s_cal1 = proc.null_statistics
    p_d_test = cp_vec(s_test1, s_cal1)
    x = X_mirror
    xnull = np.vstack((X_train, X_cal))
    proc = AdaDetectERM(scoring_fn=RandomForestClassifier(max_depth=10),
                        split_size=(len(X_train)) / (len(xnull)))
    proc.fit(x, alpha, xnull)
    s_test2 = proc.test_statistics
    s_cal2 = proc.null_statistics
    p_d_mirror = cp_vec(s_test2, s_cal2)
    lamdao = bh_func(np.minimum(p_d_test, p_d_mirror), BHT)[1]
    pi_hato_purf = estimate_pi_hat(S_vec, p_d_test, p_d_mirror, lamda=lamdao)
    w_hato = pi_hato_purf / (1 / 2 - pi_hato_purf)
    u_vec_scq_pu = p_d_test / w_hato
    tu_vec_scq_pu = p_d_mirror / w_hato
    R_SCQPURF = WCP(u_vec_scq_pu, tu_vec_scq_pu, alpha)
    fdp_results['SCQPURF'], ntp_results['SCQPURF'] = calculate_fdp_ntp(theta, R_SCQPURF)

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
    AMS_result['SCQ-OCC'] = PTAMS(u_veco, tu_veco, np.minimum(cp_testo, cp_mirroro), bht)
    AMS_result['SCQ-PU'] = PTAMS(u_vec_scq_pu, tu_vec_scq_pu, np.minimum(p_d_test, p_d_mirror), bht)
    AMS_result['SCQ-KDE'] = PTAMS(u_vec_scq_kde, tu_vec_scq_kde, np.minimum(ptest, pmirror), bht)
    sf_max_idx = np.argmax([res[0] for res in AMS_result.values()])
    model_select = list(AMS_result.keys())[sf_max_idx]
    R_PTAMS = list(AMS_result.values())[sf_max_idx][1]
    print(model_select, "sf:", [res[0] for res in AMS_result.values()])
    print('ntp:', [res[2] for res in AMS_result.values()])

    fdp_results['SCQAMS'], ntp_results['SCQAMS'] = calculate_fdp_ntp(theta, R_PTAMS)

    print(f'rep {j + 1} for variable {i + 1}')
    return i, j, fdp_results, ntp_results
