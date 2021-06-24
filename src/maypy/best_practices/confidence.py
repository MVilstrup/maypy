import numpy as np

def _compute_cii(self, model, verbose=3):
    if verbose>=3: print("[distfit] >Compute confidence interval [%s]" %(self.method))
    CIIup, CIIdown = None, None

    if self.alpha is not None:
        if self.bound == 'up' or self.bound == 'both' or self.bound == 'right' or self.bound == 'high':
            # CIIdown = distr.ppf(1 - self.alpha, *arg, loc=loc, scale=scale) if arg else distr.ppf(1 - self.alpha, loc=loc, scale=scale)
            CIIdown = model['model'].ppf(1 - self.alpha)
        if self.bound == 'down' or self.bound == 'both' or self.bound == 'left' or self.bound == 'low':
            # CIIup = distr.ppf(self.alpha, *arg, loc=loc, scale=scale) if arg else distr.ppf(self.alpha, loc=loc, scale=scale)
            CIIup = model['model'].ppf(self.alpha)

    if (self.method=='parametric') or (self.method=='discrete'):
        # Separate parts of parameters
        # arg = model['params'][:-2]
        # loc = model['params'][-2]
        # scale = model['params'][-1]
        # dist = getattr(st, model['name'])
        # dist = model['distr']
        # Get fitted model
        # distr = model['model']

        # Determine %CII


    elif self.method=='quantile':
        X = model
        model = {}
        CIIdown = np.quantile(X, 1 - self.alpha)
        CIIup = np.quantile(X, self.alpha)
        # model['model'] = model
    elif self.method=='percentile':
        X = model
        model = {}
        # Set Confidence intervals
        # ps = np.array([np.random.permutation(len(X)) for i in range(self.n_perm)])
        # xp = X[ps[:, :10]]
        # yp = X[ps[:, 10:]]
        # samples = np.percentile(xp, 7, axis=1) - np.percentile(yp, 7, axis=1)
        cii_high = (0 + (self.alpha / 2)) * 100
        cii_low = (1 - (self.alpha / 2)) * 100
        CIIup = np.percentile(X, cii_high)
        CIIdown = np.percentile(X, cii_low)
        # Store
        # model['samples'] = samples
    else:
        raise Exception('[distfit] >Error: method parameter can only be of type: "parametric", "quantile", "percentile" or "discrete".')

    # Store
    model['CII_min_alpha'] = CIIup
    model['CII_max_alpha'] = CIIdown
    return(model)