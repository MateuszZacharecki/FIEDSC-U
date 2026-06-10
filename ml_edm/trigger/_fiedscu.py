import numpy as np
from joblib import Parallel, delayed
from numba import njit

from ml_edm.trigger._edsc import EDSC


@njit(fastmath=True)
def _dist(sub1, sub2):

    if len(sub1) >= len(sub2):
        submax = sub1
        submin = sub2
    else:
        submax = sub2
        submin = sub1
    L = len(submin)
    n_windows = len(submax) - L + 1
    dists = np.empty(n_windows, dtype=np.float64)

    for k in range(n_windows):
        dist = 0
    
        for i in range(L):
            dist += ((submax[k+i] - submin[i])**2)
        
        dists[k] = np.sqrt(dist / L)
    
    return dists

@njit(fastmath=True)
def _Tdist(sub1, sub2, lambd):

    if len(sub1) >= len(sub2):
        submax = sub1
        submin = sub2
    else:
        submax = sub2
        submin = sub1
    L = len(submin)
    n_windows = len(submax) - L + 1
    dists = np.empty(n_windows, dtype=np.float64)

    for k in range(n_windows):
        dist = (submax[k] - submin[0])**2
        flag = np.sign(submax[k] - submin[0])
        
        for i in range(1, L):
            diff = submax[k+i] - submin[i]
            current_flag = np.sign(diff)
            if current_flag == flag or current_flag == 0 or flag == 0:
                dist += diff**2
            else:
                dist += (diff**2) * lambd
            flag = current_flag if current_flag != 0 else flag

        dists[k] = np.sqrt(dist / L)
            
    return dists

@njit(fastmath=True)
def _deltadist(sub1, sub2):
    
    if len(sub1) >= len(sub2):
        submax = sub1
        submin = sub2
    else:
        submax = sub2
        submin = sub1
    L = len(submin)
    n_windows = len(submax) - L + 1
    dists = np.empty(n_windows, dtype=np.float64)

    for k in range(n_windows):
        dist = 0
    
        for i in range(1, L):
            deriv1 = submax[k+i] - submax[k+i-1]
            deriv2 = submin[i] - submin[i-1]
            diff1 = abs(submax[k+i] - submin[i])
            diff2 = abs(submax[k+i-1] - submin[i-1])
            diff = (diff1 + diff2) / 2
            dist += (diff**2) * (1.0 + abs(deriv1 - deriv2))
        dists[k] = np.sqrt(dist / (L-1))
    
    return dists


class FIEDSCU(EDSC):
    def __init__(self,
                 min_length,
                 max_length,
                 uncertainty_threshold=0.3,
                 threshold_learning='kde',
                 prob_threshold=.95,
                 bound_threshold=3,
                 alpha=3,
                 min_coverage=1.,
                 n_jobs=1,
                 lambd=1.1,
                 step_size=8,
                 gamma=2,
                 eta=5,
                 metrics='deltadist',
                 sample=True):
        
        super().__init__(min_length, max_length, threshold_learning, prob_threshold,
                         bound_threshold, alpha, min_coverage, n_jobs)
        
        ######Constant attributes#######
        self.require_classifiers = False
        self.require_past_probas = False
        ################################

        self.lambd = lambd
        self.step_size = step_size
        self.gamma = gamma
        self.eta = eta
        self.uncertainty_threshold = uncertainty_threshold
        self._user_max_length = max_length
        self.metrics = metrics
        self.sample = sample

    def _sample_time_series(self, X, y):

        # sampled_X = []
        # sampled_y = []
        sampled_indices_map = []

        classes = np.unique(y)
        
        for c in classes:
            class_idx = np.where(y == c)[0]
            X_c = X[class_idx]
            
            # select criteria time series
            sums = np.sum(X_c, axis=1)
            mean_sum = np.mean(sums)
            
            criteria_idx = np.argmin(np.abs(sums - mean_sum))
            T_c = X_c[criteria_idx]
            
            # split subclass
            # dists_to_Tc = np.linalg.norm(X_c - T_c, axis=1)
            if self.metrics == 'dist':
                dists_to_Tc = [_dist(X_c[i], T_c)[0]
                            for i in range(X_c.shape[0])]
            elif self.metrics == 'Tdist':
                dists_to_Tc = [_Tdist(X_c[i], T_c, self.lambd)[0]
                            for i in range(X_c.shape[0])]
            elif self.metrics == 'deltadist':
                dists_to_Tc = [_deltadist(X_c[i], T_c)[0]
                            for i in range(X_c.shape[0])]
            
            sorted_args = np.argsort(dists_to_Tc)
            sorted_dists = np.array(dists_to_Tc)[sorted_args]
            
            discrepancies = np.diff(sorted_dists)
            
            if len(discrepancies) == 0:
                # sampled_X.append(X_c[0])
                # sampled_y.append(c)
                sampled_indices_map.append(class_idx[0])
                continue
            
            std_dev = np.std(discrepancies)
            threshold = std_dev / 2.0
            
            subclasses = []
            current_subclass = [sorted_args[0]]
            
            for i, disc in enumerate(discrepancies):
                if disc > threshold:
                    subclasses.append(current_subclass)
                    current_subclass = [sorted_args[i + 1]]
                else:
                    current_subclass.append(sorted_args[i + 1])
            subclasses.append(current_subclass)
            
            # sample time series
            for subclass in subclasses:
                if len(subclass) == 1:
                    selected_local_idx = subclass[0]
                else:
                    min_sum_dist = float('inf')
                    selected_local_idx = subclass[0]
                    
                    for idx1 in subclass:
                        current_sum_dist = 0.0
                        for idx2 in subclass:
                            if idx1 != idx2:
                                if self.metrics == 'dist':
                                    current_sum_dist += _dist(X_c[idx1], X_c[idx2])[0]
                                elif self.metrics == 'Tdist':
                                    current_sum_dist += _Tdist(X_c[idx1], X_c[idx2], self.lambd)[0]
                                elif self.metrics == 'deltadist':
                                    current_sum_dist += _deltadist(X_c[idx1], X_c[idx2])[0]
                                
                        if current_sum_dist < min_sum_dist:
                            min_sum_dist = current_sum_dist
                            selected_local_idx = idx1
                            
                # sampled_X.append(X_c[selected_local_idx])
                # sampled_y.append(c)
                sampled_indices_map.append(class_idx[selected_local_idx])

        return np.array(sampled_indices_map)

    def _get_utility_trend(self, X, y, shapelet, bmd_list, serie_idx):

        y = np.delete(y, serie_idx)
        bmd_list_tmp = bmd_list

        eml_list = []
        for i, ts in enumerate(X):
            
            if i == serie_idx: # if considered shapelet comes from this serie
                bmd_list_tmp = np.insert(bmd_list, i, 0.0)
                continue

            if shapelet[1] >= bmd_list_tmp[i]:
                if self.metrics == 'dist':
                    dists = _dist(X[i], shapelet[0])
                elif self.metrics == 'Tdist':
                    dists = _Tdist(X[i], shapelet[0], self.lambd)
                elif self.metrics == 'deltadist':
                    dists = _deltadist(X[i], shapelet[0])
                eml = np.where(np.array(dists) <= shapelet[1])[0]
            else:
                eml = np.array([])

            eml = eml[0] if len(eml) > 0 else np.inf
            eml_list.append(eml+len(shapelet[0]))

        class_mask = (y == shapelet[2])
        wrecall = np.power(eml_list, (-1/self.alpha))[class_mask].mean()
        #wrecall = np.power(eml_list, (-1/self.alpha)).sum() / (~class_mask).sum()

        thresh_mask = (np.array(bmd_list) <= shapelet[1])
        match_mask = (y[thresh_mask] == shapelet[2]) 
        precision = np.mean(match_mask) if len(match_mask) > 0 else 0

        # utility = 2 * precision * wrecall / (precision + wrecall)
        if (precision + wrecall) == 0:
            utility = 0
        else:
            utility = 2 * precision * wrecall / (precision + wrecall)
        coverage = thresh_mask & class_mask

        return np.nan_to_num(utility), np.insert(coverage, serie_idx, True), precision

    def _evaluate_candidate(self, X, y, candidate):
        
        serie_idx, start_pos, length = candidate
        
        bmd_list = []
        for i, ts in enumerate(X):
            if i == serie_idx:
                continue
            if self.metrics == 'dist':
                dists = _dist(X[i], X[serie_idx][start_pos:start_pos+length])
            elif self.metrics == 'Tdist':
                dists = _Tdist(X[i], X[serie_idx][start_pos:start_pos+length], self.lambd)
            elif self.metrics == 'deltadist':
                dists = _deltadist(X[i], X[serie_idx][start_pos:start_pos+length])
            bmd_list.append(np.min(dists))
        
        bmd_list = np.array(bmd_list)

        if self.threshold_learning == 'kde':
            p = self._kde_threshold(bmd_list, y, serie_idx)
        elif self.threshold_learning == 'che':
            p = self._che_threshold(bmd_list, y, serie_idx)
            
        if p[0] > 0:
            filtered_bmd = bmd_list[bmd_list <= p[0]]
            if len(filtered_bmd) > 0:
                sigma = np.std(filtered_bmd)
            else:
                sigma = 0.0
            feature = (X[serie_idx][start_pos:start_pos+length], p[0], p[1], sigma)
            # self.subsequences[length][i,j,:]
            utility, coverage, precision = self._get_utility_trend(X, y, feature, bmd_list, serie_idx)
            metadata = (serie_idx, start_pos, length)
            return (feature, coverage, utility, metadata, precision)
        
        return None

    def _selection_shapelet(self, shapelets, n_samples):

        selected = []
        covered = np.zeros(n_samples, dtype=bool)
        for shapelet in shapelets:
            coverage = shapelet[1]
            if (covered | coverage).sum() > covered.sum():
                selected.append(shapelet)
                covered = covered | coverage
            if covered.all():
                break
        
        return selected

    def _has_extended_self_similarity(self, metadata, selected_meta_list):

        serie_idx1, start_pos1, length1 = metadata
        for (serie_idx2, start_pos2, length2) in selected_meta_list:
            if serie_idx1 == serie_idx2 and abs(start_pos1 - start_pos2) <= self.gamma and abs(length1 - length2) <= self.eta:
                return True
        return False

    def _fit(self, X, y):
        self.max_length = self._user_max_length
        length_arr = list(range(self.min_length, self.max_length + 1, self.step_size))
        if self.max_length not in length_arr:
            length_arr.append(self.max_length)

        if self.sample:
            print("Sampling time series...")
            sampled_indices_map = self._sample_time_series(X, y)
        else:
            sampled_indices_map = [i for i in range(X.shape[0])]

        print("Learning shapelets (Phase 1)...")
        
        candidates1 = []
        for i in sampled_indices_map:
            for l in length_arr:
                for j in range(len(X[i]) - l + 1):
                    candidates1.append((i, j, l))

        results1 = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_candidate)(X, y, candidate) for candidate in candidates1
        )
        shapelets1 = [result for result in results1 if result is not None]
        
        shapelets1.sort(key=lambda x: x[2], reverse=True)
        better_shapelets = self._selection_shapelet(shapelets1, len(X))
        
        start_positions = set((meta[0], meta[1]) for _, _, _, meta, _ in better_shapelets)

        print("Learning shapelets (Phase 2)...")

        candidates2 = []
        for (serie_idx, start_pos) in start_positions:
            for l in range(self.min_length, self.max_length + 1):
                if l not in length_arr and start_pos + l <= len(X[serie_idx]):
                    candidates2.append((serie_idx, start_pos, l))

        results2 = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_candidate)(X, y, candidate) for candidate in candidates2
        )
        shapelets2 = [result for result in results2 if result is not None]

        self.shapelets = shapelets1 + shapelets2
        self.shapelets.sort(key=lambda x: x[2], reverse=True)
        
        self.features = []

        current_cov = np.zeros(len(X), dtype=bool)

        print(f"Shapelet Pruning among {len(self.shapelets)} shapelets...")
        
        i = 0
        while i < len(self.shapelets):
            batch = [self.shapelets[i]]
            current_utility = self.shapelets[i][2]
            
            j = i + 1
            while j < len(self.shapelets) and self.shapelets[j][2] == current_utility:
                batch.append(self.shapelets[j])
                j += 1
            
            batch_improves_coverage = False
            
            batch_coverage = np.zeros(len(X), dtype=bool)
            for s in batch:
                batch_coverage = batch_coverage | s[1]
                
            if (current_cov | batch_coverage).sum() > current_cov.sum():
                 batch_improves_coverage = True
            
            if batch_improves_coverage:
                for s in batch:
                    actual_subsequence = s[0][0]
                    thresh = s[0][1]
                    target = s[0][2]
                    sigma = s[0][3]
                    precision = s[4]
                    
                    self.features.append((actual_subsequence, thresh, target, sigma, precision))
                    
                    current_cov = current_cov | s[1]
            
            if current_cov.mean() >= self.min_coverage:
                break
                
            i = j

        return self

    def _predict(self, X, X_timestamps=None):
        
        n_samples = X.shape[0]
        all_preds, all_triggers, all_t_star, all_uncerts = (np.full((len(X),), np.nan), 
                                                            np.zeros((len(X),), dtype=bool), 
                                                            np.full((len(X),), np.nan),
                                                            np.full((len(X),), np.nan),)
        locked_until = np.zeros((n_samples, len(self.features)), dtype=np.int32)
        if self.features:
            min_L = min([len(fts[0]) for fts in self.features])
            target = [fts[2] for fts in self.features]
            classes = np.unique(target)
            class_to_idx = {cls: i for i, cls in enumerate(classes)}
            target_indices = np.array([class_to_idx[t] for t in target])

            accumulated_uncerts = np.ones((n_samples, len(classes)), dtype=np.float64)

            for l in range(min_L, X.shape[1]):     
                for idx, ts in enumerate(X[:, :l]):
                    
                    if all_triggers[idx]:
                        continue
                    
                    for j, fts in enumerate(self.features):
                        sub, threshold, target, sigma, precision = fts
                        
                        if l < locked_until[idx, j]:
                            continue
                            
                        if len(sub) > len(ts):
                            continue
                        
                        if self.metrics == 'dist':
                            dist = _dist(ts[-len(sub):], sub)[0]
                        elif self.metrics == 'Tdist':
                            dist = _Tdist(ts[-len(sub):], sub, self.lambd)[0]
                        elif self.metrics == 'deltadist':
                            dist = _deltadist(ts[-len(sub):], sub)[0]
                        
                        if dist <= threshold:
                            numerator = (threshold - dist) ** 2
                            denominator = (sigma ** 2) + numerator
                            confidence = precision * (numerator / denominator)
                            accumulated_uncerts[idx, target_indices[j]] *= (1.0 - confidence)
                            
                            locked_until[idx, j] = l + int(len(sub) / 2)

                    min_uncertainty = 2.0
                    best_class = -1
                    
                    for c_idx in range(len(classes)):
                        if accumulated_uncerts[idx, c_idx] < min_uncertainty:
                            min_uncertainty = accumulated_uncerts[idx, c_idx]
                            best_class = classes[c_idx]
                    
                    if min_uncertainty <= self.uncertainty_threshold:
                        all_preds[idx] = best_class
                        all_triggers[idx] = True
                        all_t_star[idx] = l
                        all_uncerts[idx] = min_uncertainty

                    else:
                        all_preds[idx] = best_class
                        all_uncerts[idx] = min_uncertainty
                        
                if all_triggers.all():
                    break

        return all_preds, all_triggers, np.nan_to_num(all_t_star, nan=X.shape[1]), all_uncerts