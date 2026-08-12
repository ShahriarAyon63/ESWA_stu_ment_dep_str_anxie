"""
Complexity-Aware Genetic Algorithm–Optimized Logistic Regression (CA-GA-LR)
============================================================================
Reference: "Explainable Genetic Algorithm-Optimized Machine Learning Framework 
for Digital Screening of Depression, Anxiety, and Stress Among University Students"

Methodology mapping to paper sections:
    - Sec 3.5.2 : Genetic Encoding with Reduced Search Space  -> _init_chromosome()
    - Sec 3.5.3 : Fitness Function with Early Pruning          -> _evaluate_fitness()
    - Sec 3.5.3 : Adaptive threshold τ_g = μ_g − σ_g          -> _prune_population()
    - Sec 3.5.4 : Lightweight Evolutionary Operators          -> _crossover(), _mutate()
    - Sec 3.5.5 : Per-Generation Computational Burden         -> mini_val_size=0.30
    - Sec 3.8   : DEAP-style GA pipeline                      -> fit() loop

Author: Generated for Q1 journal revision (Reviewer #1 ethical concerns addressed 
in separate text insertions; this module provides the core algorithmic engine).
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.exceptions import ConvergenceWarning


class CAGALRClassifier:
    """
    Complexity-Aware Genetic Algorithm–Optimized Logistic Regression.

    Parameters
    ----------
    population_size : int, default=200
        Number of individuals P in each generation (paper Sec 3.4).
    generations : int, default=200
        Maximum number of generations G (paper Sec 3.4).
    cx_prob : float, default=0.7
        Crossover probability p_c (paper Sec 3.4).
    mut_prob : float, default=0.2
        Mutation probability p_m (paper Sec 3.4).
    tournament_size : int, default=3
        Tournament selection size (paper Sec 3.4).
    alpha : float, default=0.9
        Weight for validation accuracy in fitness (paper Sec 3.5.3; α+β=1).
    beta : float, default=0.1
        Weight for L2 weight-norm penalty in fitness (paper Sec 3.5.3).
    mini_val_size : float, default=0.30
        Fraction of training data used for stratified mini-validation 
        (paper Sec 3.5.3 / 3.5.5, N′ = 0.30N).
    log_c_bounds : tuple, default=(-4, 4)
        Bounds for log10(C). C is recovered as 10**log_c.
    solver_set : list, default=['lbfgs', 'liblinear', 'saga']
        Candidate solvers (paper Sec 3.5.2).
    penalty : str, default='l2'
        Regularization penalty passed to LogisticRegression.
    max_iter : int, default=1000
        Maximum iterations for LR solver convergence.
    random_state : int, default=None
        Reproducibility seed.
    verbose : bool, default=True
        Print generation-wise progress.
    """

    def __init__(
        self,
        population_size=200,
        generations=200,
        cx_prob=0.7,
        mut_prob=0.2,
        tournament_size=3,
        alpha=0.9,
        beta=0.1,
        mini_val_size=0.30,
        log_c_bounds=(-4, 4),
        solver_set=None,
        penalty="l2",
        max_iter=1000,
        random_state=None,
        verbose=True,
    ):
        if not np.isclose(alpha + beta, 1.0):
            raise ValueError("alpha + beta must equal 1.0 (paper Sec 3.5.3).")

        self.population_size = population_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.beta = beta
        self.mini_val_size = mini_val_size
        self.log_c_min, self.log_c_max = log_c_bounds
        self.penalty = penalty
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        self.rng = np.random.RandomState(random_state)

        # Solver compatibility matrix (solver, penalty, multiclass) — paper Sec 3.5.2
        # sklearn LR compatibility:
        #   lbfgs  : l2, multinomial/ovr
        #   liblinear: l1, l2, ovr ONLY (no multinomial)
        #   saga   : l1, l2, elasticnet, multinomial/ovr
        self.solver_set = solver_set if solver_set is not None else ["lbfgs", "liblinear", "saga"]
        self._build_compatibility_map()

        # Internal state
        self.best_chromosome_ = None
        self.best_fitness_history_ = []
        self.population_stats_ = []
        self.final_model_ = None
        self.scaler_ = None

    # ------------------------------------------------------------------
    # 1. COMPATIBILITY & ENCODING  (Sec 3.5.2)
    # ------------------------------------------------------------------
    def _build_compatibility_map(self):
        """
        Pre-eliminate infeasible (solver, penalty, multiclass) triples.
        Reduces search space by 40–60% as claimed in paper.
        """
        self.compatible_solvers = []
        for sol in self.solver_set:
            ok = False
            if self.penalty == "l2":
                ok = sol in ["lbfgs", "liblinear", "saga", "newton-cg"]
            elif self.penalty == "l1":
                ok = sol in ["liblinear", "saga"]
            elif self.penalty == "elasticnet":
                ok = sol == "saga"
            elif self.penalty == "none":
                ok = sol in ["lbfgs", "saga", "newton-cg"]
            if ok:
                self.compatible_solvers.append(sol)
        if not self.compatible_solvers:
            raise ValueError(f"No solvers compatible with penalty='{self.penalty}'.")
        self.n_solvers = len(self.compatible_solvers)

    def _encode_chromosome(self, log_c, solver_idx):
        """Chromosome χ = [log(C), s]  (paper Eq. in Sec 3.5.2)."""
        return np.array([float(log_c), int(solver_idx)], dtype=object)

    def _decode_chromosome(self, chromosome):
        """Recover C and solver string from chromosome."""
        log_c = float(chromosome[0])
        # Clamp to bounds
        log_c = np.clip(log_c, self.log_c_min, self.log_c_max)
        c_val = 10.0 ** log_c
        solver_idx = int(np.clip(chromosome[1], 0, self.n_solvers - 1))
        solver = self.compatible_solvers[solver_idx]
        return c_val, solver

    # ------------------------------------------------------------------
    # 2. FITNESS EVALUATION  (Sec 3.5.3)
    # ------------------------------------------------------------------
    def _evaluate_fitness(self, chromosome, X_tr, y_tr):
        """
        Stratified mini-validation fitness:
            F(χ) = α·Acc_val(χ) − β·||W||₂
        Uses 30% stratified validation split (N′ = 0.30N) per paper Sec 3.5.5.
        """
        c_val, solver = self._decode_chromosome(chromosome)

        # Determine multiclass strategy based on solver compatibility
        if solver == "liblinear":
            multi_class = "ovr"  # liblinear does not support multinomial
        else:
            multi_class = "multinomial"  # matches softmax Eq. 2 in paper

        try:
            # Stratified mini-validation split
            if self.mini_val_size > 0 and self.mini_val_size < 1.0:
                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=self.mini_val_size,
                    random_state=self.rng.randint(0, 2**31),
                )
                for mini_train_idx, mini_val_idx in sss.split(X_tr, y_tr):
                    X_mini, y_mini = X_tr[mini_train_idx], y_tr[mini_train_idx]
                    X_val, y_val = X_tr[mini_val_idx], y_tr[mini_val_idx]
            else:
                X_mini, y_mini = X_tr, y_tr
                X_val, y_val = X_tr, y_tr

            model = LogisticRegression(
                C=c_val,
                solver=solver,
                penalty=self.penalty,
                multi_class=multi_class,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            model.fit(X_mini, y_mini)
            acc = accuracy_score(y_val, model.predict(X_val))

            # L2 weight norm ||W||₂ (paper Sec 3.5.3)
            # coef_ shape: (n_classes, n_features) for multiclass
            w_norm = np.linalg.norm(model.coef_)

            fitness = self.alpha * acc - self.beta * w_norm
            return fitness, acc, w_norm

        except Exception as e:
            # Infeasible configurations return strongly penalized fitness
            return -1e6, 0.0, 1e6

    # ------------------------------------------------------------------
    # 3. SELECTION  (Tournament, paper Sec 3.4)
    # ------------------------------------------------------------------
    def _select_tournament(self, population, fitnesses):
        """Tournament selection with size=3."""
        selected = []
        pop_size = len(population)
        for _ in range(pop_size):
            contenders = self.rng.choice(pop_size, size=self.tournament_size, replace=False)
            contender_fits = [fitnesses[c] for c in contenders]
            winner = contenders[np.argmax(contender_fits)]
            selected.append(population[winner].copy())
        return selected

    # ------------------------------------------------------------------
    # 4. ADAPTIVE PRUNING  (Sec 3.5.3)
    # ------------------------------------------------------------------
    def _prune_population(self, population, fitnesses):
        """
        Discard individuals below adaptive threshold τ_g = μ_g − σ_g.
        Empirically keeps roughly top 60–70% (paper Sec 3.5.5, ρ_g ≈ 0.33).
        """
        if len(population) <= 2:
            return population, fitnesses

        mu = np.mean(fitnesses)
        sigma = np.std(fitnesses)
        tau = mu - sigma  # τ_g = μ_g − σ_g

        pruned_pop = []
        pruned_fit = []
        for ind, fit in zip(population, fitnesses):
            if fit >= tau:
                pruned_pop.append(ind)
                pruned_fit.append(fit)

        # Safety: if pruning is too aggressive, keep top half
        if len(pruned_pop) < max(2, int(0.2 * self.population_size)):
            n_keep = max(2, int(0.5 * self.population_size))
            idx = np.argsort(fitnesses)[-n_keep:]
            pruned_pop = [population[i] for i in idx]
            pruned_fit = [fitnesses[i] for i in idx]

        return pruned_pop, pruned_fit

    # ------------------------------------------------------------------
    # 5. EVOLUTIONARY OPERATORS  (Sec 3.5.4)
    # ------------------------------------------------------------------
    def _crossover(self, p1, p2):
        """
        Blend (arithmetic) crossover on log(C);
        uniform inheritance for solver index.
        Paper formula: χ^(c) = θ·χ^(p1) + (1−θ)·χ^(p2) applied to log(C) gene.
        """
        if self.rng.rand() > self.cx_prob:
            return p1.copy(), p2.copy()

        c1 = self._encode_chromosome(0.0, 0)
        c2 = self._encode_chromosome(0.0, 0)

        theta = self.rng.rand()
        # Blend log(C)
        c1[0] = theta * float(p1[0]) + (1.0 - theta) * float(p2[0])
        c2[0] = (1.0 - theta) * float(p1[0]) + theta * float(p2[0])

        # Uniform crossover for categorical solver
        if self.rng.rand() < 0.5:
            c1[1] = int(p1[1])
            c2[1] = int(p2[1])
        else:
            c1[1] = int(p2[1])
            c2[1] = int(p1[1])

        # Bounds enforcement
        c1[0] = np.clip(c1[0], self.log_c_min, self.log_c_max)
        c2[0] = np.clip(c2[0], self.log_c_min, self.log_c_max)
        c1[1] = int(np.clip(c1[1], 0, self.n_solvers - 1))
        c2[1] = int(np.clip(c2[1], 0, self.n_solvers - 1))

        return c1, c2

    def _mutate(self, chromosome):
        """
        Mutate exactly ONE gene per individual (paper Sec 3.5.4).
        log(C)  -> bounded Gaussian perturbation.
        solver  -> random compatible solver reset.
        """
        if self.rng.rand() > self.mut_prob:
            return chromosome

        mutant = chromosome.copy()
        gene_to_mutate = self.rng.randint(0, 2)  # 0=logC, 1=solver

        if gene_to_mutate == 0:
            # Bounded Gaussian mutation on log(C)
            sigma = 0.5 * (self.log_c_max - self.log_c_min) / 6.0  # ~1/6 range
            mutant[0] = float(mutant[0]) + self.rng.normal(0, sigma)
            mutant[0] = np.clip(mutant[0], self.log_c_min, self.log_c_max)
        else:
            # Random solver reset
            mutant[1] = self.rng.randint(0, self.n_solvers)

        return mutant

    # ------------------------------------------------------------------
    # 6. MAIN GA LOOP  (Sec 3.5 / Fig. 3)
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """
        Execute the CA-GA-LR optimization pipeline.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features (should be pre-scaled/encoded).
        y : array-like, shape (n_samples,)
            Training labels (multiclass integers).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Optional: standardize for numerical stability (LBFGS/liblinear benefit)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Initialize population (paper: P=200)
        population = []
        for _ in range(self.population_size):
            log_c = self.rng.uniform(self.log_c_min, self.log_c_max)
            solver_idx = self.rng.randint(0, self.n_solvers)
            population.append(self._encode_chromosome(log_c, solver_idx))

        best_global_chromo = None
        best_global_fitness = -np.inf

        for g in range(self.generations):
            # --- Evaluate fitness ---
            fitnesses = []
            accs = []
            wnorms = []
            for ind in population:
                fit, acc, wn = self._evaluate_fitness(ind, X_scaled, y)
                fitnesses.append(fit)
                accs.append(acc)
                wnorms.append(wn)

            fitnesses = np.array(fitnesses)
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fit = fitnesses[gen_best_idx]

            if gen_best_fit > best_global_fitness:
                best_global_fitness = gen_best_fit
                best_global_chromo = population[gen_best_idx].copy()

            self.best_fitness_history_.append(best_global_fitness)
            self.population_stats_.append({
                "gen": g,
                "mean_fit": np.mean(fitnesses),
                "std_fit": np.std(fitnesses),
                "best_fit": gen_best_fit,
                "mean_acc": np.mean(accs),
                "mean_wnorm": np.mean(wnorms),
                "pruned_size": None,
            })

            if self.verbose and (g % 20 == 0 or g == self.generations - 1):
                print(
                    f"Gen {g:03d} | Best={best_global_fitness:.4f} | "
                    f"μ={np.mean(fitnesses):.4f} σ={np.std(fitnesses):.4f} | "
                    f"Acc={accs[gen_best_idx]:.4f}"
                )

            # --- Adaptive pruning (τ_g = μ_g − σ_g) ---
            population, fitnesses = self._prune_population(population, fitnesses)
            self.population_stats_[-1]["pruned_size"] = len(population)

            # --- Selection ---
            selected = self._select_tournament(population, fitnesses)

            # --- Crossover & Mutation to refill population ---
            next_pop = []
            # Elitism: preserve best individual (paper Sec 3.5 / Fig. 3)
            if best_global_chromo is not None:
                next_pop.append(best_global_chromo.copy())

            while len(next_pop) < self.population_size:
                p1 = selected[self.rng.randint(len(selected))]
                p2 = selected[self.rng.randint(len(selected))]
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                next_pop.append(c1)
                if len(next_pop) < self.population_size:
                    next_pop.append(c2)

            population = next_pop[: self.population_size]

        # --- Train final LR with best chromosome χ* on FULL training data ---
        self.best_chromosome_ = best_global_chromo
        c_star, solver_star = self._decode_chromosome(best_global_chromo)
        multi_class_star = "ovr" if solver_star == "liblinear" else "multinomial"

        self.final_model_ = LogisticRegression(
            C=c_star,
            solver=solver_star,
            penalty=self.penalty,
            multi_class=multi_class_star,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.final_model_.fit(X_scaled, y)

        if self.verbose:
            print(f"\n[CA-GA-LR] Final model trained: C={c_star:.4f}, solver={solver_star}")
        return self

    # ------------------------------------------------------------------
    # 7. PREDICTION INTERFACE
    # ------------------------------------------------------------------
    def predict(self, X):
        if self.final_model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler_.transform(X)
        return self.final_model_.predict(X_scaled)

    def predict_proba(self, X):
        if self.final_model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler_.transform(X)
        return self.final_model_.predict_proba(X_scaled)

    def get_best_params(self):
        """Return the optimized hyperparameters (χ* decoded)."""
        if self.best_chromosome_ is None:
            return None
        c_val, solver = self._decode_chromosome(self.best_chromosome_)
        return {"C": c_val, "solver": solver, "penalty": self.penalty}


# =============================================================================
# USAGE EXAMPLE (matching the paper's experimental pipeline)
# =============================================================================
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Synthetic multiclass data mimicking student mental health tabular features
    X, y = make_classification(
        n_samples=1977,
        n_features=12,          # e.g., ensemble-selected 12 features (depression)
        n_informative=8,
        n_redundant=2,
        n_classes=6,            # 6-class depression (paper)
        weights=[0.02, 0.05, 0.21, 0.23, 0.25, 0.24],  # imbalanced
        flip_y=0.05,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # --- Baseline LR (paper Sec 3.4) ---
    baseline = LogisticRegression(max_iter=1000, multi_class="multinomial")
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    print(f"Baseline LR Accuracy: {acc_base:.4f}")

    # --- Proposed CA-GA-LR (paper Sec 3.5) ---
    # NOTE: For full reproducibility with SMOTE, apply SMOTE to X_train/y_train
    # BEFORE calling fit(), as the paper specifies training-fold-only SMOTE (Sec 3.2).
    ca_ga_lr = CAGALRClassifier(
        population_size=200,
        generations=50,         # Use 200 for final experiments; 50 for quick demo
        cx_prob=0.7,
        mut_prob=0.2,
        tournament_size=3,
        alpha=0.9,
        beta=0.1,
        mini_val_size=0.30,
        random_state=42,
        verbose=True,
    )
    ca_ga_lr.fit(X_train, y_train)
    y_pred = ca_ga_lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nCA-GA-LR Accuracy: {acc:.4f}")
    print(f"Best hyperparameters: {ca_ga_lr.get_best_params()}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
