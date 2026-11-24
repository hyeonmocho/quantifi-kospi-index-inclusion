"""Past-only Korean index forecast adaptations; never a replication claim.

Franz's official-rank anchor is NOT available here. ``anchored_rank_update``
substitutes a prior vendor review-average-cap rank for that missing anchor,
then updates it by the change in vendor spot-cap rank. ``path_frequency`` is
our implementation of rule selection under
market-cap uncertainty, not Wilkens's unobserved proprietary model. A scenario
selection frequency is not a calibrated inclusion probability.

All numeric settings below are fixed before inspecting any strategy returns.
The original Engines, source snapshots, eligibility and selectors stay intact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "02_code/modules/index_forecast_broker_20261117/asof_engines.py"
sys.path.insert(0, str(ROOT / "02_code/modules/signal_a_rules_v3"))
from engine_extension import select_snapshot, v2_engine


@dataclass(frozen=True)
class ModelConfig:
    rank_lookback: int = 20
    volatility_lookback: int = 126
    minimum_returns: int = 60
    scenarios: int = 256
    seed: int = 20261117
    beta_shrink: float = 0.50
    sector_covariance_diagonal_shrink: float = 0.25
    beta_lower: float = 0.0
    beta_upper: float = 2.0
    volatility_floor: float = 0.005
    volatility_ceiling: float = 0.10
    historical_winsor_lower: float = 0.01
    historical_winsor_upper: float = 0.99
    frequency_target_rule: str = "same_number_as_frozen_baseline"
    allow_underfilled_k100: bool = False


def _load_engines():
    spec = importlib.util.spec_from_file_location("paper_adaptation_frozen_engines", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Engines()


def _hash_frame(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


class PaperModels:
    """One frozen baseline plus two clearly labelled independent adaptations.

    ``forecast`` returns targets and audit metadata; it never reads post-info
    returns, actual additions or broker targets. Prediction-count is frozen to
    the baseline for the frequency strategy, so its improvement cannot come
    merely from taking fewer positions. Entry/exit/costs belong to the caller.
    """

    def __init__(self, engines=None, config=None):
        self.engines = engines if engines is not None else _load_engines()
        self.config = config if config is not None else ModelConfig()
        self._cache = {}
        self._simulation_cache = {}
        self.underfill_audit = []
        if self.config.frequency_target_rule != "same_number_as_frozen_baseline":
            raise ValueError("Only the predeclared equal-position-count comparison is implemented")

    def protocol(self):
        return {
            "config": asdict(self.config),
            "model_revision": "3_prefix_preserving_rank_and_historical_data_gate",
            "frozen_source": str(SOURCE.relative_to(ROOT)),
            "frozen_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "anchored_rank_update_is_Franz_replication": False,
            "path_frequency_is_Wilkens_replication": False,
            "official_historical_review_ranking_available": False,
            "selection_frequency_is_calibrated_probability": False,
            "return_outcomes_used_by_models": False,
            "eligibility_and_liquidity_projection": "unchanged_frozen_asof_baseline",
            "preserved_computable_gate": "02_code/modules/signal_a_rules_v3/run_v3.py:108: bool(frame.eligibility_state.eq('pass').any())",
            "no_verified_eligible_pool_policy": "all_targets_None_not_zero_return_cash",
            "KOSPI100_rule_limitation": "80/120 buffer remains an unverified proxy",
            "KRX300_rule_limitation": "historical BMI proxy and dated roster availability limits remain",
        }

    def forecast(self, index, event, info):
        info = pd.Timestamp(info).normalize()
        key = (index, str(event), info)
        if key in self._cache:
            return self._cache[key]
        e = self.engines
        baseline, base_meta, frame = e.forecast(index, str(event), info)
        common = dict(index=index, event=str(event), information_date=info.isoformat(),
                      baseline_metadata=base_meta, config=asdict(self.config),
                      calibrated_probability=False, uses_future_observations=False)
        if baseline is None or frame is None:
            result = {m: dict(targets=None, scores={}, metadata=dict(common, status="baseline_unavailable"))
                      for m in ("baseline_carry", "anchored_rank_update", "path_frequency")}
            self._cache[key] = result
            return result
        frame = frame.copy()
        common["asof_frame_hash"] = _hash_frame(frame)
        common["baseline_target_count"] = len(baseline)
        # Preserve V3's pre-existing computable gate. An empty forecast caused
        # by a missing verified eligible pool is unknown, not a cash decision.
        # Partial-unknown competitor rows retain the old reserve/exclude rules.
        if "eligibility_state" in frame and not bool(frame.eligibility_state.eq("pass").any()):
            result = {m: dict(targets=None, scores={}, metadata=dict(common,
                status="blocked_signal_data_no_verified_eligible_pool",
                computable=False, computable_gate_source="signal_a_rules_v3/run_v3.py:108"))
                for m in ("baseline_carry", "anchored_rank_update", "path_frequency")}
            self._cache[key] = result
            return result
        selector, select_audit = self._selector(index, str(event), info, frame, baseline)
        common.update(select_audit)
        projection = self._projection(index, str(event), info, frame)
        current_mc = pd.to_numeric(frame.mc, errors="coerce").to_numpy()
        result = {"baseline_carry": dict(targets=sorted(baseline), scores={}, metadata=dict(common, status="ok"))}
        drift_mc, drift_audit = self._anchored_rank_update(index, str(event), info, frame, projection)
        drift_targets = sorted(selector(drift_mc))
        result["anchored_rank_update"] = dict(targets=drift_targets, scores={},
                                    metadata=dict(common, **drift_audit, status="ok"))
        # With a completed review window no unknown market-cap observations are
        # simulated. Current rule/committee uncertainty is not disguised as MC.
        if int(projection["remaining"]) <= 0:
            frequency = {c: float(c in baseline) for c in frame.index}
            result["path_frequency"] = dict(targets=sorted(baseline), scores=frequency,
                metadata=dict(common, status="ok", simulation_status="review_cap_window_complete_carry",
                              remaining_sessions=0, historical_price_max_date=None,
                              candidate_count_comparable=True))
        else:
            caps, simulation_audit = self._simulate(frame, info, projection, index, str(event))
            counts = pd.Series(0.0, index=frame.index)
            for scenario_index, path in enumerate(caps):
                try:
                    scenario_targets = selector(path)
                except ValueError as error:
                    # An incomplete KOSPI100 proxy roster makes the WHOLE MC
                    # candidate score undefined. Never delete this path, fill
                    # the roster, or renormalize the remaining path counts.
                    if index != "KOSPI100" or str(error) != "KOSPI100 scenario underfill; no unverified fill":
                        raise
                    result["path_frequency"] = dict(targets=None, scores={},
                        metadata=dict(common, **simulation_audit,
                            status="incomplete_scenario_roster",
                            first_incomplete_scenario_index=int(scenario_index),
                            incomplete_scenario_reason=str(error),
                            scenarios_dropped=0, incomplete_roster_filled=False,
                            partial_frequency_counts_published=False))
                    self._cache[key] = result
                    return result
                counts.loc[list(scenario_targets)] += 1.0
            freq = counts / self.config.scenarios
            # Never include actual additions/realised outcomes in ranking. For
            # exact frequency ties baseline membership, current projected cap,
            # then security code provide deterministic pre-outcome tie breaks.
            rank = pd.DataFrame({"frequency": freq, "baseline": freq.index.isin(baseline),
                                 "mc": current_mc, "security_key": freq.index}, index=freq.index)
            rank = rank.loc[rank.frequency.gt(0)].sort_values(
                ["frequency", "baseline", "mc", "security_key"], ascending=[False, False, False, True], kind="stable")
            selected = rank.head(len(baseline)).index.tolist()
            if len(selected) != len(baseline):
                # Since every baseline candidate is eligible and our scenario
                # set contains the carry path, this signals a selector bug.
                raise AssertionError("Scenario candidate union smaller than frozen baseline")
            result["path_frequency"] = dict(targets=sorted(selected), scores=freq.to_dict(),
                metadata=dict(common, **simulation_audit, status="ok",
                              candidate_count_comparable=True,
                              minimum_selected_frequency=float(freq.loc[selected].min()) if selected else None))
        self._cache[key] = result
        return result

    def _selector(self, index, event, info, frame, baseline):
        e = self.engines
        if index in {"KOSPI200", "KOSPI100"}:
            _, snap, selected, _ = e.k200(event, info)
            prepared = v2_engine().prepare_selector(frame, snap.profile)
            carry = prepared.select(frame.mc.to_numpy(), frame.tv.to_numpy())
            if set(frame.index[carry]) != set(selected.core_selected):
                raise AssertionError("Prepared selector differs from frozen V3 core on carry inputs")
            if index == "KOSPI200":
                blocked = set(selected.known_offcycle) | set(selected.removed_by_exception)
                def choose(cap):
                    a = prepared.additions_mask(prepared.select(cap, frame.tv.to_numpy()))
                    return set(frame.index[a]) - blocked
            else:
                prior = e.k100.index[e.k100.index < info]
                if not len(prior):
                    raise ValueError("No dated KOSPI100 membership")
                current = set(e.k100.columns[e.k100.loc[prior[-1]].fillna(False)])
                def choose(cap):
                    members = prepared.select(cap, frame.tv.to_numpy())
                    values = pd.Series(cap, index=frame.index).where(members)
                    target, diag = e.broad.rank_select(values, current, 100, True)
                    if diag["underfill"]:
                        if not self.config.allow_underfilled_k100:
                            raise ValueError("KOSPI100 scenario underfill; no unverified fill")
                        self.underfill_audit.append(dict(index=index,event=event,information_date=str(info),
                            parent_count=int(members.sum()),selected_count=len(target),
                            missing_count=int(diag['underfill']),invented_names=0))
                    return set(target) - current
            if set(choose(frame.mc.to_numpy())) != set(baseline):
                raise AssertionError("Adaptation selector does not reproduce frozen baseline candidates")
            return choose, dict(selector_carry_exact_match=True, rule_profile=snap.profile.name)
        if index != "KRX300":
            raise ValueError("Only KOSPI200, KOSPI100 and KRX300 are supported")
        old, roster_date, age = e.broad.old_roster(index, info, e.snaps)
        if old is None or age > 220:
            raise ValueError("Unavailable or stale KRX300 prior roster")
        def choose(cap):
            f = frame.loc[:, ["tv", "grp"]].copy()
            f["mc"] = np.asarray(cap)
            f = f.sort_values("mc", ascending=False)
            f["global_rank"] = np.arange(1, len(f) + 1)
            f["liquid"] = False
            f["stage1"] = False
            for _, group in f.groupby("grp"):
                z = group.sort_values("mc", ascending=False)
                prev = z.mc.cumsum() / z.mc.sum() - z.mc / z.mc.sum()
                rank = pd.Series(np.arange(1, len(z) + 1), index=z.index)
                liquid = z.tv.rank(ascending=False, method="min").le(.8 * len(z))
                f.loc[z.index, "liquid"] = liquid
                f.loc[z.index, "stage1"] = (prev.lt(.8) | rank.le(.3 * len(z))) & liquid
            target = set(f.index[f.stage1]) | set(f.index[f.index.isin(old) & f.global_rank.le(330) & f.liquid])
            if len(target) > 300:
                target = set(f.loc[list(target)].sort_values("mc", ascending=False).head(300).index)
            if len(target) < 300:
                target.update(f.loc[~f.index.isin(target) & f.liquid].sort_values("mc", ascending=False).head(300-len(target)).index)
            return set(target) - set(old)
        if set(choose(frame.mc.to_numpy())) != set(baseline):
            raise AssertionError("KRX300 adaptation selector differs from frozen BMI proxy")
        return choose, dict(selector_carry_exact_match=True, roster_date=str(roster_date), roster_age_days=int(age),
                            rule_profile="frozen_KRX300_BMI_proxy")

    def _projection(self, index, event, info, frame):
        if index in {"KOSPI200", "KOSPI100"}:
            remaining = pd.to_numeric(frame.remaining_sessions).clip(lower=0)
            # All future observations have the same target cutoff; per-name
            # reset starts affect past sum and denominator, which stay intact.
            return dict(remaining=int(remaining.max()),
                        past_sum=pd.to_numeric(frame.mc_sum_observed).to_numpy(),
                        full_days=pd.to_numeric(frame.review_sessions).to_numpy(),
                        last=pd.to_numeric(frame.mc_current).to_numpy())
        e = self.engines
        dates = e.calendar(index, event)
        eff = dates["effective"]
        cutoff = e.S[(e.S.year == eff.year) & (e.S.month == eff.month - 2)][-1]
        start = cutoff.replace(day=1) - pd.DateOffset(months=5)
        window = e.S[(e.S >= start) & (e.S <= cutoff)]
        observed = window[window <= info]
        past = e.re.MCD.reindex(index=observed, columns=frame.index).sum(min_count=1).to_numpy()
        return dict(remaining=int((window > info).sum()), past_sum=past,
                    full_days=np.full(len(frame), len(window), dtype=float),
                    last=e.re.MCD.loc[info].reindex(frame.index).to_numpy())

    def _anchored_rank_update(self, index, event, info, frame, projection):
        config = self.config
        baseline = pd.to_numeric(frame.mc).copy()
        sessions = self.engines.S
        i = int(sessions.get_loc(info))
        if projection["remaining"] <= 0 or i < config.rank_lookback:
            return baseline.to_numpy(), dict(rank_model_status="completed_review_or_no_rank_history_carry",
                rank_anchor="none_no_official_rank", remaining_sessions=int(projection["remaining"]))
        prior_date = sessions[i-config.rank_lookback]
        _, prior_meta, prior_frame = self.engines.forecast(index, event, prior_date)
        if prior_frame is None or "mc" not in prior_frame:
            return baseline.to_numpy(), dict(rank_model_status="prior_asof_frame_unavailable_carry",
                prior_information_date=str(prior_date), rank_anchor="none_no_official_rank")
        previous = pd.to_numeric(prior_frame.mc).reindex(frame.index)
        spot_current = self.engines.re.MCD.loc[info].reindex(frame.index)
        spot_previous = self.engines.re.MCD.loc[prior_date].reindex(frame.index)
        future_cap = pd.Series(projection["last"], index=frame.index, dtype=float)
        groups = frame.grp.fillna("unknown")
        pass_mask = frame.eligibility_state.eq("pass") if "eligibility_state" in frame else pd.Series(True, index=frame.index)
        changed = 0
        mapped_codes = set()
        for _, ids in groups.loc[pass_mask & baseline.gt(0)].groupby(groups).groups.items():
            ids = pd.Index(ids)
            common = ids[previous.reindex(ids).gt(0) & spot_current.reindex(ids).gt(0) &
                         spot_previous.reindex(ids).gt(0)]
            if len(common) < 2:
                continue
            # Use a COMMON as-of candidate population at both observations;
            # entry/exit of unavailable stocks cannot manufacture rank change.
            anchor = previous.loc[common].rank(ascending=False, method="average")
            rc = spot_current.loc[common].rank(ascending=False, method="average")
            rp = spot_previous.loc[common].rank(ascending=False, method="average")
            predicted_rank = anchor + (rc-rp)
            # A rank correction gives an ordering, not an identified cap level.
            # Reorder POSITIVE future-carry caps, not the review-average caps:
            # the latter could imply a negative remaining-period cap when the
            # already observed prefix is large. Keep that prefix immutable.
            ordered = pd.DataFrame({"rank": predicted_rank, "current_mc": baseline.loc[common],
                                    "security_key": common}, index=common).sort_values(
                ["rank", "current_mc", "security_key"], ascending=[True, False, True], kind="stable").index
            curve = future_cap.loc[common].sort_values(ascending=False).to_numpy()
            future_cap.loc[ordered] = curve
            mapped_codes.update(common)
            changed += int((rc != rp).sum())
        adjusted_average = ((projection["past_sum"] + projection["remaining"]*future_cap.to_numpy()) /
                            projection["full_days"])
        # Preserve all rows outside the explicitly mapped eligible common
        # universe exactly, including unavailable/ineligible observations.
        if mapped_codes:
            ids = sorted(mapped_codes)
            baseline.loc[ids] = pd.Series(adjusted_average, index=frame.index).loc[ids]
        return baseline.to_numpy(), dict(rank_model_status="independent_Korean_vendor_anchored_rank_update",
            prior_information_date=str(prior_date), prior_frame_hash=_hash_frame(prior_frame),
            rank_anchor="prior_vendor_projected_review_average_cap_rank_proxy_not_official",
            rank_basis="prior_vendor_average_rank_plus_change_in_vendor_spot_cap_rank_within_current_sector_common_universe",
            cap_level_mapping="positive_current_sector_future_carry_cap_multiset_permutation_then_fixed_prefix_average",
            observed_review_prefix_preserved=True,
            sector_future_cap_multiset_preserved=True,
            minimum_future_cap=float(future_cap.loc[pass_mask & future_cap.gt(0)].min()),
            rank_change_rows=changed, remaining_sessions=int(projection["remaining"]),
            original_Franz_formula_replicated=False)

    def _simulate(self, frame, info, projection, index, event):
        c = self.config
        # KOSPI100 and KOSPI200 share the same underlying KOSPI200 rule frame.
        # They must receive identical cap scenarios, rather than distinct luck.
        universe_id = "KOSPI200_universe" if index in {"KOSPI200", "KOSPI100"} else index
        cache_key = (universe_id, event, info, _hash_frame(frame))
        if cache_key in self._simulation_cache:
            return self._simulation_cache[cache_key]
        history = self.engines.px.loc[:info, frame.index].tail(c.volatility_lookback+1)
        if not len(history) or history.index.max() > info:
            raise AssertionError("Future price rows entered risk estimation")
        raw = np.log(history).diff().replace([np.inf, -np.inf], np.nan).iloc[1:]
        # Historical outliers only; missing prices never become zero returns.
        low = raw.quantile(c.historical_winsor_lower)
        high = raw.quantile(c.historical_winsor_upper)
        returns = raw.clip(lower=low, upper=high, axis=1)
        own_sd = returns.std(ddof=1)
        enough = returns.notna().sum().ge(c.minimum_returns)
        fallback_sd = float(own_sd.loc[enough].median())
        if not np.isfinite(fallback_sd):
            fallback_sd = .025  # fixed daily fallback, never fitted to outcomes
        own_sd = own_sd.where(enough).fillna(fallback_sd).clip(c.volatility_floor, c.volatility_ceiling)
        groups = frame.grp.fillna("unknown").astype(str)
        unique = sorted(groups.unique())
        common_sd = np.zeros(len(unique))
        sector_history = pd.DataFrame(index=returns.index)
        beta = pd.Series(1.0, index=frame.index)
        residual_sd = own_sd.copy()
        market = returns.median(axis=1)
        market_sd = float(market.std(ddof=1))
        if not np.isfinite(market_sd) or market_sd <= 0:
            market_sd = fallback_sd / 2
        nonfinite_beta_estimates = 0
        for j, group in enumerate(unique):
            ids = groups.index[groups.eq(group)]
            eligible = ids[enough.reindex(ids).fillna(False)]
            factor = returns[eligible].mean(axis=1) if len(eligible) >= 3 else market
            sector_history[group] = factor
            var = float(factor.var(ddof=1))
            common_sd[j] = np.sqrt(var) if np.isfinite(var) and var > 0 else market_sd
            for code in eligible:
                pair = pd.concat([returns[code], factor], axis=1).dropna()
                if len(pair) < c.minimum_returns or not np.isfinite(var) or var <= 0:
                    continue
                pair_factor_variance = float(pair.iloc[:, 1].var(ddof=1))
                if not np.isfinite(pair_factor_variance) or pair_factor_variance <= 0:
                    nonfinite_beta_estimates += 1
                    continue  # preserve the existing beta=1 / residual-risk fallback
                fitted = float(pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair_factor_variance)
                if not np.isfinite(fitted):
                    nonfinite_beta_estimates += 1
                    continue
                b = np.clip((1-c.beta_shrink)*fitted + c.beta_shrink, c.beta_lower, c.beta_upper)
                beta.loc[code] = b
                residual_sd.loc[code] = (pair.iloc[:, 0] - b*pair.iloc[:, 1]).std(ddof=1)
        residual_sd = residual_sd.fillna(fallback_sd).clip(c.volatility_floor, c.volatility_ceiling)
        # Keep the historical correlation between sector factors. Treating all
        # sectors independently would erase the common market shock and change
        # cross-sector selection uncertainty. The diagonal shrink is fixed ex
        # ante and the symmetric PSD repair is numerical, not outcome-fitted.
        covariance = sector_history.cov(min_periods=c.minimum_returns).reindex(index=unique, columns=unique).to_numpy()
        covariance = np.nan_to_num(covariance, nan=0.0)
        np.fill_diagonal(covariance, common_sd**2)
        covariance = ((1-c.sector_covariance_diagonal_shrink)*covariance +
                      c.sector_covariance_diagonal_shrink*np.diag(np.diag(covariance)))
        eigenvalues, eigenvectors = np.linalg.eigh((covariance+covariance.T)/2)
        covariance_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12)))
        covariance = covariance_root @ covariance_root.T
        common_sd = np.sqrt(np.diag(covariance))
        group_i = np.array([unique.index(x) for x in groups])
        b = beta.to_numpy()
        rsd = residual_sd.to_numpy()
        var = b*b*common_sd[group_i]**2 + rsd*rsd
        # Stable per-case randomness is independent of iteration/corpus order.
        tag = hashlib.sha256(f"{universe_id}|{event}|{info.date()}".encode()).digest()
        seed = (c.seed + int.from_bytes(tag[:4], "little")) % (2**32)
        rng = np.random.default_rng(seed)
        n = len(frame)
        paths = np.tile(projection["last"], (c.scenarios, 1))
        sums = np.zeros_like(paths)
        for _ in range(int(projection["remaining"])):
            sector_shock = rng.standard_normal((c.scenarios, len(unique))) @ covariance_root.T
            idio_shock = rng.standard_normal((c.scenarios, n)) * rsd
            shocks = sector_shock[:, group_i] * b + idio_shock
            # E[future cap | info] = current cap; no past stock alpha or
            # realised future return is smuggled into the drift.
            paths *= np.exp(shocks - .5*var)
            sums += paths
        average = (projection["past_sum"] + sums) / projection["full_days"]
        # A deterministic carry path guarantees the predeclared baseline
        # candidate count is feasible in the union of scenario selections.
        average[0] = pd.to_numeric(frame.mc).to_numpy()
        valid = np.isfinite(average) & (average > 0)
        average = np.where(valid, average, pd.to_numeric(frame.mc).to_numpy()[None, :])
        audit = dict(simulation_status="independent_sector_common_idiosyncratic_cap_paths",
            remaining_sessions=int(projection["remaining"]), historical_price_max_date=str(history.index.max()),
            historical_price_min_date=str(history.index.min()), history_hash=_hash_frame(history),
            return_observations_max=int(returns.notna().sum().max()),
            insufficient_history_rows=int((~enough).sum()),
            nonfinite_beta_estimates=int(nonfinite_beta_estimates),
            insufficient_history_policy="fixed_cross_section_daily_volatility_fallback_no_candidate_drop",
            scenario_count=c.scenarios, stochastic_scenarios=c.scenarios-1, deterministic_carry_scenarios=1,
            scenario_seed=int(seed), drift="zero_expected_simple_cap_return", future_share_count="held_constant",
            turnover_and_eligibility="fixed_asof_baseline", stock_alpha_removed=False,
            sector_dependence="historical_sector_factor_covariance_with_fixed_diagonal_shrinkage",
            original_Wilkens_model_replicated=False,
            simulation_frequency_is_calibrated_probability=False)
        self._simulation_cache[cache_key] = (average, audit)
        return average, audit


__all__ = ["ModelConfig", "PaperModels"]
