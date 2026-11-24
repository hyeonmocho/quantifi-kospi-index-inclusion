"""Versioned KOSPI200 rules used by the current V3 forecast path.

Selection is deterministic conditional on its inputs. Unknown inputs, historical
profile approximations and committee discretion are reported separately from a
confirmed rule pass. Simulation frequency is not an empirical sample count.
"""
from dataclasses import dataclass, field, asdict, replace
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleProfile:
    name: str
    sector_scheme: str
    cumulative: float
    period_months: int
    listing_months: int
    liquidity_fraction: float = .85
    incumbent_buffer: float = 1.10
    entrant_buffer: float = .90
    target_n: int = 200
    float_minimum: float = .10
    legacy_manufacturing_fill: bool = False
    legacy_min_industry_share: float = .01
    legacy_paired_presence: bool = False
    initial_liquidity_replacement: bool = False
    underfill_open_pool: bool = False
    industry_trim_floor_fraction: float = 0.
    corporate_action_minimum_sessions: int = 0
    review_window_style: str = "whole_month"
    reserve_list: bool = True
    effective_from: str = "1900-01-01"
    known_from: str = "1900-01-01"
    first_applicable_review: str = "1900-01"
    historical_fully_restored: bool = False
    limitations: tuple = ()


PROFILES = (
    RuleProfile("legacy8_partial", "legacy8", .70, 12, 12,
                legacy_manufacturing_fill=True,
                limitations=("Legacy8 classification and historical year-by-year rules are partial proxies",)),
    RuleProfile("gics9_2017", "GICS9", .80, 12, 12,
                initial_liquidity_replacement=True,
                industry_trim_floor_fraction=.90,
                corporate_action_minimum_sessions=30,
                effective_from="2017-06-09", known_from="2017-03-29",
                limitations=("2017 public-rule transition; dated sector coverage still requires checking",)),
    RuleProfile("gics10_2018", "GICS10", .80, 12, 12,
                initial_liquidity_replacement=True,
                industry_trim_floor_fraction=.90,
                corporate_action_minimum_sessions=30,
                effective_from="2018-10-31", known_from="2018-10-31",
                limitations=("Earlier publication date of industry split not reconstructed; conservative known-from",)),
    RuleProfile("gics10_semiannual", "GICS10", .85, 6, 6,
                effective_from="2019-12-11", known_from="2019-12-12",
                first_applicable_review="2020-06",
                limitations=("Amended2019-12-11; published2019-12-12; first regular application2020June",
                             "No industry trim floor in2020.12/2021.07 source; exact removal date from2017 rule unresolved",
                             "Full-month review window follows2021.07 text; earlier six-month wording boundary uncertain",)),
    RuleProfile("gics10_2024_underfill", "GICS10", .85, 6, 6,
                underfill_open_pool=True,
                effective_from="2024-02-01", known_from="2023-12-23",
                limitations=("2024-02 underfill may add any unselected eligible liquid name; special review remains discretionary",)),
    RuleProfile("krics9_2026", "KRICS9", .85, 6, 6, reserve_list=False,
                underfill_open_pool=True,
                effective_from="2026-11-02", known_from="2026-09-29",
                limitations=("2026 issuer notice application date; official classification requires its own publication-date gate",)),
)


def effective_review_date(event):
    """Nominal regular effective date, next day after second Thursday.

    This is sufficient to select historical methodology regimes, not a certified
    future exchange trading calendar. Actual trade dates come from event inputs.
    """
    year, month = map(int, str(event).split("-")[:2])
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(3 - first.weekday()) % 7 + 8)


def profile_for(event, asof=None, cutoff_date=None):
    """Gate publication, amendment and first-applicable review separately.

    This chooses a rule profile; it does not authenticate that an arbitrary
    event string was an actual scheduled regular review.
    """
    target = effective_review_date(event)
    information_date = pd.Timestamp(asof if asof is not None else cutoff_date if cutoff_date is not None else target)
    possible = [p for p in PROFILES
                if pd.Timestamp(p.effective_from) <= target
                and pd.Timestamp(p.known_from) <= information_date
                and pd.Period(str(event)[:7], freq="M") >= pd.Period(p.first_applicable_review, freq="M")]
    return possible[-1]


def review_window(event, sessions, profile=None):
    sessions = pd.DatetimeIndex(sessions)
    year, month = map(int, str(event).split("-")[:2])
    cutoff_candidates = sessions[(sessions.year == year) & (sessions.month == month - 2)]
    if not len(cutoff_candidates):
        raise ValueError(f"No review-cutoff sessions for {event}")
    cutoff = cutoff_candidates[-1]
    p = profile or profile_for(event, cutoff)
    if p.review_window_style not in {"whole_month", "rolling_date"}:
        raise ValueError("Unknown review-window style")
    if p.period_months == 6 and p.review_window_style == "whole_month":
        start = cutoff.replace(day=1) - pd.DateOffset(months=5)
    else:
        start = cutoff - pd.DateOffset(months=p.period_months) + pd.Timedelta(days=1)
    return sessions.searchsorted(start), sessions.get_loc(cutoff)


@dataclass
class SelectionResult:
    frame: pd.DataFrame
    selected: set
    current: set
    a_candidates: set
    reserved_incumbents: set = field(default_factory=set)
    discretionary_candidates: set = field(default_factory=set)
    funnel: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)

    @property
    def confirmed_complete(self):
        return not self.issues and bool(self.profile.get("historical_fully_restored", False))

    @property
    def judge_evaluable(self):
        """Numerically complete conditional inputs; not official certification."""
        return (self.funnel.get("eligibility_unknown", 0) == 0 and
                self.funnel.get("selection_underfill", 0) == 0 and
                self.funnel.get("selection_overfill", 0) == 0 and
                "unknown_incumbent_industry_counts" not in self.issues)


def _sorted(frame, ascending=False):
    return frame.assign(_code=frame.index.astype(str)).sort_values(
        ["mc", "_code"], ascending=[ascending, True], kind="mergesort")


def run_selection(frame, profile=None, target_n=None, largecap_policy="exclude_unverified",
                  legacy_paired_presence=None):
    """Apply published-order selection to a complete as-of market frame.

    Required columns: mc, tv, grp, cur. build_snapshot supplies eligibility_state
    and its evidence. Hand-made fixtures may omit eligibility_state to assert an
    already eligible universe. Unknown incumbents reserve slots but are not
    returned as confirmed selected securities. Existing industry counts include
    all known-sector current members, including those outside the eligible pool.

    largecap_policy='include_sensitivity' activates the discretionary regular
    large-cap clause as a research bound, never as a guaranteed decision.
    """
    p = profile or PROFILES[4]
    if isinstance(p, dict):
        p = RuleProfile(**p)
    if largecap_policy not in {"exclude_unverified", "include_sensitivity", "verified_only"}:
        raise ValueError("Unknown large-cap policy")
    n_target = p.target_n if target_n is None else int(target_n)
    if n_target < 0:
        raise ValueError("target_n must be nonnegative")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate securities")
    needed = {"mc", "tv", "grp", "cur"}
    if not needed.issubset(frame):
        raise ValueError("Missing columns: " + str(needed - set(frame)))
    out = frame.copy().sort_index()
    out["cur"] = out.cur.fillna(False).astype(bool)
    state = out.get("eligibility_state", pd.Series("pass", index=out.index)).fillna("unknown").astype(str)
    if not state.isin(["pass", "fail", "unknown"]).all():
        raise ValueError("Eligibility must be pass/fail/unknown")
    # Unknown cap/industry cannot disappear silently from a competitive universe.
    valid_cap = np.isfinite(pd.to_numeric(out.mc, errors="coerce")) & out.mc.gt(0)
    valid_tv = np.isfinite(pd.to_numeric(out.tv, errors="coerce")) & out.tv.ge(0)
    known_group = out.grp.notna() & ~out.grp.astype(str).isin(["", "미분류", "unknown"])
    state.loc[state.eq("pass") & ~(valid_cap & valid_tv & known_group)] = "unknown"
    if "float_ratio" in out:
        f = pd.to_numeric(out.float_ratio, errors="coerce")
        valid_float = np.isfinite(f) & f.between(0, 1)
        state.loc[state.ne("fail") & ~valid_float] = "unknown"
        state.loc[valid_float & (f < p.float_minimum)] = "fail"
    out["eligibility_state"] = state
    current = set(out.index[out.cur])
    reserved = set(out.index[out.cur & state.eq("unknown")])
    issues = []
    if reserved:
        issues.append("unknown_incumbents_reserved")
    if (out.cur & ~known_group).any():
        issues.append("unknown_incumbent_industry_counts")
    if ((~out.cur) & state.eq("unknown")).any():
        issues.append("unknown_competitors_excluded_from_confirmed_pool")
    age_exception = out.get("largecap_age_exception_unverified", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    if age_exception.any():
        issues.append("regular_largecap_listing_age_exception_not_reconstructed")
        issues.append("largecap_sensitivity_is_conditional_not_exhaustive")
    pool = out.loc[state.eq("pass")].copy()
    known_current_counts = out.loc[out.cur & known_group].groupby("grp").size()
    pool["industry_current_count"] = pool.grp.map(known_current_counts).fillna(0).astype(int)
    pool["mc_rank"] = np.nan
    pool["liquidity_rank"] = np.nan
    pool["liquidity_pass"] = False
    pool["cumulative_before"] = np.nan
    pool["first_stage"] = False
    pool["active_industry"] = True
    pool["selected_reason"] = ""
    pool["industry_trim_floor_protected"] = False
    for group, original in pool.groupby("grp", sort=True):
        g = _sorted(original)
        pool.loc[g.index, "mc_rank"] = np.arange(1, len(g) + 1)
        tv_rank = g.tv.rank(ascending=False, method="average")
        pool.loc[g.index, "liquidity_rank"] = tv_rank
        pool.loc[g.index, "liquidity_pass"] = tv_rank <= p.liquidity_fraction * len(g)
        before = g.mc.cumsum().shift(fill_value=0) / g.mc.sum()
        pool.loc[g.index, "cumulative_before"] = before
        pool.loc[g.index, "first_stage"] = before < p.cumulative
        if p.initial_liquidity_replacement:
            # KRX2019-12-12 describes the old80% rule: replace a liquidity
            # failure by the next liquid market-cap name, preserving S1 quota.
            # The revised85% rule does not contain this replacement step.
            quota = int((before < p.cumulative).sum())
            candidates = g.index[pool.loc[g.index, "liquidity_pass"]][:quota]
            pool.loc[g.index, "first_stage"] = False
            pool.loc[candidates, "first_stage"] = True
        if g.mc.duplicated().any() or g.tv.duplicated().any():
            issues.append("deterministic_tie_handling_not_officially_certified")
    if p.legacy_manufacturing_fill and len(pool):
        shares = pool.groupby("grp").mc.sum() / pool.mc.sum()
        active = set(shares[shares >= p.legacy_min_industry_share].index)
        pool["active_industry"] = pool.grp.isin(active)
        pool.loc[~pool.active_industry, "liquidity_pass"] = False
        # Preserve the documented quota after replacing illiquid nonmanufacturers.
        for group, original in pool.groupby("grp", sort=True):
            if group == "제조업":
                continue
            quota = int(original.first_stage.sum()) if group in active else 0
            candidates = _sorted(original.loc[original.liquidity_pass]).index[:quota]
            pool.loc[original.index, "first_stage"] = False
            pool.loc[candidates, "first_stage"] = True
        issues.append("legacy_manufacturing_profile_is_partial")
    s1 = pool.first_stage & pool.liquidity_pass
    keep = pool.cur & pool.liquidity_pass & (s1 | (pool.mc_rank <= p.incumbent_buffer * pool.industry_current_count))
    new = (~pool.cur) & s1 & (pool.mc_rank <= p.entrant_buffer * pool.industry_current_count)
    legacy_local_restore = set()
    legacy_manufacturing_restore = set()
    legacy_manufacturing_removed = set()
    if p.legacy_manufacturing_fill:
        # Before the2017 reform, nonmanufacturing shortages were restored within
        # each industry. Do not substitute the modern global restoration rule.
        nonm = ~pool.grp.eq("제조업")
        for group, g in pool.loc[nonm].groupby("grp", sort=True):
            existing_n = int(g.industry_current_count.iloc[0])
            group_reserved = int((out.cur & out.grp.eq(group) & state.eq("unknown")).sum())
            target_group = max(0, existing_n - group_reserved)
            chosen_n = int((keep.loc[g.index] | new.loc[g.index]).sum())
            candidates = _sorted(g.loc[g.cur & g.liquidity_pass & ~(keep.loc[g.index] | new.loc[g.index])])
            add = candidates.index[:max(0, target_group - chosen_n)]
            keep.loc[add] = True
            legacy_local_restore.update(add)
        n_nonm = int(((keep | new) & nonm).sum())
        manufacturing = _sorted(pool.loc[pool.grp.eq("제조업") & pool.liquidity_pass])
        remaining = max(0, n_target - len(reserved) - n_nonm)
        m1 = pd.Series(False, index=pool.index)
        m1.loc[manufacturing.index[:remaining]] = True
        pool.loc[~nonm, "first_stage"] = m1.loc[~nonm]
        s1.loc[~nonm] = m1.loc[~nonm] & pool.loc[~nonm, "liquidity_pass"]
        keep.loc[~nonm] = pool.loc[~nonm, "cur"] & pool.loc[~nonm, "liquidity_pass"] & (
            m1.loc[~nonm] | (pool.loc[~nonm, "mc_rank"] <= p.incumbent_buffer * pool.loc[~nonm, "industry_current_count"]))
        new.loc[~nonm] = (~pool.loc[~nonm, "cur"]) & m1.loc[~nonm] & (
            pool.loc[~nonm, "mc_rank"] <= p.entrant_buffer * pool.loc[~nonm, "industry_current_count"])
        m_selected = keep | new
        m_count = int(m_selected.loc[~nonm].sum())
        if m_count < remaining:
            candidates = manufacturing.loc[manufacturing.cur & ~m_selected.reindex(manufacturing.index)]
            add = candidates.index[:remaining - m_count]
            keep.loc[add] = True
            legacy_manufacturing_restore.update(add)
        elif m_count > remaining:
            remove = _sorted(pool.loc[(~nonm) & m_selected], ascending=True).index[:m_count - remaining]
            keep.loc[remove] = False
            new.loc[remove] = False
            legacy_manufacturing_removed.update(remove)
    pairing = p.legacy_paired_presence if legacy_paired_presence is None else bool(legacy_paired_presence)
    if pairing:
        if not p.legacy_manufacturing_fill:
            raise ValueError("Legacy paired-presence sensitivity cannot be imposed on modern methodology")
        for _, g in pool.groupby("grp"):
            has_new = bool(new.loc[g.index].any())
            discretionary_exits = g.cur & g.liquidity_pass & ~keep.loc[g.index]
            if not discretionary_exits.any():
                new.loc[g.index] = False
            if not has_new:
                keep.loc[g.index] |= g.cur & g.liquidity_pass
        issues.append("legacy_paired_presence_sensitivity")
    selected = keep | new
    pool.loc[keep, "selected_reason"] = "liquid_incumbent_first_stage_or_110buffer"
    pool.loc[new, "selected_reason"] = "liquid_new_first_stage_and_90buffer"
    pool.loc[list(legacy_local_restore), "selected_reason"] = "legacy_same_industry_existing_restore"
    pool.loc[list(legacy_manufacturing_restore), "selected_reason"] = "legacy_manufacturing_existing_restore"
    pool.loc[list(legacy_manufacturing_removed), "selected_reason"] = "legacy_manufacturing_low_cap_removal"
    target_confirmed = max(0, n_target - len(reserved))
    initial_n = int(selected.sum())
    if initial_n < target_confirmed and not p.legacy_manufacturing_fill:
        scope = (~selected) & pool.liquidity_pass
        if not p.underfill_open_pool:
            scope &= pool.cur
        candidates = _sorted(pool.loc[scope])
        restore = candidates.index[:target_confirmed - initial_n]
        selected.loc[restore] = True
        pool.loc[restore, "selected_reason"] = ("global_any_eligible_liquid_restore" if p.underfill_open_pool
                                                 else "global_existing_liquid_restore")
    elif initial_n > target_confirmed:
        candidates = _sorted(pool.loc[selected], ascending=True).index
        selected_counts = pool.loc[selected].groupby("grp").size().to_dict()
        reserved_counts = out.loc[out.index.isin(reserved) & known_group].groupby("grp").size()
        left = initial_n - target_confirmed
        for code in candidates:
            if left == 0:
                break
            group = pool.at[code, "grp"]
            # The2017.4 method protects90% of the old industry membership
            # during >200 trimming. Unknown incumbents occupy reserved slots;
            # they are never relabelled as confirmed eligible selections.
            minimum = max(0, int(np.ceil(p.industry_trim_floor_fraction * known_current_counts.get(group, 0)))
                          - int(reserved_counts.get(group, 0)))
            if selected_counts[group] - 1 < minimum:
                pool.at[code, "industry_trim_floor_protected"] = True
                continue
            selected.loc[code] = False
            pool.loc[code, "selected_reason"] = "global_low_cap_removal"
            selected_counts[group] -= 1
            left -= 1
        if left:
            issues.append("industry_trim_floor_prevents_exact_target")
    # §6.4.3 is a committee-permitted regular-review exception. A rank threshold
    # by itself is not a guaranteed selection; no new-listing off-cycle logic here.
    market_rank = pd.to_numeric(out.get("mc15_market_rank", pd.Series(np.nan, index=out.index)), errors="coerce")
    large = set(pool.index[(market_rank.reindex(pool.index) <= 50) & ~selected])
    verified = out.get("largecap_decision_verified", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    apply_large = large if largecap_policy == "include_sensitivity" else (large & set(verified.index[verified]))
    if large:
        issues.append("regular_largecap_discretion_unverified" if largecap_policy != "include_sensitivity" else "regular_largecap_included_as_sensitivity")
    for code in sorted(apply_large, key=lambda c: (market_rank.get(c, np.inf), str(c))):
        victims = _sorted(pool.loc[selected & ~pool.index.isin(apply_large)], ascending=True)
        if selected.sum() >= target_confirmed and len(victims):
            victim = victims.index[0]
            selected.loc[victim] = False
            pool.loc[victim, "selected_reason"] = "replaced_by_discretionary_regular_largecap"
        if target_confirmed > 0:
            selected.loc[code] = True
            pool.loc[code, "selected_reason"] = "discretionary_regular_largecap_sensitivity" if largecap_policy == "include_sensitivity" else "verified_regular_largecap_exception"
    pool["selected"] = selected
    for column in ["industry_current_count", "mc_rank", "liquidity_rank", "liquidity_pass", "cumulative_before", "first_stage", "active_industry", "selected_reason", "industry_trim_floor_protected", "selected"]:
        out[column] = pool[column].reindex(out.index)
    out["selected"] = out.selected.fillna(False).astype(bool)
    out["liquidity_pass"] = out.liquidity_pass.fillna(False).astype(bool)
    confirmed = set(out.index[out.selected])
    a_reason = out.get("a_exclusion_reason", pd.Series("", index=out.index)).fillna("").astype(str)
    a_candidates = confirmed - current - set(out.index[a_reason.ne("")])
    out["a_candidate"] = out.index.isin(a_candidates)
    total_n = len(confirmed) + len(reserved)
    if total_n < n_target:
        issues.append("underfilled_no_undocumented_newcomer_fallback")
    if total_n > n_target:
        issues.append("selected_plus_reserved_exceeds_target")
    funnel = dict(input_rows=len(out), current_members=len(current), eligible_pass=int(state.eq("pass").sum()),
                  eligibility_unknown=int(state.eq("unknown").sum()), eligibility_fail=int(state.eq("fail").sum()),
                  reserved_incumbents=len(reserved), first_stage=int(s1.sum()), initial_selected=initial_n,
                  selected=len(confirmed), selected_plus_reserved=total_n, a_candidates=len(a_candidates),
                  discretionary_candidates=len(large), target_n=n_target,
                   unreconstructed_age_exception_candidates=int(age_exception.sum()),
                   selection_underfill=max(0, n_target - total_n),
                   selection_overfill=max(0, total_n - n_target))
    return SelectionResult(out, confirmed, current, a_candidates, reserved, large, funnel,
                           sorted(set(issues)), asdict(p))


@dataclass
class Snapshot:
    frame: pd.DataFrame
    profile: RuleProfile
    metadata: dict

    def __iter__(self):
        yield self.frame
        yield self.metadata


def _average_descending_rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.repeat((starts + 1 + ends) / 2., ends - starts)
    return ranks


class PreparedSelector:
    """Fast numpy equivalent for conditional joint-MC/turnover simulations.

    Input and returned boolean arrays align with `codes`, in the supplied frame
    order. Eligibility, group, current state and as-of exclusions are fixed.
    Optional mc15_market_rank can vary by scenario for discretionary bounds.
    """
    def __init__(self, frame, profile=None, target_n=None, largecap_policy="exclude_unverified"):
        self.profile = profile or PROFILES[4]
        if isinstance(self.profile, dict):
            self.profile = RuleProfile(**self.profile)
        self.target_n = self.profile.target_n if target_n is None else int(target_n)
        self.largecap_policy = largecap_policy
        self.codes = frame.index.to_numpy(copy=True)
        checked = run_selection(frame, self.profile, self.target_n, largecap_policy).frame.reindex(frame.index)
        self.frame = checked
        self.cur = checked.cur.to_numpy(dtype=bool)
        self.groups = checked.grp.fillna("unknown").astype(str).to_numpy()
        self.pool = checked.eligibility_state.eq("pass").to_numpy()
        self.reserved = self.cur & checked.eligibility_state.eq("unknown").to_numpy()
        self.a_allowed = (~self.cur) & checked.get("a_exclusion_reason", pd.Series("", index=checked.index)).fillna("").eq("").to_numpy()
        self.mc15_rank = pd.to_numeric(checked.get("mc15_market_rank", pd.Series(np.nan, index=checked.index)), errors="coerce").to_numpy()
        self.largecap_verified = checked.get("largecap_decision_verified", pd.Series(False, index=checked.index)).fillna(False).to_numpy(dtype=bool)
        self.indices = np.flatnonzero(self.pool)
        self.group_indices = {}
        self.ncur = np.zeros(len(checked), dtype=int)
        for group in sorted(set(self.groups[self.pool])):
            ids = self.indices[self.groups[self.indices] == group]
            ids = ids[np.argsort(self.codes[ids], kind="stable")]
            self.group_indices[group] = ids
            self.ncur[ids] = np.count_nonzero(self.cur & (self.groups == group))

    def select(self, mc_array, tv_array, mc15_market_rank=None):
        mc = np.asarray(mc_array, dtype=float)
        tv = np.asarray(tv_array, dtype=float)
        if mc.shape != (len(self.codes),) or tv.shape != mc.shape:
            raise ValueError("Simulation arrays must align with prepared.codes")
        if not (np.isfinite(mc[self.pool]) & (mc[self.pool] > 0) & np.isfinite(tv[self.pool]) & (tv[self.pool] >= 0)).all():
            raise ValueError("Invalid simulation values in the fixed eligible pool")
        p = self.profile
        n = len(mc)
        rank = np.full(n, np.nan)
        liq = np.zeros(n, dtype=bool)
        s1 = np.zeros(n, dtype=bool)
        sorted_groups = {}
        total = mc[self.pool].sum()
        for group, ids in self.group_indices.items():
            order = ids[np.argsort(-mc[ids], kind="stable")]
            sorted_groups[group] = order
            rank[order] = np.arange(1, len(order) + 1)
            liq[ids] = _average_descending_rank(tv[ids]) <= p.liquidity_fraction * len(ids)
            cumulative = np.r_[0., np.cumsum(mc[order])[:-1]] / mc[order].sum()
            s1[order] = cumulative < p.cumulative
            if p.initial_liquidity_replacement:
                quota = int(s1[ids].sum())
                s1[ids] = False
                s1[order[liq[order]][:quota]] = True
            if p.legacy_manufacturing_fill:
                if mc[ids].sum() / total < p.legacy_min_industry_share:
                    liq[ids] = False
                if group != "제조업":
                    quota = int(s1[ids].sum()) if liq[ids].any() else 0
                    s1[ids] = False
                    s1[order[liq[order]][:quota]] = True
        s1 &= liq
        keep = self.pool & self.cur & liq & (s1 | (rank <= p.incumbent_buffer * self.ncur))
        new = self.pool & ~self.cur & s1 & (rank <= p.entrant_buffer * self.ncur)
        if p.legacy_manufacturing_fill:
            manufacturing = self.pool & (self.groups == "제조업")
            initial = keep | new
            for group, ids in self.group_indices.items():
                if group == "제조업":
                    continue
                target_group = max(0, self.ncur[ids[0]] - np.count_nonzero(self.reserved & (self.groups == group)))
                needed = max(0, target_group - np.count_nonzero(initial[ids]))
                order = sorted_groups[group]
                add = order[self.cur[order] & liq[order] & ~initial[order]][:needed]
                keep[add] = True
            nonm_selected = np.count_nonzero((keep | new) & ~manufacturing)
            remaining = max(0, self.target_n - np.count_nonzero(self.reserved) - nonm_selected)
            order = sorted_groups.get("제조업", np.empty(0, dtype=int))
            m1 = np.zeros(n, dtype=bool)
            m1[order[liq[order]][:remaining]] = True
            keep[manufacturing] = self.cur[manufacturing] & liq[manufacturing] & (m1[manufacturing] | (rank[manufacturing] <= p.incumbent_buffer * self.ncur[manufacturing]))
            new[manufacturing] = ~self.cur[manufacturing] & m1[manufacturing] & (rank[manufacturing] <= p.entrant_buffer * self.ncur[manufacturing])
            m_selected = keep | new
            m_count = np.count_nonzero(m_selected & manufacturing)
            if m_count < remaining:
                add = order[self.cur[order] & liq[order] & ~m_selected[order]][:remaining - m_count]
                keep[add] = True
            elif m_count > remaining:
                ids = np.flatnonzero(m_selected & manufacturing)
                ids = ids[np.argsort(self.codes[ids], kind="stable")]
                remove = ids[np.argsort(mc[ids], kind="stable")][:m_count - remaining]
                keep[remove] = False
                new[remove] = False
        if p.legacy_paired_presence:
            for ids in self.group_indices.values():
                has_new = new[ids].any()
                if not (self.cur[ids] & liq[ids] & ~keep[ids]).any():
                    new[ids] = False
                if not has_new:
                    keep[ids] |= self.cur[ids] & liq[ids]
        selected = keep | new
        target = max(0, self.target_n - np.count_nonzero(self.reserved))
        if selected.sum() < target and not p.legacy_manufacturing_fill:
            scope = ~selected[self.indices] & liq[self.indices]
            if not p.underfill_open_pool:
                scope &= self.cur[self.indices]
            candidates = self.indices[scope]
            candidates = candidates[np.argsort(self.codes[candidates], kind="stable")]
            order = candidates[np.argsort(-mc[candidates], kind="stable")]
            selected[order[:target - int(selected.sum())]] = True
        elif selected.sum() > target:
            ids = np.flatnonzero(selected)
            ids = ids[np.argsort(self.codes[ids], kind="stable")]
            order = ids[np.argsort(mc[ids], kind="stable")]
            left = int(selected.sum()) - target
            counts = {g: int(np.count_nonzero(selected[ids])) for g, ids in self.group_indices.items()}
            for code_i in order:
                if left == 0:
                    break
                group = self.groups[code_i]
                floor = max(0, int(np.ceil(p.industry_trim_floor_fraction * self.ncur[code_i]))
                            - np.count_nonzero(self.reserved & (self.groups == group)))
                if counts[group] - 1 < floor:
                    continue
                selected[code_i] = False
                counts[group] -= 1
                left -= 1
        ranks15 = self.mc15_rank if mc15_market_rank is None else np.asarray(mc15_market_rank, dtype=float)
        large = self.pool & (ranks15 <= 50) & ~selected
        apply = large if self.largecap_policy == "include_sensitivity" else (large & self.largecap_verified)
        ids = np.flatnonzero(apply)
        ids = ids[np.argsort(self.codes[ids], kind="stable")]
        ids = ids[np.argsort(ranks15[ids], kind="stable")]
        for code_i in ids:
            victims = np.flatnonzero(selected & ~apply)
            victims = victims[np.argsort(self.codes[victims], kind="stable")]
            victims = victims[np.argsort(mc[victims], kind="stable")]
            if selected.sum() >= target and len(victims):
                selected[victims[0]] = False
            if target > 0:
                selected[code_i] = True
        return selected

    def select_many(self, mc, tv, mc15_market_rank=None):
        mc, tv = np.asarray(mc), np.asarray(tv)
        if mc.ndim != 2 or mc.shape != tv.shape:
            raise ValueError("select_many requires equal path x security arrays")
        return np.stack([self.select(a, b, None if mc15_market_rank is None else mc15_market_rank[i])
                         for i, (a, b) in enumerate(zip(mc, tv))])

    def additions_mask(self, selected):
        return np.asarray(selected, dtype=bool) & self.a_allowed


def prepare_selector(frame, profile=None, target_n=None, largecap_policy="exclude_unverified"):
    return PreparedSelector(frame, profile, target_n, largecap_policy)


def _column_or(frame, names, default):
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(default, index=frame.index)


def build_snapshot(re, event, signal=None, inputs=None, lead=60, tv_method="adv20", mc_method="current",
                   review_window_style=None):
    """Build past-only rule inputs including sums for parent joint simulations.

    Data-source module `re` is the original read-only rule_engine; `inputs`
    supplies independently restored float/sector/membership/action snapshots.
    Market-cap units remain the native source's million KRW, turnover KRW.
    """
    if inputs is None:
        raise ValueError("An audited as-of inputs adapter is required")
    if tv_method not in {"adv20", "observed_mean"} or mc_method != "current":
        raise ValueError("Unknown projection policy")
    sessions = pd.DatetimeIndex(re.S)
    # The cutoff calendar is independent of the as-of methodology announcement.
    _, cutoff = review_window(event, sessions, profile_for(event))
    if signal is None:
        signal_i = cutoff - int(lead)
    elif isinstance(signal, (int, np.integer)):
        signal_i = int(signal)
    else:
        signal_i = int(sessions.get_loc(pd.Timestamp(signal)))
    if signal_i < 0 or signal_i > cutoff:
        raise ValueError("Signal must be an available session no later than review cutoff")
    t = sessions[signal_i]
    profile = profile_for(event, asof=t)
    if review_window_style is not None:
        if review_window_style not in {"whole_month", "rolling_date"}:
            raise ValueError("Unknown review-window style")
        profile = replace(profile, review_window_style=review_window_style)
    base_start, cutoff = review_window(event, sessions, profile)
    codes = re.px.columns
    try:
        current = inputs.current_members(t, codes=codes)
    except TypeError:
        current = inputs.current_members(t).reindex(codes, fill_value=False)
    current = current.reindex(codes, fill_value=False).fillna(False).astype(bool)
    try:
        sectors = inputs.get_sector(t, event, codes, profile=profile.sector_scheme)
    except TypeError:
        sectors = inputs.get_sector(t, event, codes)
        returned = sectors.attrs.get("profile")
        if returned and returned != profile.sector_scheme:
            raise ValueError("Sector adapter selected a future/unannounced methodology")
    sectors = sectors.reindex(codes)
    floats = inputs.get_float(t, codes).reindex(codes)
    group = _column_or(sectors, ["grp", "group"], np.nan)
    float_ratio = pd.to_numeric(_column_or(floats, ["float_ratio", "ratio"], np.nan), errors="coerce")
    first = pd.to_datetime(re.FIRST_PX.reindex(codes), errors="coerce").copy()
    known_listings = re.KL.loc[re.KL.list_date <= t]
    dated_first = pd.to_datetime(known_listings.groupby("code").list_date.min().reindex(codes), errors="coerce")
    # A transfer is not a brand-new economic listing: retain the earlier known
    # KOSDAQ/listing observation rather than resetting age at KOSPI transfer.
    first = pd.concat([first, dated_first], axis=1).min(axis=1)
    listing_source = pd.Series(np.where(dated_first.notna(), "dated_listing_and_first_price_earliest", "first_price_proxy"), index=codes)
    # Calendar-month anniversaries are not invertible at month ends: Oct31+6m
    # is Apr30, while Apr30-6m is Oct30. Use the forward anniversary explicitly.
    listing_age_date = first + pd.DateOffset(months=profile.listing_months)
    age_pass = first.notna() & listing_age_date.le(sessions[cutoff])
    # Existing members have the explicitly documented newly-listed/split-member
    # age exception. It does not waive float or liquidity requirements.
    age_pass |= current
    market = re.market_at(t).reindex(codes)
    rows = re.MGT.index[re.MGT.index <= t]
    management = re.MGT.loc[rows[-1]].reindex(codes).astype(str).eq("관리") if len(rows) else pd.Series(False, index=codes)
    historical_names = re._nm_hist.loc[re._nm_hist.date <= t].sort_values("date").drop_duplicates("code", keep="last")
    nm = historical_names.set_index("code")["name"].reindex(codes)
    excluded_name = nm.astype(str).str.contains(re.EXCL)
    nonstock = pd.Series([c in re.NONSTK_FROM.index and re.NONSTK_FROM[c] <= t for c in codes], index=codes)
    ordinary = re.CODE_OK.reindex(codes).fillna(False) & ~excluded_name & ~nonstock
    alive = pd.Series(np.isfinite(re.PXV[signal_i]) & (re.PXV[signal_i] > 0), index=codes)
    now_mc = re.MCD.iloc[signal_i].reindex(codes)
    now_tv = re.TVD.iloc[signal_i].reindex(codes)
    starts = pd.Series(base_start, index=codes, dtype=int)
    reset_status = pd.Series("no_verified_reset_record", index=codes, dtype=object)
    actions = inputs.known_action_resets(t, event=event)
    used_actions = []
    pending_actions = []
    if actions is not None and len(actions):
        for row in actions.to_dict("records"):
            code = row.get("code")
            known = row.get("known_from", row.get("announced_at", row.get("published_at", row.get("publication_date"))))
            effective = row.get("effective_date", row.get("event_date"))
            start_date = row.get("review_start_date", row.get("reset_start_date", row.get("start_date", effective)))
            if code not in starts or not known or not effective or not start_date:
                continue
            if pd.Timestamp(known) > t:
                raise ValueError("Corporate-action information was not public at signal")
            if not bool(row.get("verified", True)):
                continue
            action_type = str(row.get("event_type", row.get("action_type", ""))).lower()
            if action_type in {"small_merger", "small_scale_merger"} or row.get("is_small_merger") is True:
                reset_status.loc[code] = "verified_small_merger_does_not_reset_review"
                continue
            if pd.Timestamp(effective) > t:
                pending_actions.append(row)
                reset_status.loc[code] = "announced_future_action_requires_explicit_path_adjustment"
                continue
            j = int(sessions.searchsorted(pd.Timestamp(start_date)))
            if base_start <= j <= signal_i:
                starts.loc[code] = j
                reset_status.loc[code] = "verified_action_period_reset"
                used_actions.append(row)
                minimum = row.get("minimum_sessions")
                if pd.isna(minimum):
                    if action_type in {"merger", "split", "corporate_split", "company_split"}:
                        minimum = profile.corporate_action_minimum_sessions
                    elif profile.corporate_action_minimum_sessions and cutoff - j + 1 < profile.corporate_action_minimum_sessions and not action_type:
                        reset_status.loc[code] = "verified_reset_action_type_needed_for_minimum_period_rule"
                if pd.notna(minimum) and int(minimum) > 0 and cutoff - j + 1 < int(minimum):
                    # Reconstructing hypothetical prelisting cap would invent
                    # data. Preserve an explicit gate instead of fabricating it.
                    reset_status.loc[code] = "verified_reset_requires_unavailable_minimum_period_adjustment"
    unresolved_actions = []
    if hasattr(inputs, "corporate_action_flags"):
        flags = inputs.corporate_action_flags(t, event=event)
        if flags is not None and len(flags):
            for row in flags.to_dict("records"):
                code = row.get("code")
                published = row.get("announced_at", row.get("known_from"))
                effective = row.get("effective_date")
                if code not in starts or pd.isna(published) or pd.isna(effective):
                    continue
                if pd.Timestamp(published) > t:
                    raise ValueError("Unpublished action flag supplied by adapter")
                if (sessions[base_start] <= pd.Timestamp(effective) <= effective_review_date(event)
                        and reset_status.loc[code] not in {"verified_action_period_reset", "verified_small_merger_does_not_reset_review"}):
                    reset_status.loc[code] = "known_action_review_period_unresolved"
                    unresolved_actions.append(row)
    sum_mc = pd.Series(np.nan, index=codes)
    sum_tv = pd.Series(np.nan, index=codes)
    count_mc = pd.Series(0, index=codes, dtype=int)
    count_tv = pd.Series(0, index=codes, dtype=int)
    expected_obs = (signal_i - starts + 1).clip(lower=0)
    full_days = cutoff - starts + 1
    remaining = pd.Series(cutoff - signal_i, index=codes, dtype=int)
    for begin, subcodes in starts.groupby(starts).groups.items():
        if begin <= signal_i:
            m = re.MCD.iloc[int(begin):signal_i + 1].reindex(columns=subcodes)
            v = re.TVD.iloc[int(begin):signal_i + 1].reindex(columns=subcodes)
            sum_mc.loc[subcodes] = m.sum(min_count=1)
            sum_tv.loc[subcodes] = v.sum(min_count=1)
            count_mc.loc[subcodes] = m.notna().sum()
            count_tv.loc[subcodes] = v.notna().sum()
    adv_block = re.TVD.iloc[max(0, signal_i - 19):signal_i + 1]
    adv20 = adv_block.mean().where(adv_block.notna().sum() >= 10).reindex(codes)
    projected_mc = (sum_mc + remaining * now_mc) / full_days
    future_tv = adv20 if tv_method == "adv20" else sum_tv / expected_obs.replace(0, np.nan)
    projected_tv = (sum_tv + remaining * future_tv) / full_days
    # All KOSPI ordinary shares form the rank50 comparison population; neither
    # listing-age nor industry-liquidity filters shrink this rank universe.
    mc15_block = re.MCD.iloc[max(0, signal_i - 14):signal_i + 1]
    mc15_count = mc15_block.notna().sum().reindex(codes)
    mc15 = mc15_block.mean().where(mc15_count.ge(15)).reindex(codes)
    all_ordinary = market.eq("유가증권시장") & ordinary & alive & mc15.gt(0)
    mc15_rank = mc15.where(all_ordinary).rank(ascending=False, method="min")
    age_exception = ((~current) & ~age_pass & market.eq("유가증권시장") & ordinary & ~management & alive &
                     float_ratio.ge(profile.float_minimum) & float_ratio.le(1) & group.notna() & mc15_rank.le(50))
    state = pd.Series("pass", index=codes, dtype=object)
    reasons = pd.Series("", index=codes, dtype=object)
    fail = ~market.eq("유가증권시장") | ~ordinary | management | ~age_pass
    state.loc[fail] = "fail"
    reasons.loc[fail] = "market_security_management_or_calendar_age_failure"
    float_fail = float_ratio.notna() & float_ratio.lt(profile.float_minimum)
    state.loc[float_fail] = "fail"
    reasons.loc[float_fail] = "float_below_10pct"
    unknown = (~alive | ~np.isfinite(now_mc) | ~now_mc.gt(0) | float_ratio.isna() |
               group.isna() | (count_mc < expected_obs) | (count_tv < expected_obs) |
               projected_mc.isna() | projected_tv.isna() |
               reset_status.isin(["verified_reset_requires_unavailable_minimum_period_adjustment",
                                   "announced_future_action_requires_explicit_path_adjustment",
                                    "verified_reset_action_type_needed_for_minimum_period_rule",
                                   "known_action_review_period_unresolved"]))
    state.loc[state.ne("fail") & unknown] = "unknown"
    reasons.loc[state.eq("unknown")] = "unverified_float_sector_or_complete_past_observations"
    frame = pd.DataFrame(dict(mc=projected_mc, tv=projected_tv, grp=group, cur=current,
                              eligibility_state=state, eligibility_reason=reasons, float_ratio=float_ratio,
                              float_status=_column_or(floats, ["status"], "unknown"),
                              float_observation_date=_column_or(floats, ["observation_date", "source_date"], pd.NaT),
                              sector_status=_column_or(sectors, ["status"], "unknown"),
                              sector_source_date=_column_or(sectors, ["source_date"], pd.NaT),
                              mc_current=now_mc, tv_current=now_tv, adv20=adv20, mc15=mc15,
                               mc15_market_rank=mc15_rank, mc15_observed_count=mc15_count,
                               largecap_age_exception_unverified=age_exception,
                               ordinary_market_pool=all_ordinary,
                              review_start_index=starts, review_sessions=full_days,
                              observed_sessions=expected_obs, observed_mc_count=count_mc, observed_tv_count=count_tv,
                              remaining_sessions=remaining, mc_sum_observed=sum_mc, tv_sum_observed=sum_tv,
                              future_tv=future_tv, corporate_action_status=reset_status,
                              listing_date=first, listing_date_source=listing_source,
                              listing_age_eligible_date=listing_age_date,
                              a_exclusion_reason=np.where(age_exception, "unreconstructed_regular_largecap_listing_age_exception", ""),
                              name=nm.fillna(pd.Series(re.names).reindex(codes))), index=codes)
    # State includes known current members even if they are now ineligible;
    # relevant unknown competitors remain visible to the parent coverage audit.
    visible = market.eq("유가증권시장") | current
    frame = frame.loc[visible].copy()
    # Explicit aliases keep forecast orchestration separate from rule internals.
    frame["mc_last"] = frame.mc_current
    frame["observed_cap_sum"] = frame.mc_sum_observed
    frame["observed_tv_sum"] = frame.tv_sum_observed
    frame["observed_cap_count"] = frame.observed_mc_count
    frame["observed_tv_count"] = frame.observed_tv_count
    frame["special_exclusion"] = frame.a_exclusion_reason
    frame["calendar_age_pass"] = age_pass.reindex(frame.index)
    try:
        from eligibility import strategy_a_exclusion, LISTINGS
        provisional = frame.loc[frame.eligibility_state.eq("pass") & ~frame.cur]
        # Only a recent new/transfer listing can have an unexpired 15/30-day
        # observation followed by the next quarterly special-entry window. Older
        # listings have no pending special window; avoid repeatedly scanning the
        # original full listing table for hundreds of mature nonmembers.
        recent_codes = set(LISTINGS.loc[(LISTINGS.list_date <= t) &
                                       (LISTINGS.list_date > t - pd.DateOffset(years=1)), "code"])
        review_codes = provisional.index[provisional.index.isin(recent_codes)]
        frame.loc[review_codes, "a_exclusion_reason"] = [strategy_a_exclusion(c, "KOSPI200", signal_i) for c in review_codes]
        frame["special_exclusion"] = frame.a_exclusion_reason
    except ImportError:
        raise RuntimeError("Original strategy A exclusion helper must be importable read-only")
    meta = dict(event=event, signal_date=t, cutoff_date=sessions[cutoff], signal_index=signal_i,
                cutoff_index=cutoff, lead_sessions=cutoff - signal_i, profile=asdict(profile),
                rule_known_from=profile.known_from, rule_effective_from=profile.effective_from,
                 first_applicable_review=profile.first_applicable_review,
                 review_window_style=profile.review_window_style,
                 review_window_style_override=review_window_style is not None,
                rule_announcement_gate_respected=True, tv_projection=tv_method, mc_projection=mc_method,
                current_membership_attrs=dict(getattr(current, "attrs", {})),
                security_type_current_name_fallback_used=False,
                security_type_heuristic_rows=int(nm.reindex(frame.index).isna().sum()),
                corporate_action_ledger_partial=True, applied_action_resets=used_actions,
                 regular_largecap_listing_age_exception_coverage_complete=False,
                 unverified_regular_listing_age_exception_codes=list(age_exception.index[age_exception]),
                 liquidation_trading_status_history_verified=False,
                announced_pending_actions=pending_actions,
                known_unresolved_corporate_actions=unresolved_actions,
                complete_past_required=True, future_prices_used=False,
                historical_full_rule_reconstruction=False)
    return Snapshot(frame, profile, meta)


def required_remaining_mean(frame, code, threshold_mc):
    """Algebra conditional on an externally solved final-average MC boundary."""
    row = frame.loc[code]
    remaining = float(row.remaining_sessions)
    if remaining <= 0:
        return np.nan
    return (float(row.review_sessions) * float(threshold_mc) - float(row.mc_sum_observed)) / remaining


def solve_entry_boundary(frame, code, profile=None, target_n=None, largecap_policy="exclude_unverified",
                         max_multiple=100., iterations=30, selection_scope="index"):
    """Rerun all competitors/rules while varying one final-average MC input.

    This is a conditional inversion, not an unconditional price target or a
    guaranteed monotonic theorem. Observed grid violations are explicitly flagged.
    """
    if code not in frame.index:
        raise KeyError(code)
    if selection_scope not in {"index", "strategy_a"}:
        raise ValueError("Unknown inversion scope")
    row = frame.loc[code]
    if row.cur:
        return dict(code=code, status="already_constituent", threshold_mc=np.nan, required_remaining_mean=np.nan)
    base = float(row.mc)
    if not np.isfinite(base) or base <= 0 or row.get("eligibility_state", "pass") != "pass":
        return dict(code=code, status="eligibility_or_data_gate", threshold_mc=np.nan, required_remaining_mean=np.nan)
    prepared = prepare_selector(frame, profile, target_n, largecap_policy)
    code_i = int(np.flatnonzero(prepared.codes == code)[0])
    candidate_mc = frame.mc.to_numpy(dtype=float, copy=True)
    candidate_tv = frame.tv.to_numpy(dtype=float, copy=True)
    def passes(value):
        candidate_mc[code_i] = value
        selected = prepared.select(candidate_mc, candidate_tv)
        return bool(selected[code_i] and (selection_scope == "index" or prepared.a_allowed[code_i]))
    grid = base * np.geomspace(.01, max_multiple, 17)
    states = [passes(x) for x in grid]
    if any(states[i] and not states[i + 1] for i in range(len(states) - 1)):
        return dict(code=code, status="nonmonotone_grid_requires_review", threshold_mc=np.nan, required_remaining_mean=np.nan)
    hit = next((i for i, value in enumerate(states) if value), None)
    if hit is None:
        return dict(code=code, status="no_entry_within_grid", threshold_mc=np.nan, required_remaining_mean=np.nan)
    low, high = (0., grid[0]) if hit == 0 else (grid[hit - 1], grid[hit])
    for _ in range(iterations):
        mid = (low + high) / 2
        if passes(max(mid, np.finfo(float).tiny)):
            high = mid
        else:
            low = mid
    return dict(code=code, status="conditional_boundary_solved", threshold_mc=float(high),
                required_remaining_mean=float(required_remaining_mean(frame, code, high)),
                current_projected_mc=base, threshold_ratio=float(high / base),
                other_competitors_fixed=True, discretionary_largecap_policy=largecap_policy,
                selection_scope=selection_scope, a_exclusion_reason=row.get("a_exclusion_reason", ""))


__all__ = ["RuleProfile", "PROFILES", "SelectionResult", "Snapshot", "PreparedSelector", "prepare_selector", "profile_for", "review_window",
           "run_selection", "build_snapshot", "required_remaining_mean", "solve_entry_boundary"]
