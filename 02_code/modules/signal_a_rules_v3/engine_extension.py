"""Source-qualified regular-large-cap research extension of immutable V2.

The committee may select a qualifying large cap; the rank threshold alone is
never an official decision. Regular and off-cycle new-listing decisions remain
separate. This module does not learn rules from the five observed V2 misses.
"""
from dataclasses import dataclass, field, asdict, replace
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib.util
import sys
import numpy as np
import pandas as pd

V2_PATH = Path(__file__).resolve().parents[1] / "signal_a_rules_v2/engine.py"
V2_EXPECTED_SHA256 = "1166a8fefa26fb69126cffb8e9e962c876a78e59c188c2d42d8af164501ef8a9"


@lru_cache(maxsize=1)
def v2_engine():
    if hashlib.sha256(V2_PATH.read_bytes()).hexdigest() != V2_EXPECTED_SHA256:
        raise RuntimeError("Immutable V2 engine no longer matches the frozen research source")
    spec = importlib.util.spec_from_file_location("signal_a_v2_immutable_for_v3", V2_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class RegularLargecapProfile:
    name: str
    effective_from: str
    known_from: str | None
    young_listing_exception_allowed: bool
    minimum_observations: int = 15
    top_rank: int = 50
    rule_source: str = ""
    age_exception_publication_verified: bool = False
    announcement_date: str | None = None
    limitations: tuple = ()


LARGECAP_PROFILES = (
    RegularLargecapProfile("before2015_no_documented_age_waiver", "1900-01-01", "1900-01-01", False,
        rule_source="KIND2011 disclosure; KRX2020/2021 history distinguishes pre2015 requirement"),
    RegularLargecapProfile("2015_regular_age_waiver_historical", "2015-01-26", None, True,
        rule_source="KRX2020.12/2021.07 primary methodology history:2015-01-26 regular age-waiver introduction",
        limitations=("Original2014/2015 first-publication notice not recovered; historical legality is stronger evidence than publication timing",
                     "Date2015-01-26 in later KRX history differs from contemporaneous2014 press reporting; do not fabricate a first-known timestamp")),
    RegularLargecapProfile("2024_nov_young_regular_excluded", "2024-11-01", "2024-10-22", False,
        rule_source="KIND2024-10-21 TIMEFOLIO issuer notice; effective2024-11-01",
        age_exception_publication_verified=True, announcement_date="2024-10-21",
        limitations=("Date-only announcement; executable knowledge is conservatively next trading day2024-10-22",)),
)


def regular_profile_for(event, asof):
    """Known announcements gate future-effective rules; unknown dates stay marked."""
    target = v2_engine().effective_review_date(event)
    t = _kst(asof)
    candidates = [p for p in LARGECAP_PROFILES
                  if pd.Timestamp(p.effective_from) <= target
                  and (p.known_from is None or pd.Timestamp(p.known_from) <= t)]
    return candidates[-1]


@dataclass
class ExtensionSnapshot:
    frame: pd.DataFrame
    profile: object
    metadata: dict


@dataclass
class ExtensionSelection:
    frame: pd.DataFrame
    selected: set
    current: set
    a_candidates: set
    core_selected: set = field(default_factory=set)
    possible_regular: set = field(default_factory=set)
    pending_regular: set = field(default_factory=set)
    applied_regular: set = field(default_factory=set)
    confirmed_regular_decisions: set = field(default_factory=set)
    known_offcycle: set = field(default_factory=set)
    removed_by_exception: set = field(default_factory=set)
    issues: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _series(frame, key, default):
    return frame[key] if key in frame else pd.Series(default, index=frame.index)


def _kst(value):
    t = pd.Timestamp(value)
    return t.tz_convert("Asia/Seoul").tz_localize(None) if t.tzinfo is not None else t


def _verified(value):
    return value is True or isinstance(value, np.bool_) and bool(value) or isinstance(value, str) and value.strip().lower() in {"true", "1"}


def previous_regular_effective_date(event):
    year, month = map(int, str(event).split("-")[:2])
    previous = f"{year-1}-06" if str(event) <= "2020-06" else f"{year}-06" if month == 12 else f"{year-1}-12"
    return v2_engine().effective_review_date(previous)


def _knowledge_date(row):
    for key in ["known_from", "announced_at", "publication_date"]:
        if pd.notna(row.get(key)):
            return _kst(row[key])
    return pd.NaT


def _decision_frame(decisions, asof, event):
    if decisions is None or not len(decisions):
        return pd.DataFrame(columns=["code", "decision_type", "known_from", "verified"])
    d = decisions.copy()
    if "code" not in d or "decision_type" not in d:
        raise ValueError("Membership decisions require code and decision_type")
    d["_known"] = [_knowledge_date(row) for row in d.to_dict("records")]
    verified = d.get("verified", pd.Series(False, index=d.index)).map(_verified)
    d = d.loc[verified & d._known.notna() & d._known.le(_kst(asof))]
    if "decision_status" in d:
        status = d.decision_status.astype("string").str.strip().str.lower()
        # Old verified-decision rows may have no status field/value. Explicit
        # pending/conditional document annotations are never final decisions.
        d = d.loc[status.isna() | status.eq("") | status.isin(["final", "confirmed"])]
    if "event" in d:
        # Off-cycle records may be registered outside a regular event label.
        regular = d.decision_type.astype(str).str.startswith("regular")
        d = d.loc[~regular | d.event.astype(str).eq(str(event))]
    # Resolve explicit supersession only after applying the knowledge gate;
    # tomorrow's correction must not erase today's known announcement.
    if "source_id" in d:
        superseded = pd.Series(False, index=d.index)
        for row in d.to_dict("records"):
            prior = set()
            for field in ["supersedes", "supersedes_source_id"]:
                value = row.get(field)
                if value is None or not isinstance(value, (list, tuple, set)) and pd.isna(value):
                    continue
                values = value if isinstance(value, (list, tuple, set)) else str(value).replace(";", ",").split(",")
                prior.update(str(x).strip() for x in values if str(x).strip())
            match = d.source_id.astype(str).isin(prior)
            if str(row.get("correction_scope", "row")).lower() != "source":
                match &= d.code.eq(row["code"])
                if "event" in d and pd.notna(row.get("event")):
                    match &= d.event.eq(row["event"])
            superseded |= match
        d = d.loc[~superseded]
    # Latest known state within the regular/offcycle channel wins, before the
    # inclusion-only filter. A known cancellation must not leave an old add alive.
    d = d.assign(_channel=np.where(d.decision_type.astype(str).str.startswith("regular"), "regular", "offcycle"))
    keys = (["event"] if "event" in d else []) + ["code", "_channel"]
    if len(d) and "side" in d:
        latest = d._known.eq(d.groupby(keys, dropna=False)._known.transform("max"))
        conflict = d.loc[latest].groupby(keys, dropna=False).side.nunique().gt(1)
        if conflict.any():
            bad_keys = set(conflict.index[conflict])
            row_keys = [tuple(row) for row in d[keys].fillna("__missing_event__").to_numpy()]
            normalized_bad = {tuple("__missing_event__" if pd.isna(x) else x for x in (key if isinstance(key, tuple) else (key,))) for key in bad_keys}
            d = d.loc[[key not in normalized_bad for key in row_keys]]
    d = d.sort_values("_known", kind="stable").drop_duplicates(keys, keep="last")
    # A past off-cycle inclusion is a dated state transition, not a permanent
    # blacklist. The current review cycle is the maximum exclusion interval.
    offcycle = d.decision_type.astype(str).str.startswith("offcycle")
    effective = d.get("effective_date", pd.Series(pd.NaT, index=d.index)).map(lambda x: _kst(x) if pd.notna(x) else pd.NaT)
    current_cycle = effective.gt(previous_regular_effective_date(event)) & effective.le(v2_engine().effective_review_date(event))
    d = d.loc[~offcycle | current_cycle]
    if "side" in d:
        d = d.loc[d.side.astype(str).str.lower().isin(["add", "inclusion", "편입"])]
    return d


def select_with_regular_largecap(frame, profile, *, event, asof,
                                policy="core", decisions=None,
                                strict_source_timing=False):
    """Apply core/conditional-discretion/confirmed-asof policies explicitly.

    Required extension columns for young candidates are
    regular_hard_eligibility_state, regular_rank15, regular_count15 and
    regular_window_end. When these are unavailable the candidate stays unknown.
    V2 eligibility is never relaxed wholesale. A young IPO's natural lack of a
    full review year is not treated as a failure of the independent15-day test.
    """
    if policy not in {"core", "conditional_discretion", "confirmed_asof"}:
        raise ValueError("Unknown regular-largecap policy")
    v2 = v2_engine()
    if isinstance(profile, dict):
        profile = v2.RuleProfile(**profile)
    f = frame.copy()
    f["largecap_decision_verified"] = False
    base = v2.run_selection(f, profile, largecap_policy="exclude_unverified")
    f = base.frame.reindex(frame.index).copy()
    t = _kst(asof)
    lp = regular_profile_for(event, t)
    d = _decision_frame(decisions, t, event)
    regular = set(d.loc[d.decision_type.astype(str).str.startswith("regular"), "code"])
    offcycle = set(d.loc[d.decision_type.astype(str).str.startswith("offcycle"), "code"])
    hard = _series(f, "regular_hard_eligibility_state", "unknown").fillna("unknown").astype(str)
    # Mature hand-made/V2 rows can use their existing hard eligibility; age
    # exceptions require the builder's independent non-age gate.
    age_pass = _series(f, "calendar_age_pass", True).fillna(False).astype(bool)
    if "regular_hard_eligibility_state" not in f:
        hard.loc[age_pass] = f.loc[age_pass, "eligibility_state"]
    if "float_ratio" in f:
        ratio = pd.to_numeric(f.float_ratio, errors="coerce")
        valid_ratio = np.isfinite(ratio) & ratio.between(0, 1)
        hard.loc[valid_ratio & ratio.lt(profile.float_minimum)] = "fail"
        hard.loc[hard.ne("fail") & ~valid_ratio] = "unknown"
    ranks = pd.to_numeric(_series(f, "regular_rank15", np.nan), errors="coerce")
    counts = pd.to_numeric(_series(f, "regular_count15", np.nan), errors="coerce")
    ends = _series(f, "regular_window_end", pd.NaT).map(lambda x: _kst(x) if pd.notna(x) else pd.NaT)
    future_data = ends.notna() & ends.gt(t)
    status = pd.Series("not_largecap_candidate", index=f.index, dtype=object)
    noncurrent = ~f.cur
    status.loc[noncurrent & hard.eq("unknown")] = "unknown_nonage_hard_eligibility"
    status.loc[noncurrent & hard.eq("fail")] = "failed_nonage_hard_eligibility"
    gate = noncurrent & hard.eq("pass")
    status.loc[gate & (counts.isna() | counts.lt(lp.minimum_observations))] = "pending_complete15_observations"
    status.loc[gate & counts.ge(lp.minimum_observations) & ranks.isna()] = "unknown_all_market_rank15"
    status.loc[gate & ends.isna()] = "unknown_market_window_date"
    status.loc[gate & future_data] = "future_market_data_rejected"
    complete = gate & counts.ge(lp.minimum_observations) & ranks.notna() & ends.notna() & ~future_data
    status.loc[complete & ranks.gt(lp.top_rank)] = "outside_all_market_top50"
    threshold = complete & ranks.le(lp.top_rank)
    young = threshold & ~age_pass
    status.loc[threshold & age_pass] = "conditional_regular_discretion"
    status.loc[young] = "conditional_regular_age_exception" if lp.young_listing_exception_allowed else "young_regular_exception_not_permitted"
    if strict_source_timing and not lp.age_exception_publication_verified:
        status.loc[young & lp.young_listing_exception_allowed] = "age_exception_first_publication_unverified"
    possible_mask = status.isin(["conditional_regular_discretion", "conditional_regular_age_exception"])
    f["regular_largecap_status"] = status
    f["regular_official_decision_known"] = f.index.isin(regular)
    f["known_offcycle_decision"] = f.index.isin(offcycle)
    possible = set(f.index[possible_mask]) - offcycle
    pending = set(f.index[status.str.startswith(("unknown", "pending", "age_exception_first"))])
    apply = set() if policy == "core" else possible if policy == "conditional_discretion" else possible & regular
    selected = set(base.selected)
    applied, removed = set(), set()
    target = int(base.funnel["target_n"]) - len(base.reserved_incumbents)
    for code in sorted(apply - selected, key=lambda c: (ranks.loc[c], str(c))):
        victims = f.loc[f.index.isin(selected - apply)].copy()
        if len(selected) >= target:
            if not len(victims):
                continue
            victims = victims.assign(_code=victims.index.astype(str)).sort_values(["mc", "_code"], kind="stable")
            victim = victims.index[0]
            selected.remove(victim)
            removed.add(victim)
        if target > 0:
            selected.add(code)
            applied.add(code)
    # Existing strategy exclusions are retained, except the now-explicit regular
    # age-exception placeholder. Only an off-cycle route excludes all regular
    # exceptions automatically; no blanket new-listing/transfer-name blacklist.
    reasons = _series(f, "a_exclusion_reason", "").fillna("").astype(str)
    placeholder = reasons.str.startswith("unreconstructed_regular_largecap")
    route_offcycle = _series(f, "strategy_a_route", "regular_candidate").fillna("unknown").eq("offcycle")
    excluded = set(f.index[(reasons.ne("") & ~placeholder) | route_offcycle]) | offcycle
    route_unknown = _series(f, "strategy_a_route", "regular_candidate").fillna("unknown").eq("unknown")
    excluded |= set(f.index[route_unknown])
    a_candidates = selected - set(base.current) - excluded
    f["extension_selected"] = f.index.isin(selected)
    f["extension_a_candidate"] = f.index.isin(a_candidates)
    f["regular_exception_applied"] = f.index.isin(applied)
    known_not_applied = (regular & set(f.index[~f.cur])) - selected
    f["known_regular_but_not_applied"] = f.index.isin(known_not_applied)
    issues = list(base.issues)
    if possible:
        issues.append("regular_largecap_selection_is_committee_discretion")
    if lp.young_listing_exception_allowed and not lp.age_exception_publication_verified:
        issues.append("historical_age_waiver_first_primary_publication_not_recovered")
    if pending:
        issues.append("some_regular_largecap_inputs_pending_or_unknown")
    if future_data.any():
        issues.append("future_rank_window_rejected")
    if known_not_applied:
        issues.append("known_regular_fact_not_promoted_under_policy_or_conservative_input_gate")
    return ExtensionSelection(f, selected, set(base.current), a_candidates, set(base.selected),
        possible, pending, applied, regular, offcycle, removed, sorted(set(issues)),
        dict(policy=policy, asof=str(t), event=event, regular_profile=asdict(lp),
             strict_source_timing=strict_source_timing, observed_decisions=len(d),
             known_regular_not_applied=sorted(known_not_applied),
             conditional_selection_is_official_confirmation=False,
             past_return_outcomes_used=False, full_index_management_reconstruction=False))


def select_snapshot(snapshot, inputs, policy="core", strict_source_timing=False):
    meta = snapshot.metadata
    asof = meta["information_asof"]
    event = meta["event"]
    if hasattr(inputs, "official_decisions"):
        decisions = inputs.official_decisions(asof, event=event)
    elif hasattr(inputs, "official_membership_decisions"):
        decisions = inputs.official_membership_decisions(asof, event=event)
    else:
        decisions = None
    return select_with_regular_largecap(snapshot.frame, snapshot.profile, event=event,
        asof=asof, policy=policy, decisions=decisions, strict_source_timing=strict_source_timing)


def _column_or(frame, names, default=np.nan):
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(default, index=frame.index)


# build_extension_snapshot is implemented below without importing the original
# strategy_A helper's global calendar. Extended2026 panels must supply a date hook.


def review_window(event, sessions, profile=None):
    return v2_engine().review_window(event,sessions,profile)

def profile_for(event,asof=None,cutoff_date=None):
    return v2_engine().profile_for(event,asof,cutoff_date)

def effective_review_date(event):
    return v2_engine().effective_review_date(event)

def build_extension_snapshot(re, event, signal=None, inputs=None, lead=60, tv_method="adv20", mc_method="current",
                   review_window_style=None, asof=None, allow_after_cutoff=False):
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
    if signal_i < 0 or signal_i >= len(sessions) or (signal_i > cutoff and not allow_after_cutoff):
        raise ValueError("Signal must be an available session no later than review cutoff")
    observation_t = sessions[signal_i]
    t = _kst(asof) if asof is not None else observation_t
    if t < observation_t:
        raise ValueError("Information asof cannot precede its market observations")
    review_observed_i = min(signal_i, cutoff)
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
    group = _column_or(sectors, ["grp", "group"], np.nan).copy()
    float_ratio = pd.to_numeric(_column_or(floats, ["float_ratio", "ratio"], np.nan), errors="coerce")
    sector_dates = _column_or(sectors, ["known_from", "source_date"], pd.NaT).map(lambda x: _kst(x) if pd.notna(x) else pd.NaT)
    float_dates = _column_or(floats, ["known_from", "source_date", "observation_date"], pd.NaT).map(lambda x: _kst(x) if pd.notna(x) else pd.NaT)
    future_sector = sector_dates.gt(t)
    future_float = float_dates.gt(t)
    future_input = future_sector | future_float
    group.loc[future_sector] = np.nan
    float_ratio.loc[future_float] = np.nan
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
            if _kst(known) > t:
                raise ValueError("Corporate-action information was not public at signal")
            if not _verified(row.get("verified", False)):
                continue
            action_type = str(row.get("event_type", row.get("action_type", ""))).lower()
            if action_type in {"small_merger", "small_scale_merger"} or row.get("is_small_merger") is True:
                reset_status.loc[code] = "verified_small_merger_does_not_reset_review"
                continue
            if _kst(effective) > t:
                pending_actions.append(row)
                reset_status.loc[code] = "announced_future_action_requires_explicit_path_adjustment"
                continue
            j = int(sessions.searchsorted(_kst(start_date)))
            if j > cutoff:
                pending_actions.append(row)
                reset_status.loc[code] = "known_after_cutoff_action_requires_specific_adjustment"
                continue
            if base_start <= j <= review_observed_i:
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
                if _kst(published) > t:
                    raise ValueError("Unpublished action flag supplied by adapter")
                if (sessions[base_start] <= _kst(effective) <= effective_review_date(event)
                        and reset_status.loc[code] not in {"verified_action_period_reset", "verified_small_merger_does_not_reset_review"}):
                    reset_status.loc[code] = "known_action_review_period_unresolved"
                    unresolved_actions.append(row)
    sum_mc = pd.Series(np.nan, index=codes)
    sum_tv = pd.Series(np.nan, index=codes)
    count_mc = pd.Series(0, index=codes, dtype=int)
    count_tv = pd.Series(0, index=codes, dtype=int)
    expected_obs = (review_observed_i - starts + 1).clip(lower=0)
    full_days = cutoff - starts + 1
    remaining = pd.Series(max(0, cutoff - signal_i), index=codes, dtype=int)
    for begin, subcodes in starts.groupby(starts).groups.items():
        if begin <= review_observed_i:
            m = re.MCD.iloc[int(begin):review_observed_i + 1].reindex(columns=subcodes)
            v = re.TVD.iloc[int(begin):review_observed_i + 1].reindex(columns=subcodes)
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
    unknown = (future_input | ~alive | ~np.isfinite(now_mc) | ~now_mc.gt(0) | float_ratio.isna() |
               group.isna() | (count_mc < expected_obs) | (count_tv < expected_obs) |
               projected_mc.isna() | projected_tv.isna() |
               reset_status.isin(["verified_reset_requires_unavailable_minimum_period_adjustment",
                                   "announced_future_action_requires_explicit_path_adjustment",
                                    "verified_reset_action_type_needed_for_minimum_period_rule",
                                   "known_action_review_period_unresolved",
                                    "known_after_cutoff_action_requires_specific_adjustment"]))
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
    frame["strategy_a_route"] = "regular_candidate"
    frame["a_exclusion_reason"] = ""
    route_hook = getattr(inputs, "strategy_a_exclusion", None)
    if route_hook is None:
        frame.loc[~frame.cur, "strategy_a_route"] = "unknown"
        frame.loc[~frame.cur, "a_exclusion_reason"] = "unknown_date_based_strategy_a_route"
    else:
        for code in frame.index[~frame.cur]:
            exclusion = route_hook(code, event, t)
            if isinstance(exclusion, dict):
                frame.at[code, "strategy_a_route"] = exclusion.get("route", "unknown")
                frame.at[code, "a_exclusion_reason"] = exclusion.get("reason", "")
            elif exclusion:
                frame.at[code, "strategy_a_route"] = "offcycle"
                frame.at[code, "a_exclusion_reason"] = str(exclusion)
    frame["special_exclusion"] = frame.a_exclusion_reason
    frame["regular_hard_eligibility_state"] = "pass"
    # A failure of age alone can be tested under the historical regular exception.
    # Security type, management and float requirements are never waived.
    hard_fail = (~market.eq("유가증권시장") | ~ordinary | management | float_fail).reindex(frame.index)
    frame.loc[hard_fail, "regular_hard_eligibility_state"] = "fail"
    hard_unknown = (future_input | float_ratio.isna() | ~float_ratio.between(0, 1) | group.isna() |
                    ~alive | ~now_mc.gt(0) | ~np.isfinite(now_mc) |
                    reset_status.isin(["verified_reset_requires_unavailable_minimum_period_adjustment",
                                       "announced_future_action_requires_explicit_path_adjustment",
                                       "verified_reset_action_type_needed_for_minimum_period_rule",
                                       "known_action_review_period_unresolved",
                                       "known_after_cutoff_action_requires_specific_adjustment"]))
    frame.loc[frame.regular_hard_eligibility_state.ne("fail") & hard_unknown.reindex(frame.index), "regular_hard_eligibility_state"] = "unknown"
    frame["regular_hard_eligibility_reason"] = np.where(frame.regular_hard_eligibility_state.eq("pass"), "known_nonage_requirements_pass", "nonage_requirement_failed_or_unverified")
    frame["regular_rank15"] = frame.mc15_market_rank
    tv15_count = re.TVD.iloc[max(0, signal_i-14):signal_i+1].notna().sum().reindex(frame.index)
    frame["regular_turnover_count15"] = tv15_count
    frame["regular_count15"] = np.minimum(frame.mc15_observed_count, tv15_count)
    frame["regular_window_end"] = observation_t
    frame["regular_window_basis"] = "observed_asof_window_not_future_committee_outcome"
    frame["committee_review_date"] = pd.NaT
    frame["committee_date_known"] = False
    if hasattr(inputs, "regular_committee_calendar"):
        calendar = inputs.regular_committee_calendar(event, t)
        if isinstance(calendar, dict):
            calendar = pd.DataFrame([calendar])
        if calendar is not None and len(calendar):
            for row in calendar.to_dict("records"):
                known = _knowledge_date(row)
                review_date = pd.to_datetime(row.get("review_date"), errors="coerce")
                if pd.notna(known) and known <= t and pd.notna(review_date) and _verified(row.get("verified", False)):
                    frame["committee_review_date"] = review_date
                    frame["committee_date_known"] = True
                    # Known schedules do not reveal future market observations.
                    if review_date > observation_t:
                        frame["regular_window_basis"] = "current_observed_window_future_known_committee_pending"
                    else:
                        # Date-only committee information does not establish
                        # whether that day's closing prices existed at the meeting.
                        # Use the previous observed session as a conservative
                        # sensitivity, explicitly not a certified official window.
                        end_i = int(sessions.searchsorted(review_date, side="left")) - 1
                        if end_i >= 14:
                            block = re.MCD.iloc[end_i-14:end_i+1]
                            count = block.notna().sum().reindex(codes)
                            tv_count = re.TVD.iloc[end_i-14:end_i+1].notna().sum().reindex(codes)
                            mean = block.mean().where(count.ge(15)).reindex(codes)
                            mk = re.market_at(sessions[end_i]).reindex(codes)
                            rank_pool = mk.eq("유가증권시장") & ordinary & mean.gt(0)
                            rank = mean.where(rank_pool).rank(ascending=False, method="min")
                            frame["regular_rank15"] = rank.reindex(frame.index)
                            frame["regular_count15"] = np.minimum(count.reindex(frame.index), tv_count.reindex(frame.index))
                            frame["regular_turnover_count15"] = tv_count.reindex(frame.index)
                            frame["regular_window_end"] = sessions[end_i]
                            frame["regular_window_basis"] = "committee_previous_session_conservative_time_unknown"
    meta = dict(event=event, signal_date=observation_t, information_asof=t, observation_date=observation_t, cutoff_date=sessions[cutoff], signal_index=signal_i,
                cutoff_index=cutoff, lead_sessions=max(0, cutoff - signal_i), after_cutoff_monitoring=signal_i > cutoff, days_after_cutoff=max(0,signal_i-cutoff), profile=asdict(profile),
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
                 strategy_a_date_hook_used=route_hook is not None,
                 forward_record_created_before_outcome=False,
                 review_sums_end_date=sessions[review_observed_i],
                 unverified_regular_listing_age_exception_codes=list(age_exception.index[age_exception]),
                 liquidation_trading_status_history_verified=False,
                announced_pending_actions=pending_actions,
                known_unresolved_corporate_actions=unresolved_actions,
                complete_past_required=True, future_prices_used=False,
                historical_full_rule_reconstruction=False)
    return ExtensionSnapshot(frame, profile, meta)

