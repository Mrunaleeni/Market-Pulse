"""
events.py
---------
Tags each headline with one of 8 event categories using a two-pass approach:

  Pass 1 — Keyword / regex rules (fast, deterministic, ~80% coverage)
  Pass 2 — Zero-shot classifier (facebook/bart-large-mnli) for headlines
            that the rules didn't match. Lazy-loaded; skipped when
            USE_ZERO_SHOT=false (default in CI) to avoid the ~1.6 GB download.

Event categories
----------------
  company_results    : earnings beat/miss, guidance, dividend, buyback
  corporate_action   : M&A, stake sale, block deal, fundraise, demerger
  management         : CEO/CFO change, promoter pledge, resignation
  analyst_action     : rating upgrade/downgrade, target price
  regulatory_legal   : RBI/SEBI action, penalty, tax notice, fraud
  business_news      : order win, product launch, plant shutdown, strike
  sector_policy      : tariff, subsidy/PLI, drug price, fuel price
  macro              : RBI rate, crude, rupee, FII, budget
  unclassified       : nothing matched (rare after the two passes)

Usage
-----
  from src.events import tag_event_type

  tag_event_type("Infosys beats Q1 estimates, raises guidance")
  # → "company_results"

  # Batch (faster for DataFrames)
  from src.events import tag_events_batch
  df["Event_Type"] = tag_events_batch(df["Headline"].tolist())
"""

import re
import os
import logging
from typing import Sequence

log = logging.getLogger("events")

# ── Env flag ──────────────────────────────────────────────────────────────────
# Set USE_ZERO_SHOT=true in the environment to enable the BART fallback.
# Defaults to false so CI stays lean.
_USE_ZERO_SHOT = os.environ.get("USE_ZERO_SHOT", "false").lower() == "true"

# ── Rule table ────────────────────────────────────────────────────────────────
# Each entry: (category_name, [list of regex patterns])
# Patterns are matched case-insensitively against the full headline.
# Order matters — first match wins, so put more-specific rules first.

_RULES: list[tuple[str, list[str]]] = [
    # ── 1. Company results ────────────────────────────────────────────────────
    ("company_results", [
        r"\bearning[s]?\b",
        r"\bquarterly.{0,15}result",
        r"\bq[1-4]\s*(fy\d+|20\d\d)?\s*(result|profit|revenue|loss|pat|eps)",
        r"\bnet\s+profit\b",
        r"\brev(?:enue)?\s+(?:up|down|rose|fell|grew|declined)",
        r"\bebit(?:da)?\b",
        r"\bbeats?\s+(?:q[1-4]|quarterly|annual|fy\d+|20\d\d|estimate|forecast|expectation|consensus)",
        r"\bmisses?\s+(?:q[1-4]|quarterly|estimate|forecast|expectation|consensus)",
        r"\bguidance\s+(?:raised?|cut|lowered?|revised?|increased?|upgraded?)",
        r"\braises?\s+guidance",
        r"\bdividend\b",
        r"\bbuyback\b",
        r"\bshare\s+repurchase",
        r"\bbonus\s+share",
        r"\bright[s]?\s+issue",
        # Sales / volume results
        r"\bsales\s+(?:rise|rise|jump|surge|fall|decline|grow|drop).{0,20}(?:unit|lakh|crore|%|percent)",
        r"\bsales\s+(?:up|down)\s+\d",
        r"\btotal\s+sales\s+(?:rise|fall|jump|grow)",
        r"\bdispatches\b.{0,20}(?:rise|fall|up|down)",
        r"\bvolume[s]?\s+(?:rise|fall|up|down|grow|decline)",
        r"\bpat\s+(?:up|down|rise|fall|grow|decline)",
        r"\bq[1-4]\s*(?:fy)?\d*\s*(?:results?|profit|revenue|pat|eps|ebitda)",
        # Subscriber / market share data
        r"\bTRAI\b",
        r"\bmobile\s+subscriber\b",
        r"\bleads?\b.{0,20}(?:subscriber|user|market)",
        r"\bmarket\s+share\b",
        r"\bunit\s+sales\b",
        r"\bbusiness\s+update\b",
        r"\bgross\s+advance[s]?\b.{0,20}(?:rise|fall|up|down|grew|declined)",
        r"\badvance[s]?\b.{0,15}(?:rise|grew|fell|up|down|\d+%)",
        r"\bdeposit[s]?\b.{0,15}(?:rise|grew|fell|up|down|\d+%)",
        r"\b(?:Q[1-4]|quarterly)\s+biz\b",
        r"\bmobilis(?:es?|ed)\b.{0,30}(?:deposit|crore|lakh|billion|trillion)",
        r"\bdeposit\s+(?:mobilisation|growth|base)\b",
        r"\b(?:net\s+interest|NIM|NPA|GNPA|NNPA)\b",
    ]),

    # ── 2. Corporate actions ──────────────────────────────────────────────────
    ("corporate_action", [
        r"\bacquisition\b",
        r"\bacquires?\b",
        r"\bmerger\b",
        r"\bmerges?\b",
        r"\bdemerger\b",
        r"\bde-?merger\b",
        r"\bspin.?off\b",
        r"\bstake\s+sale\b",
        r"\bstake\s+(?:acquired?|sold|buys?|bought)\b",
        r"\bblock\s+deal\b",
        r"\bbulk\s+deal\b",
        r"\bopen\s+offer\b",
        r"\bfundraising?\b",
        r"\bQIP\b",
        r"\bIPO\b",
        r"\bpreferential\s+allotment\b",
        r"\bNCD\b",
        r"\bdebt\s+(?:raise[sd]?|issuance|placement)\b",
        r"\bjoint\s+venture\b",
        r"\bstrategic\s+(?:partnership|deal|alliance)\b",
        r"\btake-?over\b",
        # Stock exchange listings / market debuts
        r"\blists?\s+on\b",
        r"\bmarket\s+debut\b",
        r"\bstock\s+exchange\s+debut\b",
        r"\bLSE\b",
        r"\blondon\s+stock\s+exchange\b",
        r"\bdebuts?\s+on\b",
        r"\bfloat[s]?\s+on\b",
        r"\blondon\s+listing\b",
        r"\b(?:plans?|eyes?|aims?\s+for)\b.{0,20}\b(?:listing|ipo)\b",
        r"\blisting\s+drought\b",
        r"\bdebut[s]?\s+on\b",
        # ESOP / allotment
        r"\bESOP\b",
        r"\ballot(?:ment)?\s+of\s+(?:share|equity)",
        r"\bshares?\s+allot",
        # Fundraise with explicit amount
        r"\braises?\s+\$?\d",
        r"\blooks?\s+to\s+raise\b",
        r"\bseeks?\s+to\s+raise\b",
    ]),

    # ── 3. Management ─────────────────────────────────────────────────────────
    ("management", [
        r"\bCEO\b",
        r"\bCFO\b",
        r"\bCOO\b",
        r"\bMD\b(?!\s+(?:and|&)\s+CEO)",   # standalone MD (Managing Director)
        r"\b(?:managing|executive)\s+director\b",
        r"\bchairman\b",
        r"\bappoint(?:s|ed|ment)\b.{0,30}(?:CEO|CFO|COO|director|chief|head)",
        r"\bresign(?:s|ed|ation)\b",
        r"\bquits?\b.{0,15}(?:CEO|CFO|board|company|director)",
        r"\bsteps?\s+down\b",
        r"\bpromoter\s+pledge\b",
        r"\bpledge[sd]?\s+shares?\b",
        r"\binsider\s+(?:trading|buying|selling)\b",
        r"\bboard\s+(?:approves?|meets?|meeting|reshuffle)\b",
        r"\bappoints?\b.{0,40}(?:additional|independent|non-executive|wholetime)\b",
        r"\bappoints?\b.{0,40}(?:director|president|head|chief|officer)\b",
        r"\bnamed\b.{0,30}(?:CEO|CFO|COO|director|head|chief|president)\b",
        r"\bjoins?\s+as\s+(?:CEO|CFO|COO|director|head|chief|president)\b",
        r"\btakes?\s+charge\s+as\b",
        r"\bnew\s+(?:CEO|CFO|COO|MD|chairman|chief|director)\b",
        r"\bnames?\b.{0,40}(?:as|to\s+be)\b.{0,30}(?:chief|head|director|officer|president|CEO|CFO|COO|CMD)\b",
        r"\bto\s+head\b.{0,20}(?:bank|company|firm|group|division)",
        r"\bat\s+helm\b",
        r"\bat\s+the\s+helm\b",
        r"\blifts?\s+overhang\b",
        r"\bappoints?\b.{0,30}(?:as\s+)?(?:chief|head|officer|president|CMD)\b",
        r"\bretires?\s+after\b",
        r"\bsuccession\b",
    ]),

    # ── 4. Analyst actions ────────────────────────────────────────────────────
    ("analyst_action", [
        r"\bupgrade[sd]?\b.{0,20}(?:buy|hold|outperform|accumulate|overweight)",
        r"\bdowngrade[sd]?\b.{0,20}(?:sell|hold|underperform|reduce|underweight)",
        r"\brating\b.{0,20}(?:upgrade|downgrade|change|cut|raised?)",
        r"\btarget\s+price\b",
        r"\bprice\s+target\b",
        r"\bTP\s+(?:raised?|cut|lowered?|revised?)",
        r"\binitiat(?:es?|ing|ed)\s+coverage\b",
        r"\bbuy\s+(?:call|rating|recommend)",
        r"\bsell\s+(?:call|rating|recommend)",
        r"\bhold\s+(?:rating|recommend)",
        r"\boutperform\b",
        r"\bunderperform\b",
        r"\boverweight\b",
        r"\bunderweight\b",
        r"\bneutral\s+(?:rating|call|recommend)\b",
        r"\banalyst[s]?\s+(?:see|expect|raise|cut|upgrade|downgrade)\b",
        r"\bbrokerage\b",
        # "prefers X" / "bullish on X" style analyst calls
        r"\bprefers?\b.{0,30}(?:in|over|stock|sector)",
        r"\bbullish\s+on\b",
        r"\bbearish\s+on\b",
        r"\breiterates?\s+(?:buy|sell|hold|target)",
        r"\b(?:buy|sell|hold)\s+rating\b",
        r"\binitiates?\s+(?:buy|sell|hold|coverage)\b",
        r"\b(?:maintains?|retains?)\s+(?:buy|sell|hold|outperform|underperform)\b",
    ]),

    # ── 5. Regulatory and legal ───────────────────────────────────────────────
    ("regulatory_legal", [
        r"\bSEBI\b",
        r"\bRBI\b.{0,20}(?:order|notice|penalty|action|ban|probe|directive)",
        r"\bCCI\b",                       # Competition Commission of India
        r"\bNCLT\b",
        r"\bNDPS\b",
        r"\bEnforcement\s+Directorate\b",
        r"\bED\s+(?:raid|probe|notice|arrest)\b",
        r"\bIncome\s+Tax\b.{0,20}(?:notice|raid|order|demand)",
        r"\bIT\s+(?:notice|raid|demand)\b",
        r"\bGST\s+(?:notice|demand|order|raid)\b",
        r"\bpenalt(?:y|ies)\b",
        r"\bfine[sd]?\b.{0,20}(?:crore|lakh|million|billion|rupee)",
        r"\bfraud\b",
        r"\bscam\b",
        r"\ballega(?:tion|ted)\b",
        r"\blawsuit\b",
        r"\blitigation\b",
        r"\bcourt\s+(?:order|ruling|judgment|hearing)\b",
        r"\bregulatory\s+(?:approval|action|ban|inquiry|probe)\b",
        r"\bFIPB\b",
        r"\bCBDT\b",
        r"\bsued\b",
        r"\bsecurities\s+law\s+violation",
        r"\bclass\s+action\b",
        r"\binvestigation\b.{0,20}(?:fraud|account|SEBI|CBI|ED)",
        r"\bforensic\s+audit\b",
    ]),

    # ── 6. Business news ──────────────────────────────────────────────────────
    ("business_news", [
        r"\border\s+win\b",
        r"\bwins?\b.{0,40}(?:order|contract|bid|deal|tender)",
        r"\bsecures?\b.{0,30}(?:order|contract|deal)\b",
        r"\bbagged?\b.{0,30}(?:order|contract|deal)\b",
        r"\bproduct\s+launch\b",
        r"\blaunch(?:es|ed|ing)?\b.{0,30}(?:product|platform|service|app|model|vehicle|drug|medicine|therapy|device|solution)",
        r"\bnew\s+(?:drug|medicine|therapy|treatment|product|platform|service|app|model|vehicle)\b",
        r"\bplant\s+(?:shutdown|closure|fire|explosion|outage)\b",
        r"\bfactory\s+(?:fire|shut|closure)\b",
        r"\bstrike\b.{0,20}(?:worker|employee|labour|union)",
        r"\bworker[s]?\s+strike\b",
        r"\brecall\b.{0,15}(?:product|vehicle|drug|batch)",
        r"\bexpansion\b.{0,20}(?:capex|capacity|plant|facility)",
        r"\bcapex\b",
        r"\bnew\s+(?:contract|facility|plant|capacity)\b",
        r"\bpartnership\s+(?:with|signs?)\b",
        r"\bMOU\b",
        # Rate cuts by a bank on its own products (MCLR, lending rate)
        r"\bMCLR\b",
        r"\blending\s+rate[s]?\s+(?:cut|reduce|lower|slash)",
        r"\bcuts?\s+(?:lending|repo|base)\s+rate",
        r"\brate\s+cut\b.{0,20}(?:basis\s+point|bps|%|percent)",
        # Capacity / production targets announced by the company
        r"\baims?\s+to\s+(?:sell|produce|manufacture|launch|expand)",
        r"\btargets?\s+(?:\d|million|lakh|crore).{0,30}(?:unit|vehicle|bed|capacity)",
        r"\bhalve[sd]?\s+(?:development|production|cost)",
        r"\bboost[s]?\s+(?:capacity|production|output|india)",
        # Tariff / price hike by the company
        r"\bhikes?\s+(?:tariff|price|rate|charge)",
        r"\braise[sd]?\s+(?:tariff|price|rate|charge)",
        r"\bpostpaid\s+(?:plan|tariff|price)",
        r"\bprepaid\s+(?:plan|tariff|price)",
        # Product bundles / plan changes
        r"\bbundle[sd]?\b",
        r"\badd[s]?\b.{0,20}(?:perk|bundle|service|benefit)",
        r"\benhanc(?:es?|ed)\b.{0,20}(?:plan|offer|service|feature)",
        r"\boffers?\s+(?:free|complementary|complimentary)\b",
        # Partnership / collaboration
        r"\bcollaborat(?:e[sd]?|ion)\b",
        r"\bteams?\s+up\s+with\b",
        r"\bties?\s+up\s+with\b",
        # Subscriber / user growth
        r"\bsubscriber\s+(?:addition|gain|growth|base)",
        r"\bmobile\s+(?:user|subscriber)\s+addition",
        r"\badds?\b.{0,20}(?:million|lakh|crore).{0,20}(?:user|subscriber|customer)",
        # Technology deployment
        r"\bAI.{0,15}(?:system|platform|tool|model)\b",
        r"\bdetect(?:s|ed)?\b.{0,20}(?:spam|fraud|call)",
        r"\bdeploy[s]?\b.{0,20}(?:AI|system|platform|technology)",
        # Payment system / fintech launch
        r"\bApple\s+Pay\b",
        r"\bpay\s+(?:launches?|debuts?|arrives?|rolls?\s+out)\b",
        r"\blaunch(?:es|ed|ing)?\s+in\s+india\b",
        r"\bnew\s+plans?\b",
        # Bank strike / work disruption
        r"\bbank\s+strike\b",
        r"\bwork\s+(?:disruption|stoppage)\b",
        # Hiring / campus recruitment (large-scale operational)
        r"\bcampus\s+(?:hire[s]?|recruit|placement)\b",
        r"\b\d[,\d]*\s+(?:campus|fresh)\s+hire[s]?\b",
    ]),

    # ── 7. Sector and policy ──────────────────────────────────────────────────
    ("sector_policy", [
        r"\btariff\b",
        r"\bimport\s+duty\b",
        r"\bexport\s+(?:ban|restriction|incentive)\b",
        r"\bsubsid(?:y|ies)\b",
        r"\bPLI\s+(?:scheme|incentive|outlay)\b",
        r"\bproduction[- ]linked\b",
        r"\bdrug\s+price\s+(?:control|cap|reduction)\b",
        r"\bNPPA\b",                       # National Pharmaceutical Pricing Authority
        r"\bfuel\s+price\b",
        r"\bpetrol\s+price\b",
        r"\bdiesel\s+price\b",
        r"\bLPG\s+price\b",
        r"\bnatural\s+gas\s+price\b",
        r"\bAPMC\b",
        r"\bFDI\s+(?:rule|norm|limit|policy|approval)\b",
        r"\bpolicy\s+(?:change|reform|announcement|review)\b",
        r"\bsector\s+(?:reform|boost|outlook)\b",
        r"\bgovernment\s+(?:approves?|plans?|announces?|orders?)\b.{0,30}(?:sector|industry|policy)",
    ]),

    # ── 8. Macro ──────────────────────────────────────────────────────────────
    ("macro", [
        r"\bRBI\b.{0,20}(?:rate|repo|policy|MPC|cut|hike|hold|pause)",
        r"\brepo\s+rate\b",
        r"\brate\s+(?:cut|hike|hold|pause)\b.{0,20}(?:RBI|Fed|MPC|central\s+bank)",
        r"\bcrude\b.{0,30}(?:price|fall|fell|rise|rose|drop|surge|plunge|high|low|gain)",
        r"\bcrude\s+oil\b",
        r"\boil\s+price\b",
        r"\brupee\b.{0,20}(?:fall|rise|depreciat|appreciat|weak|strong|hit)",
        r"\bINR\b.{0,20}(?:fall|rise|depreciat|appreciat|weak|strong)\b",
        r"\bFII\s+(?:selling|buying|outflow|inflow)\b",
        r"\bforeign\s+(?:outflow|inflow|sell|buy)\b",
        r"\bFPI\s+(?:selling|buying|outflow|inflow)\b",
        r"\bbudget\b.{0,15}(?:2\d\d\d|announce|proposal|boost|impact)",
        r"\bunion\s+budget\b",
        r"\binfla(?:tion|tionary)\b",
        r"\bGDP\s+(?:growth|data|number|forecast)\b",
        r"\bCPI\b",                        # Consumer Price Index
        r"\bWPI\b",                        # Wholesale Price Index
        r"\bNifty\b.{0,15}(?:crash|fall|plunge|rally|high|low)\b",
        r"\bSensex\b.{0,15}(?:crash|fall|plunge|rally|high|low)\b",
        r"\bmarket\s+(?:crash|rally|selloff|plunge|meltdown)\b",
    ]),
]

# Pre-compile all patterns once at import time
_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (cat, [re.compile(p, re.IGNORECASE) for p in patterns])
    for cat, patterns in _RULES
]


# ── Zero-shot fallback (lazy-loaded) ──────────────────────────────────────────
_zs_pipeline = None
_ZS_LABELS = [
    "company earnings or financial results",
    "corporate action such as merger acquisition or stake sale",
    "management change or executive appointment",
    "analyst rating or price target",
    "regulatory action legal or government penalty",
    "business news such as order win product launch or plant issue",
    "sector policy or government subsidy",
    "macroeconomic news such as interest rate crude oil or currency",
]
_ZS_LABEL_MAP = {
    "company earnings or financial results":          "company_results",
    "corporate action such as merger acquisition or stake sale": "corporate_action",
    "management change or executive appointment":     "management",
    "analyst rating or price target":                 "analyst_action",
    "regulatory action legal or government penalty":  "regulatory_legal",
    "business news such as order win product launch or plant issue": "business_news",
    "sector policy or government subsidy":            "sector_policy",
    "macroeconomic news such as interest rate crude oil or currency": "macro",
}


def _load_zs_pipeline():
    global _zs_pipeline
    if _zs_pipeline is not None:
        return _zs_pipeline
    try:
        from transformers import pipeline
        log.info("Loading zero-shot classifier (facebook/bart-large-mnli) …")
        _zs_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
        log.info("Zero-shot classifier ready.")
    except Exception as e:
        log.warning("Could not load zero-shot classifier: %s", e)
        _zs_pipeline = None
    return _zs_pipeline


# ── Core tagger ───────────────────────────────────────────────────────────────

def tag_event_type(headline: str, use_zero_shot: bool | None = None) -> str:
    """
    Tag a single headline with an event category.

    Parameters
    ----------
    headline      : the raw headline string
    use_zero_shot : override env flag (True = always try ZS, False = rules only)

    Returns
    -------
    One of the 8 category strings, or "unclassified".
    """
    if not headline or not isinstance(headline, str):
        return "unclassified"

    # Pass 1 — rules
    for category, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(headline):
                log.debug("Rule match: %s → %s", headline[:60], category)
                return category

    # Pass 2 — zero-shot (optional)
    _use_zs = _USE_ZERO_SHOT if use_zero_shot is None else use_zero_shot
    if _use_zs:
        pipe = _load_zs_pipeline()
        if pipe is not None:
            try:
                result = pipe(headline, candidate_labels=_ZS_LABELS)
                best   = result["labels"][0]
                score  = result["scores"][0]
                mapped = _ZS_LABEL_MAP.get(best, "unclassified")
                log.debug(
                    "Zero-shot: %s → %s (%.2f)", headline[:60], mapped, score
                )
                # Only accept if confidence is reasonable
                if score >= 0.40:
                    return mapped
            except Exception as e:
                log.warning("Zero-shot inference failed: %s", e)

    return "unclassified"


def tag_events_batch(headlines: Sequence[str],
                     use_zero_shot: bool | None = None) -> list[str]:
    """
    Tag a list of headlines.  Rules are applied one by one (already vectorised
    via list comprehension).  Zero-shot runs only on the unclassified subset to
    save inference time.

    Returns a list of category strings, same length as input.
    """
    results = [tag_event_type(h, use_zero_shot=False) for h in headlines]

    _use_zs = _USE_ZERO_SHOT if use_zero_shot is None else use_zero_shot
    if not _use_zs:
        return results

    # Zero-shot pass on the unclassified subset
    unclassified_idx = [i for i, r in enumerate(results) if r == "unclassified"]
    if not unclassified_idx:
        return results

    pipe = _load_zs_pipeline()
    if pipe is None:
        return results

    unclassified_headlines = [headlines[i] for i in unclassified_idx]
    log.info(
        "Running zero-shot on %d unclassified headlines …", len(unclassified_headlines)
    )
    try:
        zs_results = pipe(unclassified_headlines, candidate_labels=_ZS_LABELS,
                          batch_size=8)
        for idx, zsr in zip(unclassified_idx, zs_results):
            best   = zsr["labels"][0]
            score  = zsr["scores"][0]
            mapped = _ZS_LABEL_MAP.get(best, "unclassified")
            if score >= 0.40:
                results[idx] = mapped
    except Exception as e:
        log.warning("Batch zero-shot inference failed: %s", e)

    return results


# ── Convenience: tag an entire DataFrame ─────────────────────────────────────

def tag_dataframe(df, headline_col: str = "Headline",
                  out_col: str = "Event_Type",
                  use_zero_shot: bool | None = None):
    """
    Add an Event_Type column to a DataFrame in place and return it.

    Parameters
    ----------
    df            : pandas DataFrame with a headline column
    headline_col  : name of the column containing headline strings
    out_col       : name of the output column (default "Event_Type")
    use_zero_shot : override flag (default: read from USE_ZERO_SHOT env var)
    """
    if headline_col not in df.columns:
        raise KeyError(f"Column '{headline_col}' not found in DataFrame.")
    df = df.copy()
    df[out_col] = tag_events_batch(
        df[headline_col].tolist(), use_zero_shot=use_zero_shot
    )
    coverage = (df[out_col] != "unclassified").mean() * 100
    log.info(
        "Event tagging complete. Coverage: %.1f%% classified (n=%d).",
        coverage, len(df)
    )
    return df


# ── Human-readable labels ─────────────────────────────────────────────────────
EVENT_LABELS: dict[str, str] = {
    "company_results":  "Company Results",
    "corporate_action": "Corporate Action",
    "management":       "Management Change",
    "analyst_action":   "Analyst Action",
    "regulatory_legal": "Regulatory / Legal",
    "business_news":    "Business News",
    "sector_policy":    "Sector & Policy",
    "macro":            "Macro",
    "unclassified":     "Unclassified",
}

ALL_EVENT_TYPES: list[str] = list(EVENT_LABELS.keys())


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    test_headlines = [
        "Infosys beats Q2 estimates, raises FY26 guidance",
        "HDFC Bank CEO steps down, board appoints new MD",
        "SEBI levies Rs 5 crore penalty on Reliance Industries",
        "Tata Motors wins Rs 2,000 crore order for electric buses",
        "RBI holds repo rate at 6.25%, maintains accommodative stance",
        "Maruti Suzuki upgrades target price by Nomura to Rs 14,000",
        "SBI announces QIP to raise Rs 10,000 crore",
        "PLI scheme to boost pharma sector output by 30%",
        "Crude oil falls 3% on demand concerns",
        "Sun Pharma launches new oncology drug in US market",
    ]

    if len(sys.argv) > 1:
        test_headlines = sys.argv[1:]

    print(f"\n{'Headline':<65} Event Type")
    print("-" * 90)
    for h in test_headlines:
        tag = tag_event_type(h)
        label = EVENT_LABELS.get(tag, tag)
        print(f"{h[:64]:<65} {label}")
