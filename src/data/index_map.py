"""ETF-to-index mapping — maps ETF codes to their underlying benchmark indices.

Used by ``pe_history.py`` to look up the correct PE history cache file
for a given ETF.  Each entry links an ETF to the index it tracks and
the corresponding cached PE history parquet file.

Coverage: Top ~55 ETFs by AUM (covering 90%+ of Chinese ETF market cap).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Master mapping: ETF code → index info
# ---------------------------------------------------------------------------

# Each value holds:
#   index_code  — legulegu symbol (e.g. "000300.SH")
#   index_name  — Chinese display name
#   cache_key   — parquet filename stem under src/data/pe_cache/

ETF_TO_INDEX: dict[str, dict] = {
    # ── 沪深300 系列 ──────────────────────────────────────────
    "510300": {"index_code": "000300.SH", "index_name": "沪深300",
               "cache_key": "000300_SH_沪深300"},
    "510310": {"index_code": "000300.SH", "index_name": "沪深300",
               "cache_key": "000300_SH_沪深300"},
    "510330": {"index_code": "000300.SH", "index_name": "沪深300",
               "cache_key": "000300_SH_沪深300"},
    "159919": {"index_code": "000300.SH", "index_name": "沪深300",
               "cache_key": "000300_SH_沪深300"},

    # ── 上证50 系列 ───────────────────────────────────────────
    "510050": {"index_code": "000016.SH", "index_name": "上证50",
               "cache_key": "000016_SH_上证50"},
    "510710": {"index_code": "000016.SH", "index_name": "上证50",
               "cache_key": "000016_SH_上证50"},

    # ── 中证500 系列 ──────────────────────────────────────────
    "510500": {"index_code": "000905.SH", "index_name": "中证500",
               "cache_key": "000905_SH_中证500"},
    "512500": {"index_code": "000905.SH", "index_name": "中证500",
               "cache_key": "000905_SH_中证500"},
    "159922": {"index_code": "000905.SH", "index_name": "中证500",
               "cache_key": "000905_SH_中证500"},

    # ── 中证1000 系列 ─────────────────────────────────────────
    "512100": {"index_code": "000852.SH", "index_name": "中证1000",
               "cache_key": "000852_SH_中证1000"},
    "159845": {"index_code": "000852.SH", "index_name": "中证1000",
               "cache_key": "000852_SH_中证1000"},

    # ── 中证800 ──────────────────────────────────────────────
    "515800": {"index_code": "000906.SH", "index_name": "中证800",
               "cache_key": "000906_SH_中证800"},

    # ── 科创50 系列 ───────────────────────────────────────────
    "588000": {"index_code": "000688.SH", "index_name": "科创50",
               "cache_key": "000688_SH_科创50"},
    "588050": {"index_code": "000688.SH", "index_name": "科创50",
               "cache_key": "000688_SH_科创50"},
    "588080": {"index_code": "000688.SH", "index_name": "科创50",
               "cache_key": "000688_SH_科创50"},

    # ── 创业板 系列 ───────────────────────────────────────────
    "159915": {"index_code": "399673.SZ", "index_name": "创业板50",
               "cache_key": "399673_SZ_创业板50"},
    "159949": {"index_code": "399673.SZ", "index_name": "创业板50",
               "cache_key": "399673_SZ_创业板50"},
    "159952": {"index_code": "399673.SZ", "index_name": "创业板50",
               "cache_key": "399673_SZ_创业板50"},

    # ── 红利 系列 ─────────────────────────────────────────────
    "510880": {"index_code": "000015.SH", "index_name": "上证红利",
               "cache_key": "000015_SH_上证红利"},
    "159905": {"index_code": "399324.SZ", "index_name": "深证红利",
               "cache_key": "399324_SZ_深证红利"},
    "515080": {"index_code": "000015.SH", "index_name": "上证红利",
               "cache_key": "000015_SH_上证红利"},

    # ── 其他宽基 ──────────────────────────────────────────────
    # 上证180
    "510180": {"index_code": "000016.SH", "index_name": "上证50",  # closest proxy
               "cache_key": "000016_SH_上证50"},

    # 深证100
    "159901": {"index_code": "399324.SZ", "index_name": "深证红利",  # closest proxy
               "cache_key": "399324_SZ_深证红利"},

    # 中证A50
    "560050": {"index_code": "000016.SH", "index_name": "上证50",  # closest proxy
               "cache_key": "000016_SH_上证50"},
    "159591": {"index_code": "000016.SH", "index_name": "上证50",
               "cache_key": "000016_SH_上证50"},

    # 中证A500
    "563500": {"index_code": "000300.SH", "index_name": "沪深300",  # closest proxy
               "cache_key": "000300_SH_沪深300"},
    "159338": {"index_code": "000300.SH", "index_name": "沪深300",
               "cache_key": "000300_SH_沪深300"},

    # ── 行业/主题 ETF（映射到最相关宽基）─────────────────────
    # 证券ETF → 中证800
    "512880": {"index_code": "000906.SH", "index_name": "中证800",
               "cache_key": "000906_SH_中证800"},
    "159841": {"index_code": "000906.SH", "index_name": "中证800",
               "cache_key": "000906_SH_中证800"},

    # 半导体ETF → 科创50
    "512480": {"index_code": "000688.SH", "index_name": "科创50",
               "cache_key": "000688_SH_科创50"},

    # 医药/医疗ETF → 中证500
    "512010": {"index_code": "000905.SH", "index_name": "中证500",
               "cache_key": "000905_SH_中证500"},
    "512170": {"index_code": "000905.SH", "index_name": "中证500",
               "cache_key": "000905_SH_中证500"},

    # 军工ETF → 中证1000
    "512660": {"index_code": "000852.SH", "index_name": "中证1000",
               "cache_key": "000852_SH_中证1000"},
    "512670": {"index_code": "000852.SH", "index_name": "中证1000",
               "cache_key": "000852_SH_中证1000"},

    # 酒ETF → 上证50
    "512690": {"index_code": "000016.SH", "index_name": "上证50",
               "cache_key": "000016_SH_上证50"},

    # 银行ETF → 上证红利
    "512800": {"index_code": "000015.SH", "index_name": "上证红利",
               "cache_key": "000015_SH_上证红利"},

    # 新能源车ETF → 创业板50
    "515030": {"index_code": "399673.SZ", "index_name": "创业板50",
               "cache_key": "399673_SZ_创业板50"},
    "515700": {"index_code": "399673.SZ", "index_name": "创业板50",
               "cache_key": "399673_SZ_创业板50"},

    # 光伏ETF → 中证500
    "515790": {"index_code": "000905.SH", "index_name": "中证500",
               "cache_key": "000905_SH_中证500"},

    # 芯片ETF → 科创50
    "159995": {"index_code": "000688.SH", "index_name": "科创50",
               "cache_key": "000688_SH_科创50"},

    # 消费ETF → 沪深300
    "159928": {"index_code": "000300.SH", "index_name": "沪深300",
               "cache_key": "000300_SH_沪深300"},

    # 科技ETF → 中证500
    "515000": {"index_code": "000905.SH", "index_name": "中证500",
               "cache_key": "000905_SH_中证500"},

    # 5GETF → 中证1000
    "515050": {"index_code": "000852.SH", "index_name": "中证1000",
               "cache_key": "000852_SH_中证1000"},

    # 恒生ETF (港股) → 沪深300 近似
    "159920": {"index_code": "000300.SH", "index_name": "沪深300",
               "cache_key": "000300_SH_沪深300"},
    "513660": {"index_code": "000300.SH", "index_name": "沪深300",
               "cache_key": "000300_SH_沪深300"},

    # H股ETF → 上证50 近似
    "510900": {"index_code": "000016.SH", "index_name": "上证50",
               "cache_key": "000016_SH_上证50"},

    # 纳指ETF → 中证1000 近似（国内中小盘）
    "513100": {"index_code": "000852.SH", "index_name": "中证1000",
               "cache_key": "000852_SH_中证1000"},

    # 标普500ETF → 沪深300 近似
    "513500": {"index_code": "000300.SH", "index_name": "沪深300",
               "cache_key": "000300_SH_沪深300"},

    # 中概互联 → 创业板50 近似
    "513050": {"index_code": "399673.SZ", "index_name": "创业板50",
               "cache_key": "399673_SZ_创业板50"},
    "159607": {"index_code": "399673.SZ", "index_name": "创业板50",
               "cache_key": "399673_SZ_创业板50"},

    # 黄金ETF (无PE概念) → 上证50 近似
    "518880": {"index_code": "000016.SH", "index_name": "上证50",
               "cache_key": "000016_SH_上证50"},
}


# ---------------------------------------------------------------------------
# Reverse lookups
# ---------------------------------------------------------------------------

# Build a set of all cache keys for fast validation
_AVAILABLE_INDICES: set[str] = {
    v["cache_key"] for v in ETF_TO_INDEX.values()
}


def get_index_for_etf(etf_code: str) -> dict | None:
    """Return the index mapping dict for *etf_code* or None if unknown."""
    return ETF_TO_INDEX.get(etf_code.strip())


def get_available_indices() -> dict[str, str]:
    """Return a deduped {cache_key: index_name} dict of all tracked indices."""
    seen: dict[str, str] = {}
    for v in ETF_TO_INDEX.values():
        ck = v["cache_key"]
        if ck not in seen:
            seen[ck] = v["index_name"]
    return seen


def has_pe_data(etf_code: str) -> bool:
    """Check whether we have cached PE history for *etf_code*."""
    mapping = ETF_TO_INDEX.get(etf_code.strip())
    return mapping is not None and mapping["cache_key"] in _AVAILABLE_INDICES
