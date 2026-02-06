import pandas as pd
import re
from urllib.parse import urlparse, ParseResult
import ipaddress
import tldextract
import math
from collections import Counter

_VALID_SCHEMES = ("http://", "https://", "ftp://", "ftps://", "mailto:", "ws://", "wss://")
SHORTENERS = {
    "bit.ly","t.co","tinyurl.com","goo.gl","ow.ly","is.gd","buff.ly",
    "rebrand.ly","cutt.ly","rb.gy","lnkd.in","s.id","shrtco.de","v.gd"
}
_SPECIAL_RE = re.compile(r"[@%=&?/:\-_]")

# -----------------------
# Helpers
# -----------------------
def _normalize_for_parse(url: str) -> str:
    u = url.strip()
    if not u.lower().startswith(_VALID_SCHEMES):
        u = "http://" + u
    return u

def _parse_url(url: str) -> ParseResult:
    return urlparse(_normalize_for_parse(url))

def get_sld(url: str) -> str:
    host = get_host(url)
    if not host:
        return ""
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    ext = tldextract.extract(host)
    return ext.domain or ""

def get_tld(url: str) -> str:
    host = get_host(url)
    if not host:
        return ""
    try:
        ipaddress.ip_address(host)
        return "[ipaddress]"
    except ValueError:
        pass
    ext = tldextract.extract(host)
    return ext.suffix or ""

def get_host(url: str) -> str:
    p = _parse_url(url)
    host = p.hostname
    if p.scheme == "mailto" and "@" in p.path:
        host = p.path.split("@", 1)[1]
    return host.strip("[]").lower() if host else ""

def has_ip_address(text: str) -> bool:
    ip4_pattern = r"(?:(?:[0-9]){1,3}\.){3}(?:[0-9]){1,3}"
    ip6_pattern = r"(?:(?:[0-9A-Fa-f]{1,4})(?:::|:)){1,7}(?:[0-9A-Fa-f]{1,4}|:)"
    ip_pattern = f"{ip4_pattern}|{ip6_pattern}"
    return bool(re.search(ip_pattern, text))

def is_ip(string: str) -> bool:
    """True if the given SLD string is an IPv4 or IPv6 literal."""
    if not string:
        return False
    try:
        ipaddress.ip_address(string)
        return True
    except ValueError:
        return False

def has_www(host: str) -> bool:
    return bool(host) and host.startswith("www.")

def is_shortener(host: str) -> bool:
    return host in SHORTENERS

# simple parsed accessors
def scheme_is_https_from_parsed(p: ParseResult) -> bool:
    return p.scheme.lower() == "https"

def length_path_from_parsed(p: ParseResult) -> int:
    return len(p.path)

def length_query_from_parsed(p: ParseResult) -> int:
    return len(p.query)

def path_depth_from_parsed(p: ParseResult) -> int:
    return sum(1 for seg in p.path.split("/") if seg)

def has_query_from_parsed(p: ParseResult) -> bool:
    return bool(p.query)

def has_fragment_from_parsed(p: ParseResult) -> bool:
    return bool(p.fragment)

# extra helpers
def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    cnt = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in cnt.values())

def file_ext_from_path(path: str):
    if not path:
        return False, ""
    seg = path.rsplit("/", 1)[-1]
    if "." in seg and not seg.startswith("."):
        ext = seg.rsplit(".", 1)[-1][:10].lower()
        return True, ext
    return False, ""

def path_tokens(path: str):
    return [seg for seg in path.split("/") if seg]

def query_keys(query: str):
    if not query:
        return []
    keys = []
    for kv in query.split("&"):
        if not kv:
            continue
        key = kv.split("=", 1)[0]
        if key:
            keys.append(key.lower())
    return keys

def unicode_flag(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)

def punycode_flag(host: str) -> bool:
    return bool(host) and ("xn--" in host)

def host_labels_stats(host: str):
    if not host:
        return 0, 0, False, False
    labels = host.split(".")
    label_count = len(labels)
    subdomain_count = max(0, label_count - 2)
    has_digit = any(any(ch.isdigit() for ch in lbl) for lbl in labels)
    has_hyphen = any("-" in lbl for lbl in labels)
    return label_count, subdomain_count, has_digit, has_hyphen

def path_stats(path: str):
    toks = path_tokens(path)
    tok_count = len(toks)
    avg_len = (sum(map(len, toks)) / tok_count) if tok_count > 0 else 0.0
    has_dotdot = any(tok == ".." for tok in toks)
    has_repeat_sep = "//" in path
    return tok_count, avg_len, has_dotdot, has_repeat_sep

def url_has_credentials(p: ParseResult) -> bool:
    return "@" in p.netloc.split("/")[0] if p.netloc else False

def get_port_bucket_from_parsed(p: ParseResult):
    port_present = False
    port_num = -1
    try:
        if p.port is not None:
            port_present = True
            port_num = int(p.port)
    except ValueError:
        port_present = False
        port_num = -1
    bucket = "none"
    if port_present:
        if port_num == 80:
            bucket = "80"
        elif port_num == 443:
            bucket = "443"
        else:
            bucket = "other"
    return port_num, bucket, port_present

# -----------------------
# Easy features (~15)
# -----------------------
def get_easy_url_features(urls: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame()
    urls = urls.astype(str)
    parsed = urls.apply(_parse_url)
    hosts = urls.apply(get_host)

    slds = urls.apply(get_sld)
    out["scheme_is_https"] = parsed.apply(scheme_is_https_from_parsed)
    out["has_www"] = hosts.apply(has_www)
    out["host_is_ip"] = slds.apply(is_ip)
    out["is_shortener"] = hosts.apply(is_shortener)

    out["length_total"] = urls.str.len()
    out["length_host"] = hosts.str.len()
    out["length_path"] = parsed.apply(length_path_from_parsed)
    out["length_query"] = parsed.apply(length_query_from_parsed)
    out["path_depth"] = parsed.apply(path_depth_from_parsed)

    out["has_query"] = parsed.apply(has_query_from_parsed)
    out["has_fragment"] = parsed.apply(has_fragment_from_parsed)

    out["digit_ratio"] = urls.str.count(r"\d") / out["length_total"]
    out["hyphen_ratio"] = urls.str.count("-") / out["length_total"]
    out["percent_encoded_ratio"] = urls.str.count("%") / out["length_total"]

    return out

# -----------------------
# All features (~40)
# -----------------------
def get_all_url_features(urls: pd.Series) -> pd.Series:
    out = get_easy_url_features(urls)

    urls = urls.astype(str)
    parsed = urls.apply(_parse_url)
    hosts = urls.apply(get_host)

    # Character class ratios
    out["alpha_ratio"] = urls.str.count(r"[A-Za-z]") / out["length_total"]
    out["lower_ratio"] = urls.str.count(r"[a-z]") / out["length_total"]
    out["upper_ratio"] = urls.str.count(r"[A-Z]") / out["length_total"]
    out["hex_ratio"]   = urls.str.count(r"[A-Fa-f0-9]") / out["length_total"]
    out["special_ratio"] = urls.str.count(_SPECIAL_RE) / out["length_total"]
    out["underscore_ratio"] = urls.str.count("_") / out["length_total"]

    # Unicode / punycode flags
    out["has_unicode"] = urls.apply(unicode_flag)
    out["has_punycode"] = hosts.apply(punycode_flag)

    # Host label stats
    host_stats = hosts.apply(host_labels_stats)
    out["host_label_count"] = host_stats.apply(lambda t: t[0])
    out["subdomain_count"]  = host_stats.apply(lambda t: t[1])
    out["host_has_digit"]   = host_stats.apply(lambda t: t[2])
    out["host_has_hyphen"]  = host_stats.apply(lambda t: t[3])
    out["host_char_entropy"] = hosts.apply(char_entropy)

    # Path stats
    path_vals = parsed.apply(lambda p: p.path)
    path_stat_vals = path_vals.apply(path_stats)
    out["path_token_count"]   = path_stat_vals.apply(lambda t: t[0])
    out["path_avg_token_len"] = path_stat_vals.apply(lambda t: t[1])
    out["path_has_dotdot"]    = path_stat_vals.apply(lambda t: t[2])
    out["path_has_repeat_sep"]= path_stat_vals.apply(lambda t: t[3])

    # File extension
    file_ext_vals = path_vals.apply(file_ext_from_path)
    out["has_file_ext"] = file_ext_vals.apply(lambda t: t[0])
    out["file_ext"]     = file_ext_vals.apply(lambda t: t[1])

    # Query stats
    query_vals = parsed.apply(lambda p: p.query)
    qkeys = query_vals.apply(query_keys)
    out["query_key_count"]        = qkeys.apply(len)
    out["query_unique_key_count"] = qkeys.apply(lambda keys: len(set(keys)))
    out["query_has_sessionish"]   = qkeys.apply(lambda ks: any(k for k in ks if any(sub in k for sub in ("sess", "token", "auth", "access", "bearer"))))

    # Fragment length
    out["fragment_length"] = parsed.apply(lambda p: len(p.fragment))

    # Credentials in URL
    out["has_credentials"] = parsed.apply(url_has_credentials)

    # Port features
    port_vals = parsed.apply(get_port_bucket_from_parsed)
    out["port_number"]   = port_vals.apply(lambda t: t[0])
    out["port_bucket"]   = port_vals.apply(lambda t: t[1])
    out["port_present"]  = port_vals.apply(lambda t: t[2])

    # Component proportions
    out["host_prop"]  = out["length_host"]  / out["length_total"]
    out["path_prop"]  = out["length_path"]  / out["length_total"]
    out["query_prop"] = out["length_query"] / out["length_total"]

    # Simple counts
    out["dot_count"]   = urls.str.count(r"\.")
    out["slash_count"] = urls.str.count(r"/")
    out["at_count"]    = urls.str.count(r"@")

    # URL character entropy
    out["url_char_entropy"] = urls.apply(char_entropy)

    return out
