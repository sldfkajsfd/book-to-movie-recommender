"""Conservative movie edition aliases built from the existing metadata."""
import csv
from collections import defaultdict
import re
import unicodedata

from .data import iter_metadata

FORMAT = r"(?:blu[ -]?ray|bd|dvd|vhs|4k(?: ultra hd)?|ultra hd|uhd|digital(?: copy)?|dhd)"
EDITION = (
    rf"(?:{FORMAT}(?:\s*\+\s*{FORMAT})*|"
    r"(?:unrated|rated|theatrical|extended|widescreen|full[ -]?screen|"
    r"special|deluxe|collector(?:'s)?|limited|anniversary|remastered)"
    r"(?:\s+(?:edition|version|cut))?)"
)
SUFFIX = re.compile(rf"\s*(?:\[{EDITION}\]|\({EDITION}\)|\s+-\s+{EDITION})\s*$")
PREVIEW = re.compile(r"^(?:official\s+)?(?:trailer|teaser|preview)\b|(?:\(|\[)(?:trailer|teaser|preview)(?:\)|\])$")
WORDS = re.compile(r"[^\W_]+")
YEAR = re.compile(r"(?:\(|\[)((?:19|20)\d{2})(?:\)|\])")


def title_key(title):
    original = unicodedata.normalize("NFKC", title).casefold()
    original = original.translate(str.maketrans({"’": "'", "–": "-", "—": "-"})).strip()
    base = original
    while True:
        stripped = SUFFIX.sub("", base).strip()
        if stripped == base:
            break
        base = stripped
    key = " ".join(re.sub(r"\bgiftset\b", "gift set", base).split())
    return key, base != original, bool(PREVIEW.search(original)), frozenset(YEAR.findall(original))


def description_evidence(left, right):
    """Require substantial shared prose; title matches alone are insufficient."""
    a, b = left["tokens"], right["tokens"]
    if min(len(a), len(b)) < 20:
        return None
    if a == b:
        return "same_description"
    if left["years"] and right["years"] and left["years"].isdisjoint(right["years"]):
        return None
    shorter = min(len(left["words"]), len(right["words"]))
    if len(left["words"] & right["words"]) < max(12, .45 * shorter):
        return None
    if left["phrases"] & right["phrases"]:
        return "shared_description_passage"
    return None


def build_aliases(path):
    titles = {}
    grouped = defaultdict(list)
    for row in iter_metadata(path):
        work, title = row["work_id"], row["title"]
        if work in titles:
            raise ValueError(f"Duplicate source work ID: {work}")
        titles[work] = title
        key, edition, preview, years = title_key(title)
        if key and not preview:
            grouped[key].append((work, edition, years))
    candidates = {work for group in grouped.values() if len(group) > 1 for work, _, _ in group}
    evidence = {}
    for row in iter_metadata(path):
        work = row["work_id"]
        if work in candidates:
            tokens = WORDS.findall(unicodedata.normalize("NFKC", row["content_text"]).casefold())[:256]
            evidence[work] = {"tokens": tokens, "words": set(tokens),
                "phrases": {tuple(tokens[i:i+12]) for i in range(len(tokens)-11)},
                "years": title_key(row["title"])[3]}
    aliases = {work: work for work in titles}
    reasons = {}
    for group in grouped.values():
        if len(group) < 2:
            continue
        clusters = []
        for work, _, _ in sorted(group, key=lambda entry: (entry[1], entry[0])):
            for cluster in clusters:
                proof = [description_evidence(evidence[work], evidence[other]) for other in cluster]
                if all(proof):
                    aliases[work] = cluster[0]
                    reasons[work] = proof[0]
                    cluster.append(work)
                    break
            else:
                clusters.append([work])
    return aliases, titles, reasons


def write_aliases(path, aliases, titles, reasons):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["work_id", "canonical_work_id", "title", "canonical_title", "merge_evidence"])
        for work, canonical in aliases.items():
            writer.writerow([work, canonical, titles[work], titles[canonical], reasons.get(work, "")])
