#!/usr/bin/env python3
"""
The Real Windy City — City of Cheyenne Archive.org Speaker Video & Episode Publisher
====================================================================================
Processes City of Cheyenne government meetings from Internet Archive (`collection:cochwy`)
from NEWEST -> OLDEST, canonicalizes speaker names across YouTube ASR variants, falls back
to voice-activity + MFCC acoustic diarization when older (2012-2018) videos lack `.en.vtt`
captions, packages 1 `.zip` per speaker (each containing separate `.mp4` turn clips +
`_SPEAKER_INDEX.txt`), uploads to GitHub Releases, and commits both `SPEAKER_INDEX.md`,
`manifest.json`, and TV-Episode `episode.json` metadata.
"""

import argparse
import base64
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
from scipy.io import wavfile
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass


ARCHIVE_SEARCH_URL = (
    "https://archive.org/advancedsearch.php?"
    "q=collection:(cochwy)+AND+mediatype:(movies)"
    "&fl[]=identifier,title,date,publicdate,item_size"
    "&sort[]=date+desc&rows={rows}&page={page}&output=json"
)

MAX_RELEASE_ASSET_BYTES = 1_900_000_000

# Canonicalizer for recurring Cheyenne officials & staff across YouTube ASR homophones
CANONICAL_RULES = [
    (r"(rinne|rennie|renny|rainey|renie)", "Dr_Mark_Rinne_Council"),
    (r"(emmons|emmens|immens|nimens|edmonds|\bmins\b)", "Dr_Kathy_Emmons_Council"),
    (r"(laybourn|layborne|layborn|leybourn|leborn|labour|\blay\b)", "Councilman_Tom_Laybourn"),
    (r"(aldrich|aldridge)", "Michelle_Aldrich_Council"),
    (r"(mayor_collins|patrick_collins)", "Mayor_Patrick_Collins"),
    (r"(wolf|wolfe)", "Councilman_Larry_Wolf"),
    (r"(moody)", "Councilman_Ken_Moody"),
    (r"(esquibel|esabel|escoel)", "Councilman_Richard_Johnson_or_Esquibel"),
    (r"(segrave|crave|seagrave)", "Councilman_Pete_Laybourn_or_Segrave"),
    (r"(seth_lloyd|seth_white|sethway|subway_planning|\blloyd\b)", "Seth_Lloyd_Planning"),
    (r"(charles_bloom|\bbloom\b)", "Charles_Bloom_Planning_Director"),
    (r"(charles_miller|\bmiller\b)", "Charles_Miller"),
    (r"(mrs\.?_nais|tammy_nicely)", "Tammy_Nicely"),
    (r"(nicely|naisli|naisely)", "David_Nicely"),
    (r"(kristin_lee|kristen_lee|ms\.?_lee)", "Kristin_Lee_Attorney"),
    (r"(brody)", "John_Brody_City_Attorney"),
    (r"(hopkinson)", "David_Hopkinson_Deputy_City_Attorney"),
    (r"(chestnut)", "Dr_Lori_Chestnut"),
    (r"(dorsch)", "Dan_Dorsch_Habitat"),
    (r"(patrick_rudd|\brudd\b)", "Patrick_Rudd"),
    (r"(swain)", "Brandon_Swain"),
    (r"(kylie_soden|soden)", "Kylie_Soden"),
    (r"(cobb)", "Tom_Cobb_City_Engineer"),
    (r"(meeting_chair|mr\.?_chair|madam_chair)", "Meeting_Chair"),
    (r"(meeting_clerk)", "Meeting_Clerk"),
]


def canonicalize_speaker(raw_label: str) -> str:
    low = raw_label.lower()
    for pat, canon in CANONICAL_RULES:
        if re.search(pat, low):
            return canon
    return raw_label


def repo_for_year(owner: str, year: int) -> str:
    if year >= 2025:
        return f"{owner}/cheyenne-archives-2025-2026"
    if year >= 2023:
        return f"{owner}/cheyenne-archives-2023-2024"
    if year == 2022:
        return f"{owner}/cheyenne-archives-2022"
    if year >= 2018:
        return f"{owner}/cheyenne-archives-2018-2021"
    if year >= 2014:
        return f"{owner}/cheyenne-archives-2014-2017"
    return f"{owner}/cheyenne-archives-2008-2013"


def gh_request(method: str, url: str, token: str, data: bytes | dict | None = None, content_type: str = "application/json"):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "therealwindycity-speaker-publisher/2.0",
    }
    body = None
    if isinstance(data, dict):
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif isinstance(data, (bytes, bytearray)):
        body = data
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(body))

    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw.strip() else None
        except urllib.error.HTTPError as e:
            err_txt = e.read().decode("utf-8", errors="replace")
            if e.code == 404:
                raise
            if e.code in (409, 429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2.5 * (attempt + 1))
                continue
            raise RuntimeError(f"GitHub API {method} {url} -> HTTP {e.code}: {err_txt[:400]}")


def gh_ensure_release(full_repo: str, tag: str, name: str, body_md: str, token: str) -> dict:
    try:
        rel = gh_request("GET", f"https://api.github.com/repos/{full_repo}/releases/tags/{urllib.parse.quote(tag)}", token)
        return gh_request(
            "PATCH",
            f"https://api.github.com/repos/{full_repo}/releases/{rel['id']}",
            token,
            {"name": name, "body": body_md},
        )
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    return gh_request(
        "POST",
        f"https://api.github.com/repos/{full_repo}/releases",
        token,
        {"tag_name": tag, "name": name, "body": body_md, "draft": False, "prerelease": False},
    )


def gh_upload_release_asset(release: dict, file_path: str, asset_name: str, token: str) -> str:
    existing = {a["name"]: a for a in release.get("assets", [])}
    fsize = os.path.getsize(file_path)
    if asset_name in existing and existing[asset_name].get("size") == fsize:
        return existing[asset_name]["browser_download_url"]
    if asset_name in existing:
        gh_request("DELETE", existing[asset_name]["url"], token)

    upload_base = release["upload_url"].split("{")[0]
    upload_url = f"{upload_base}?name={urllib.parse.quote(asset_name)}"
    cmd = [
        "curl", "-sS", "--fail-with-body", "--retry", "3",
        "-X", "POST",
        "-H", f"Authorization: Bearer {token}",
        "-H", "Accept: application/vnd.github+json",
        "-H", "X-GitHub-Api-Version: 2022-11-28",
        "-H", "Content-Type: application/zip",
        "--data-binary", f"@{file_path}",
        upload_url,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)["browser_download_url"]


def gh_put_file(full_repo: str, repo_path: str, content_bytes: bytes, message: str, token: str) -> None:
    api_url = f"https://api.github.com/repos/{full_repo}/contents/{urllib.parse.quote(repo_path)}"
    for attempt in range(3):
        sha = None
        try:
            cur = gh_request("GET", api_url, token)
            if isinstance(cur, dict):
                sha = cur.get("sha")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        payload = {"message": message, "content": base64.b64encode(content_bytes).decode("ascii")}
        if sha:
            payload["sha"] = sha
        try:
            gh_request("PUT", api_url, token, payload)
            return
        except RuntimeError as e:
            if "409" in str(e) and attempt < 2:
                time.sleep(2)
                continue
            raise


def gh_get_json_file(full_repo: str, repo_path: str, token: str, default: dict) -> dict:
    api_url = f"https://api.github.com/repos/{full_repo}/contents/{urllib.parse.quote(repo_path)}"
    try:
        cur = gh_request("GET", api_url, token)
        if isinstance(cur, dict) and cur.get("content"):
            return json.loads(base64.b64decode(cur["content"]).decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return default
        raise
    return default


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "therealwindycity-speaker-publisher/2.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def range_download(url: str, dest: str, expected_size: int | None = None, workers: int = 16) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if expected_size and expected_size > 8 * 1024 * 1024:
        if os.path.exists(dest) and os.path.getsize(dest) == expected_size:
            return
        with open(dest, "wb") as f:
            f.truncate(expected_size)
        chunk_sz = (expected_size + workers - 1) // workers

        def dl_part(idx: int) -> None:
            b0 = idx * chunk_sz
            b1 = min(expected_size - 1, (idx + 1) * chunk_sz - 1)
            if b0 > b1:
                return
            for att in range(4):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={b0}-{b1}", "User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "r+b") as out:
                        out.seek(b0)
                        while True:
                            buf = r.read(262144)
                            if not buf:
                                break
                            out.write(buf)
                    return
                except Exception:
                    if att == 3:
                        raise
                    time.sleep(1.5 * (att + 1))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(dl_part, range(workers)))
    else:
        subprocess.run(["curl", "-sL", "--retry", "3", url, "-o", dest], check=True)


def ts_to_sec(s: str) -> float:
    parts = s.replace(",", ".").split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return int(parts[0]) * 60 + float(parts[1])


def fmt_hms(sec: float) -> str:
    s = max(0, int(round(sec)))
    return f"{s // 3600:02d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def sanitize_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
    return re.sub(r"_+", "_", s).strip("_")[:60]


def parse_vtt_turns(vtt_path: str) -> list[list]:
    with open(vtt_path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    events = []
    for block in re.split(r"\n\s*\n", raw):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        t_idx = next((i for i, ln in enumerate(lines[:3]) if "-->" in ln), -1)
        if t_idx < 0:
            continue
        m = re.search(r"(\d+:\d+:\d+[.,]\d+)\s*-->\s*(\d+:\d+:\d+[.,]\d+)", lines[t_idx])
        if not m:
            continue
        st, en = ts_to_sec(m.group(1)), ts_to_sec(m.group(2))
        if en - st < 0.05:
            continue
        text_lines = lines[t_idx + 1 :]
        if text_lines:
            last = text_lines[-1]
            wm = re.findall(r"<(\d+:\d+:\d+[.,]\d+)>", last)
            clean = html.unescape(re.sub(r"<[^>]+>", "", last)).strip()
            if clean:
                events.append((st, en, clean, ts_to_sec(wm[-1]) if wm else en))

    turns = []
    for st, en, clean, last_w in events:
        parts = clean.split(">>")
        for idx, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            is_new = (idx > 0) or clean.startswith(">>")
            if is_new or not turns:
                if turns and turns[-1][1] > st:
                    turns[-1][1] = st
                turns.append([st, max(en, last_w), part])
            else:
                turns[-1][1] = max(turns[-1][1], en, last_w)
                turns[-1][2] += " " + part
    return turns


def vad_fallback_turns(wav_path: str) -> list[list]:
    """Acoustic voice-activity turn segmentation for older uncaptioned (2012-2018) videos."""
    sr, audio = wavfile.read(wav_path)
    y = audio.astype(np.float32) / 32768.0
    intervals = librosa.effects.split(y, top_db=28, frame_length=4096, hop_length=1024)
    turns = []
    for i0, i1 in intervals:
        st, en = i0 / sr, i1 / sr
        if turns and (st - turns[-1][1]) < 1.2 and (en - turns[-1][0]) < 45.0:
            turns[-1][1] = en
        elif (en - st) >= 1.5:
            turns.append([st, en, "[Uncaptioned archival audio segment — acoustically diarized]"])
    return turns


def extract_speaker_hint(text: str, prev_text: str) -> str | None:
    tl = text.lower()
    patterns = [
        r"my name is ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"this is ((?:Dr\.|Mr\.|Ms\.|Councilman|Councilwoman)?\s*[A-Z][a-z]+ [A-Z][a-z]+)",
        r"^(?:Madam Chair(?:woman|man)?,?\s*)?([A-Z][a-z]+ [A-Z][a-z]+),\s*(?:city|planning|deputy|ward|executive|public)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return canonicalize_speaker(sanitize_name(m.group(1)))
    if re.match(r"^(item\s+|number\s+\d+)", tl):
        return "Meeting_Clerk"
    if any(
        p in tl
        for p in [
            "any questions for staff",
            "hearing or seeing none",
            "all in favor say aye",
            "all those in favor",
            "entertain a motion",
            "we have a motion",
            "next item, please",
            "meeting is adjourned",
        ]
    ):
        return "Meeting_Chair"
    tail = prev_text[-65:]
    m_rec = re.search(r"((?:Councilman|Councilwoman|Mayor|Dr\.|Mr\.|Ms\.|Mrs\.)\s+[A-Z][a-z]+)\.?$", tail)
    if m_rec:
        return canonicalize_speaker(sanitize_name(m_rec.group(1)))
    return None


def diarize_and_merge_turns(turns: list[list], wav_path: str | None) -> list[dict]:
    if not turns:
        return []
    hints = [extract_speaker_hint(txt, turns[i - 1][2] if i > 0 else "") for i, (_, _, txt) in enumerate(turns)]

    if wav_path and os.path.exists(wav_path) and os.path.getsize(wav_path) > 4096:
        sr, audio = wavfile.read(wav_path)
        audio_f = audio.astype(np.float32) / 32768.0
        feats = []
        for st, en, _ in turns:
            i0, i1 = max(0, int(st * sr)), min(len(audio_f), int(en * sr))
            seg = audio_f[i0:i1]
            if len(seg) < int(0.25 * sr):
                seg = np.pad(seg, (0, int(0.25 * sr) - len(seg)))
            mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
            sc = librosa.feature.spectral_centroid(y=seg, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y=seg)
            feats.append(
                np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), [np.mean(sc), np.std(sc), np.mean(zcr)]])
            )
        feats_norm = StandardScaler().fit_transform(np.array(feats))

        named_centroids = {}
        for spk in set(h for h in hints if h):
            idxs = [j for j, h in enumerate(hints) if h == spk and (turns[j][1] - turns[j][0]) >= 1.8] or [
                j for j, h in enumerate(hints) if h == spk
            ]
            named_centroids[spk] = np.mean(feats_norm[idxs], axis=0)

        unlabeled = [j for j, h in enumerate(hints) if not h]
        if unlabeled:
            if len(unlabeled) >= 4:
                n_clust = min(10, max(2, len(unlabeled) // 8))
                labels = AgglomerativeClustering(n_clusters=n_clust, metric="cosine", linkage="average").fit_predict(
                    feats_norm[unlabeled]
                )
            else:
                labels = np.zeros(len(unlabeled), dtype=int)
            for pos, j in enumerate(unlabeled):
                if named_centroids:
                    best_spk, best_sim = max(
                        ((s, float(cosine_similarity(feats_norm[j : j + 1], c.reshape(1, -1))[0, 0])) for s, c in named_centroids.items()),
                        key=lambda kv: kv[1],
                    )
                    if best_sim >= 0.32:
                        hints[j] = best_spk
                        continue
                hints[j] = f"Voice_Cluster_{labels[pos] + 1:02d}"
    else:
        hints = [h or "Speaker_Unknown" for h in hints]

    spk_order = {}
    for h in hints:
        if h not in spk_order:
            spk_order[h] = f"Speaker_{len(spk_order) + 1:02d}_{h}"
    labeled = [spk_order[h] for h in hints]

    merged = []
    for (st, en, txt), spk in zip(turns, labeled):
        if merged and merged[-1]["speaker"] == spk and (st - merged[-1]["end"]) < 8.0:
            merged[-1]["end"] = en
            merged[-1]["text"] += " " + txt
        else:
            merged.append({"start": st, "end": en, "speaker": spk, "text": txt})

    merged2 = []
    i = 0
    while i < len(merged):
        cur = merged[i]
        if (
            i + 2 < len(merged)
            and (merged[i + 1]["end"] - merged[i + 1]["start"]) <= 2.0
            and cur["speaker"] == merged[i + 2]["speaker"]
            and (merged[i + 2]["start"] - cur["end"]) <= 5.0
        ):
            cur["end"] = merged[i + 2]["end"]
            cur["text"] += f" [{merged[i + 1]['text']}] " + merged[i + 2]["text"]
            i += 3
            merged2.append(cur)
        else:
            merged2.append(cur)
            i += 1
    return [t for t in merged2 if (t["end"] - t["start"]) >= 1.3 or len(t["text"].split()) >= 3]


def extract_ordinances_from_turns(turns: list[dict]) -> list[dict]:
    items = []
    for t in turns:
        txt = t["text"]
        if re.match(r"^(item|number|an ordinance|a resolution|docket)\b", txt.strip(), re.I) or (
            "ordinance" in txt.lower() and "reading" in txt.lower() and len(txt) < 420
        ):
            items.append({"start": t["start"], "hms": fmt_hms(t["start"]), "summary": txt[:280]})
    return items


def package_speaker_zips(zips_dir: str, title: str, date_str: str, ident: str, clips_meta: list[dict]) -> list[dict]:
    os.makedirs(zips_dir, exist_ok=True)
    by_spk = defaultdict(list)
    for c in clips_meta:
        if os.path.exists(c["local_path"]) and os.path.getsize(c["local_path"]) > 1024:
            by_spk[c["speaker"]].append(c)

    built_zips = []
    for spk in sorted(by_spk.keys()):
        items = sorted(by_spk[spk], key=lambda x: x["speaker_turn"])
        parts, cur_part, cur_sz = [], [], 0
        for it in items:
            sz = os.path.getsize(it["local_path"])
            if cur_part and cur_sz + sz > MAX_RELEASE_ASSET_BYTES:
                parts.append(cur_part)
                cur_part, cur_sz = [], 0
            cur_part.append(it)
            cur_sz += sz
        if cur_part:
            parts.append(cur_part)

        for p_idx, p_items in enumerate(parts, 1):
            suffix = f"_part{p_idx}" if len(parts) > 1 else ""
            zname = f"{spk}_{len(p_items):02d}_clips{suffix}.zip"
            zpath = os.path.join(zips_dir, zname)
            idx_lines = [
                f"Meeting: {title} ({date_str})",
                f"Internet Archive: https://archive.org/details/{ident}",
                f"Speaker: {spk} ({len(p_items)} clips)",
                "",
                "Speaking Turns Included:",
            ]
            with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as zf:
                for it in p_items:
                    zf.write(it["local_path"], arcname=f"{spk}/{it['file_name']}")
                    idx_lines.append(f"  - {it['file_name']} ({it['hms']}) | {it['text_preview']}")
                zf.writestr(f"{spk}/_SPEAKER_INDEX.txt", "\n".join(idx_lines) + "\n")
            built_zips.append(
                {
                    "speaker": spk,
                    "asset_name": zname,
                    "local_zip": zpath,
                    "size_bytes": os.path.getsize(zpath),
                    "clips_count": len(p_items),
                    "items": p_items,
                }
            )
    return built_zips


def build_meeting_markdown(title: str, date_str: str, ident: str, full_repo: str, release_tag: str, zip_records: list[dict], ordinances: list[dict]) -> str:
    total_clips = sum(z["clips_count"] for z in zip_records)
    lines = [
        f"# 🎬 Episode: {title} ({date_str})",
        "",
        f"- **Internet Archive Source:** [{ident}](https://archive.org/details/{ident})",
        f"- **GitHub Release (Speaker Video ZIPs):** [`{release_tag}`](https://github.com/{full_repo}/releases/tag/{release_tag})",
        f"- **Cast Size (Speakers):** {len(set(z['speaker'] for z in zip_records))} | **Speaking Turn Clips:** {total_clips}",
        "",
    ]
    if ordinances:
        lines.extend(["## 📜 Featured Ordinances, Resolutions & Agenda Scenes", ""])
        for o in ordinances:
            lines.append(f"- **`{o['hms']}`** — {o['summary']}")
        lines.append("")

    lines.extend([
        "## 🎭 Cast of Characters & Speaker Video Archives",
        "",
        "| Cast Member / Speaker | Turns | Archive Size | Download Clips ZIP |",
        "|---|---:|---:|---|",
    ])
    for z in zip_records:
        mb = z["size_bytes"] / 1e6
        lines.append(f"| `{z['speaker']}` | {z['clips_count']} | {mb:.1f} MB | [Download `{z['asset_name']}`]({z['download_url']}) |")

    lines.extend(["", "## ⏱️ Verbatim Turn Timestamps", ""])
    for z in zip_records:
        lines.append(f"### `{z['speaker']}` ({z['clips_count']} turns)")
        for it in z["items"]:
            preview = it["text_preview"].replace("|", "/").replace("\n", " ")
            lines.append(f"- **Turn {it['speaker_turn']:02d}** (`{it['hms']}`) — *{preview}*")
        lines.append("")
    return "\n".join(lines)


def process_and_publish_one(idx: int, doc: dict, target_repo: str, token: str, work_root: str, workers: int = 6) -> dict | None:
    ident = doc["identifier"]
    title = doc.get("title", ident)
    date_str = doc.get("date", "unknown")[:10]

    meta = fetch_json(f"https://archive.org/metadata/{ident}")
    files = meta.get("files", [])
    vid_files = [
        f for f in files
        if str(f.get("name", "")).endswith((".mp4", ".mpeg4", ".mov", ".ogv"))
        and not str(f.get("name", "")).endswith(".ia.mp4")
    ]
    vid_files.sort(key=lambda f: (0 if f["name"].endswith(".mp4") else (1 if f["name"].endswith(".mpeg4") else 2)))
    vtt_files = [f for f in files if str(f.get("name", "")).endswith((".en.vtt", ".en.asr.srt", ".vtt", ".srt"))]
    vtt_files.sort(key=lambda f: (0 if f["name"].endswith(".en.vtt") else 1))
    mp3_files = [f for f in files if str(f.get("name", "")).endswith(".mp3")]

    if not vid_files:
        return {"identifier": ident, "status": "skipped_no_video", "date": date_str}

    mdir = os.path.join(work_root, sanitize_name(ident))
    shutil.rmtree(mdir, ignore_errors=True)
    os.makedirs(mdir, exist_ok=True)

    try:
        vid_f = vid_files[0]
        vid_local = os.path.join(mdir, "meeting" + os.path.splitext(vid_f["name"])[1])
        print(f"\n=== [{idx:03d}] {date_str} | {title} -> {target_repo} ===")
        range_download(
            f"https://archive.org/download/{ident}/{urllib.parse.quote(vid_f['name'])}",
            vid_local,
            expected_size=int(vid_f.get("size", 0)) or None,
        )

        wav_local = os.path.join(mdir, "audio_16k.wav")
        audio_src = vid_local
        if mp3_files:
            mp3_local = os.path.join(mdir, "audio.mp3")
            range_download(
                f"https://archive.org/download/{ident}/{urllib.parse.quote(mp3_files[0]['name'])}",
                mp3_local,
                expected_size=int(mp3_files[0].get("size", 0)) or None,
            )
            audio_src = mp3_local

        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-max_error_rate", "1.0", "-i", audio_src, "-vn", "-ac", "1", "-ar", "16000", wav_local],
            check=False,
        )

        if vtt_files:
            vtt_local = os.path.join(mdir, "captions.vtt")
            range_download(f"https://archive.org/download/{ident}/{urllib.parse.quote(vtt_files[0]['name'])}", vtt_local)
            raw_turns = parse_vtt_turns(vtt_local)
        elif os.path.exists(wav_local):
            raw_turns = vad_fallback_turns(wav_local)
        else:
            raw_turns = []

        turns = diarize_and_merge_turns(raw_turns, wav_local if os.path.exists(wav_local) else None)
        if not turns:
            return {"identifier": ident, "status": "skipped_empty_turns", "date": date_str}

        ordinances = extract_ordinances_from_turns(turns)
        clips_dir = os.path.join(mdir, "clips")
        spk_counter = Counter()
        clips_meta = []
        for g_idx, t in enumerate(turns, 1):
            spk = t["speaker"]
            spk_counter[spk] += 1
            t_idx = spk_counter[spk]
            sdir = os.path.join(clips_dir, spk)
            os.makedirs(sdir, exist_ok=True)
            fname = f"{spk}_turn{t_idx:02d}_global{g_idx:03d}_{fmt_hms(t['start'])}-{fmt_hms(t['end'])}.mp4"
            fpath = os.path.join(sdir, fname)
            clips_meta.append(
                {
                    "global_idx": g_idx,
                    "speaker": spk,
                    "speaker_turn": t_idx,
                    "start": round(t["start"], 2),
                    "end": round(t["end"], 2),
                    "hms": f"{fmt_hms(t['start'])}-{fmt_hms(t['end'])}",
                    "file_name": fname,
                    "local_path": fpath,
                    "text_preview": t["text"][:220],
                }
            )

        def cut_clip(c: dict) -> None:
            st = max(0.0, c["start"] - 0.15)
            dur = max(0.8, (c["end"] + 0.25) - st)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", f"{st:.2f}", "-i", vid_local, "-t", f"{dur:.2f}",
                    "-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart",
                    c["local_path"],
                ],
                check=False,
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(cut_clip, clips_meta))

        if os.path.exists(vid_local):
            os.remove(vid_local)

        zip_records = package_speaker_zips(os.path.join(mdir, "zips"), title, date_str, ident, clips_meta)
        if not zip_records:
            return {"identifier": ident, "status": "no_valid_clips", "date": date_str}

        slug = sanitize_name(f"{date_str}_{title}")[:75]
        release_tag = f"speaker-videos-{date_str}-{ident[-11:]}"
        release_title = f"🎬 [{date_str}] {title} — Speaker Video Clips"
        release_intro = (
            f"**The Real Windy City — Episode Speaker Video Package**\n\n"
            f"- **Source:** Internet Archive [`{ident}`](https://archive.org/details/{ident})\n"
            f"- **Cast Members (Speakers):** {len(set(z['speaker'] for z in zip_records))}\n"
            f"- **Individual Turn `.mp4` Clips:** {sum(z['clips_count'] for z in zip_records)}\n"
        )
        release = gh_ensure_release(target_repo, release_tag, release_title, release_intro, token)

        for z in zip_records:
            print(f"  -> Uploading {z['asset_name']} ({z['size_bytes']/1e6:.1f} MB)...")
            z["download_url"] = gh_upload_release_asset(release, z["local_zip"], z["asset_name"], token)

        md_text = build_meeting_markdown(title, date_str, ident, target_repo, release_tag, zip_records, ordinances)
        clean_manifest = [{k: v for k, v in c.items() if k != "local_path"} for c in clips_meta]
        episode_json = {
            "identifier": ident,
            "title": title,
            "date": date_str,
            "season": max(1, int(date_str[:4]) - 2011) if date_str[:4].isdigit() else 15,
            "repo": target_repo,
            "release_tag": release_tag,
            "release_url": f"https://github.com/{target_repo}/releases/tag/{release_tag}",
            "archive_embed_url": f"https://archive.org/embed/{ident}",
            "archive_stream_url": f"https://archive.org/download/{ident}/{urllib.parse.quote(vid_f['name'])}",
            "cast": [
                {
                    "speaker": z["speaker"],
                    "clips_count": z["clips_count"],
                    "size_bytes": z["size_bytes"],
                    "zip_url": z["download_url"],
                }
                for z in zip_records
            ],
            "ordinances": ordinances,
            "turns_count": len(clean_manifest),
        }

        folder_prefix = f"speaker-videos/{slug}"
        gh_put_file(target_repo, f"{folder_prefix}/SPEAKER_INDEX.md", md_text.encode("utf-8"), f"Add speaker video index for {title} ({date_str})", token)
        gh_put_file(target_repo, f"{folder_prefix}/manifest.json", json.dumps(clean_manifest, indent=2).encode("utf-8"), f"Add speaker manifest for {title} ({date_str})", token)
        gh_put_file(target_repo, f"{folder_prefix}/episode.json", json.dumps(episode_json, indent=2).encode("utf-8"), f"Add TV episode metadata for {title} ({date_str})", token)

        return {
            "identifier": ident,
            "date": date_str,
            "title": title,
            "repo": target_repo,
            "release_tag": release_tag,
            "release_url": f"https://github.com/{target_repo}/releases/tag/{release_tag}",
            "index_path": f"{folder_prefix}/SPEAKER_INDEX.md",
            "episode_json": f"{folder_prefix}/episode.json",
            "speakers": len(set(z["speaker"] for z in zip_records)),
            "clips": sum(z["clips_count"] for z in zip_records),
            "status": "published",
        }
    finally:
        shutil.rmtree(mdir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="therealwindycity")
    parser.add_argument("--routing", choices=["year-repos", "single-repo"], default="single-repo")
    parser.add_argument("--repo", default="therealwindycity/cheyenne-archives-2025-2026")
    parser.add_argument("--min-year", type=int, default=0)
    parser.add_argument("--max-year", type=int, default=9999)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-minutes", type=float, default=280.0)
    parser.add_argument("--work-dir", default="/tmp/cheyenne_gh_work")
    args = parser.parse_args()

    token = os.environ.get("GH_ARCHIVE_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("ERROR: Set GH_ARCHIVE_TOKEN or GITHUB_TOKEN.")

    state_repo = args.repo if args.routing == "single-repo" else f"{args.owner}/cheyenne-archives-2025-2026"
    state = gh_get_json_file(state_repo, "speaker-videos/state.json", token, {"processed": {}})
    processed = state.setdefault("processed", {})

    t_start = time.time()
    published_this_run = 0
    page, global_idx = 1, 0

    while True:
        data = fetch_json(ARCHIVE_SEARCH_URL.format(rows=100, page=page))
        docs = data.get("response", {}).get("docs", [])
        if not docs:
            break
        for doc in docs:
            global_idx += 1
            ident = doc["identifier"]
            date_str = doc.get("date", "2026-01-01")[:10]
            year = int(date_str[:4]) if date_str[:4].isdigit() else 2026
            if year < args.min_year or year > args.max_year:
                continue
            if ident in processed and processed[ident].get("status") in ("published", "skipped_no_video", "skipped_empty_turns"):
                continue
            if (time.time() - t_start) / 60.0 >= args.max_minutes:
                return

            target_repo = args.repo if args.routing == "single-repo" else repo_for_year(args.owner, year)
            res = process_and_publish_one(global_idx, doc, target_repo, token, args.work_dir)
            if res:
                processed[ident] = res
                gh_put_file(state_repo, "speaker-videos/state.json", json.dumps(state, indent=2).encode("utf-8"), f"Update speaker video state ({ident})", token)
                if res.get("status") == "published":
                    published_this_run += 1
                    if args.batch_size and published_this_run >= args.batch_size:
                        return
        page += 1


if __name__ == "__main__":
    main()
