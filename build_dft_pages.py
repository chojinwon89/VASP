#!/usr/bin/env python
"""
build_dft_pages.py
==================
Regenerate the two DFT-vs-(GOAD+SevenNet) benchmark pages for the
bond-distance-review website from the *current* analysis_out data:

  * dft_vs_goad_energy.html  - binding-energy comparison (parity plot,
      per-functional / per-metal / per-molecule accuracy tables, and a
      sortable table of every matched pair).
  * dft_comparison.html      - structure & geometry gallery: one card per
      system with the MLIP relaxed image, the metal-adsorbate bond distance,
      the binding site and the per-functional E_ads, with the DFT final
      structure / bond distance / site filled in once the cluster geometry
      extraction (compare_dft_mlip_structures.py) has been run.

All summary statistics reuse load_pairs()/regression_stats() from
analyze_dft_mlip_accuracy.py so the numbers match dft_mlip_accuracy_report.txt
exactly (same max_diff / max_eads filtering).

Usage
-----
    python build_dft_pages.py \
        --analysis-dir analysis_out \
        --gallery "/path/to/GOAD+Sevennet_Structures" \
        --out-dir /tmp/bond-pages
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

from analyze_dft_mlip_accuracy import load_pairs, regression_stats, parse_metal
from mol_canon import canon_molecule, match_keys as mol_match_keys

FUNCS = ["pbe", "pbe_d3", "r2scan", "beef_vdw"]
FUNC_LABEL = {"pbe": "PBE", "pbe_d3": "PBE+D3", "r2scan": "r²SCAN",
              "beef_vdw": "BEEF-vdW"}

# Pretty display labels (subscripts) for formula-style tokens.
MOL_LABEL = {
    "CO2": "CO₂", "H2O": "H₂O", "SO2": "SO₂", "NH3": "NH₃", "H2S": "H₂S",
    "N2": "N₂", "NO2": "NO₂", "O2": "O₂", "H2": "H₂", "CH4": "CH₄",
    "CH3": "CH₃", "CH2": "CH₂",
}

# Molecule-name canonicalisation (formula <-> common name) is centralised in
# mol_canon; mol_match_keys returns the set of spellings a token may match.
MOL_CLASS = {
    "ethane": "alkane", "propane": "alkane",
    "ethene": "alkene", "propene": "alkene",
    "ethanol": "alcohol", "propanol": "alcohol",
    "isopropanol": "alcohol", "glycerol": "polyol",
    "CO2": "other",
}

CSS = """
 :root{--bg:#0f1116;--card:#191c24;--fg:#e6e6e6;--muted:#9aa0aa;--line:#2a2e38;}
 *{box-sizing:border-box;}
 body{margin:0;background:var(--bg);color:var(--fg);
   font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
 header{padding:18px 22px;border-bottom:1px solid var(--line);}
 h1{font-size:19px;margin:0 0 6px;} h2{font-size:16px;margin:30px 0 4px;}
 h3{font-size:13.5px;margin:16px 0 2px;color:#cfd4dd;}
 a{color:#6ea8fe;}
 .wrap{max-width:1150px;margin:0 auto;padding:0 22px 70px;}
 .sub{color:var(--muted);font-size:13px;max-width:960px;}
 .note{background:#20242e;border:1px solid var(--line);border-left:3px solid #e0a800;
   border-radius:8px;padding:12px 14px;margin:14px 0;font-size:12.8px;color:#d7dbe2;}
 .note b{color:#fff;}
 .note.info{border-left-color:#4c78a8;}
 .status{background:#1c2620;border:1px solid #2f5741;border-left:3px solid #4bb97a;
   border-radius:8px;padding:12px 15px;margin:14px 0;font-size:12.8px;color:#d7f0e0;}
 .status b{color:#7fe3a6;}
 .status .live{display:inline-block;font-size:10px;font-weight:700;letter-spacing:.5px;
   color:#0c1a12;background:#4bb97a;border-radius:9px;padding:1px 7px;margin-right:8px;}
 .status .pend{background:#c9a227;color:#1c1400;}
 .fig{background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0;}
 .fig img{width:100%;display:block;}
 .legend{color:var(--muted);font-size:12px;margin:2px 0 8px;}
 table.big{width:100%;border-collapse:collapse;font-size:12.5px;margin:8px 0 4px;}
 table.big th,table.big td{padding:7px 10px;border:1px solid var(--line);text-align:right;}
 table.big th{background:#20242e;}
 table.big td.l,table.big th.l{text-align:left;}
 tr.good td{background:rgba(44,160,44,.10);} tr.bad td{background:rgba(214,39,40,.12);}
 tr.mid td{background:rgba(224,168,0,.08);}
 .kpi{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;}
 .kpi div{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:9px 13px;}
 .kpi b{font-size:17px;} .kpi span{color:var(--muted);font-size:11.5px;display:block;}
 .controls{display:flex;gap:10px;align-items:center;margin:10px 0;flex-wrap:wrap;}
 select,input{background:var(--card);color:var(--fg);border:1px solid var(--line);
   border-radius:8px;padding:6px 9px;font-size:13px;}
 table.data{width:100%;border-collapse:collapse;font-size:12px;}
 table.data th,table.data td{padding:6px 8px;border-bottom:1px solid var(--line);
   text-align:right;white-space:nowrap;}
 table.data th{position:sticky;top:0;background:#20242e;cursor:pointer;}
 table.data td.l,table.data th.l{text-align:left;}
 .u{color:#ff9a9a;} .o{color:#7ee0a1;}
 .pill{font-size:10px;padding:1px 6px;border-radius:9px;border:1px solid var(--line);color:#c9ced8;}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;margin-top:12px;}
 .g{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;}
 .g h3{margin:0 0 4px;font-size:13.5px;color:#fff;}
 .g .m{color:var(--muted);font-size:11px;margin-bottom:6px;}
 .pair{display:flex;gap:8px;}
 .pair>div{flex:1;text-align:center;}
 .pair img{width:100%;border-radius:6px;background:#fff;}
 .pair .cap{font-size:10.5px;color:var(--muted);margin-top:2px;}
 .ph{display:flex;align-items:center;justify-content:center;height:150px;border-radius:6px;
   background:#141821;border:1px dashed #39404e;color:#6b7280;font-size:11px;text-align:center;padding:6px;}
 table.mini{width:100%;border-collapse:collapse;font-size:11px;margin-top:8px;}
 table.mini th,table.mini td{padding:3px 5px;border-bottom:1px solid var(--line);text-align:right;}
 table.mini td.l,table.mini th.l{text-align:left;} table.mini th{color:var(--muted);font-weight:600;}
 .tag{font-size:10px;padding:1px 6px;border-radius:9px;background:#222833;color:#c9ced8;border:1px solid var(--line);}
"""

NAV = ('<a href="index.html">&larr; gallery</a> &middot; '
       '<a href="dft_vs_goad_energy.html">binding energy</a> &middot; '
       '<a href="dft_comparison.html">structure &amp; geometry</a> &middot; '
       '<a href="dft_energy.html">single-point DFT vs ML</a> &middot; '
       '<a href="mlip_benchmark.html" style="color:#d0a3ff">SevenNet vs MatterSim</a> &middot; '
       '<a href="flagged.html" style="color:#ff9a9a">likely failures</a>')


def esc(s):
    return html.escape(str(s))


def mlabel(m):
    return MOL_LABEL.get(m, m)


def fmt(v, nd=3, plus=False):
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"


# --------------------------------------------------------------------------
def load_binding(path):
    """(surface, molecule, functional) -> E_ads (float) for status=ok rows."""
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(path.open()):
        if r.get("status") != "ok":
            continue
        try:
            out[(r["surface"], r["molecule"], r["functional"])] = float(r["E_ads"])
        except (ValueError, KeyError):
            pass
    return out


def load_mlip_geom(path):
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(path.open()):
        out[(r["surface"], r["molecule"])] = r
    return out


def load_struct_compare(path):
    """(surface, molecule) -> row, keyed by both the raw and canonicalised
    molecule name so it matches the pairs/gallery naming regardless of whether
    the DFT job dirs used formulas (C2H4) or common names (ethene)."""
    out = {}
    if not path or not Path(path).exists():
        return out
    for r in csv.DictReader(Path(path).open()):
        for key in {(r["surface"], k) for k in mol_match_keys(r["molecule"])}:
            if key not in out or r.get("functional") == "pbe":
                out[key] = r
    return out


def stat_row(sub, css=None):
    s = regression_stats(sub)
    if s["n"] == 0:
        return None
    within = sum(1 for d, m in sub if d is not None and m is not None
                 and abs(m - d) <= 0.5) / s["n"] * 100
    s["within"] = within
    return s


# --------------------------------------------------------------------------
def page_energy(pairs, out_dir, parity_name):
    funcs = [f for f in FUNCS if any(r["functional"] == f for r in pairs)]
    n_pairs = len(pairs)
    systems = {(r["surface"], r["molecule"]) for r in pairs}

    # per functional
    frows = []
    best = None
    for fn in FUNCS:
        sub = [(r["E_ads_DFT"], r["E_ads_ML"]) for r in pairs if r["functional"] == fn]
        s = stat_row(sub)
        if s is None:
            frows.append((fn, None))
            continue
        frows.append((fn, s))
        if best is None or s["mae"] < best[1]["mae"]:
            best = (fn, s)

    # per metal (pooled)
    metals = defaultdict(list)
    for r in pairs:
        metals[parse_metal(r["surface"])].append((r["E_ads_DFT"], r["E_ads_ML"]))
    # per molecule (pooled)
    mols = defaultdict(list)
    for r in pairs:
        mols[r["molecule"]].append((r["E_ads_DFT"], r["E_ads_ML"]))

    def ftable():
        out = ['<table class="big"><tr><th class="l">DFT functional</th><th>n</th>'
               '<th>MAE (eV)</th><th>RMSE (eV)</th><th>bias &Delta; (eV)</th>'
               '<th>R&sup2;</th><th>within 0.5 eV</th><th class="l">note</th></tr>']
        for fn, s in frows:
            if s is None:
                note = ("references broken &mdash; recompute" if fn == "beef_vdw"
                        else "no matched pairs yet")
                out.append(f'<tr class="bad"><td class="l"><b>{FUNC_LABEL[fn]}</b></td>'
                           f'<td>0</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td>'
                           f'<td>n/a</td><td class="l">{note}</td></tr>')
                continue
            cls = "good" if s["mae"] <= 0.3 else ("mid" if s["mae"] <= 0.5 else "bad")
            biascls = "u" if s["bias"] < -0.05 else ("o" if s["bias"] > 0.05 else "")
            star = " &starf;" if best and fn == best[0] else ""
            out.append(
                f'<tr class="{cls}"><td class="l"><b>{FUNC_LABEL[fn]}</b>{star}</td>'
                f'<td>{s["n"]}</td><td>{fmt(s["mae"],3)}</td><td>{fmt(s["rmse"],3)}</td>'
                f'<td class="{biascls}">{fmt(s["bias"],3,plus=True)}</td>'
                f'<td>{fmt(s["r2"],3)}</td><td>{s["within"]:.0f}%</td>'
                f'<td class="l"></td></tr>')
        out.append("</table>")
        return "".join(out)

    def pooled_table(dct, label):
        out = [f'<table class="big"><tr><th class="l">{label}</th><th>n</th>'
               '<th>MAE (eV)</th><th>bias &Delta; (eV)</th><th>within 0.5 eV</th></tr>']
        for k in sorted(dct, key=lambda k: -len(dct[k])):
            s = stat_row(dct[k])
            if s is None:
                continue
            cls = "good" if s["mae"] <= 0.3 else ("mid" if s["mae"] <= 0.5 else "bad")
            biascls = "u" if s["bias"] < -0.05 else ("o" if s["bias"] > 0.05 else "")
            out.append(f'<tr class="{cls}"><td class="l">{mlabel(k)}</td><td>{s["n"]}</td>'
                       f'<td>{fmt(s["mae"],3)}</td>'
                       f'<td class="{biascls}">{fmt(s["bias"],3,plus=True)}</td>'
                       f'<td>{s["within"]:.0f}%</td></tr>')
        out.append("</table>")
        return "".join(out)

    # sortable per-pair table
    data = []
    for r in pairs:
        data.append(dict(f=FUNC_LABEL[r["functional"]], s=r["surface"],
                         m=mlabel(r["molecule"]),
                         cls=MOL_CLASS.get(r["molecule"], ""),
                         dft=round(r["E_ads_DFT"], 3), ml=round(r["E_ads_ML"], 3),
                         d=round(r["E_ads_ML"] - r["E_ads_DFT"], 3)))
    data.sort(key=lambda x: (x["f"], x["s"], x["m"]))

    best_txt = (f'{FUNC_LABEL[best[0]]} (MAE {best[1]["mae"]:.2f} eV, '
                f'bias {best[1]["bias"]:+.2f} eV, n={best[1]["n"]})' if best else "n/a")

    body = f"""
 <div class="status">
   <span class="live">DATA REFRESHED</span>
   <b>Rebuilt from the re-run DFT set</b> (slab references re-matched by metal-atom
   count in <code>vasp_slab</code>) &mdash; this supersedes the earlier published
   numbers. <span class="pend" style="display:inline-block;font-size:10px;font-weight:700;border-radius:9px;padding:1px 7px;">BEEF-vdW PENDING</span>
   its gas/slab references are non-physical and are being recomputed, so it has
   no valid pairs yet.
 </div>

 <div class="fig"><img src="{parity_name}" alt="DFT vs SevenNet-OMNI adsorption-energy parity plots per functional"></div>
 <div class="legend">One parity panel per DFT functional. Each point is a system relaxed
   <b>independently</b> by DFT and by GOAD+SevenNet-OMNI; the ML adsorption energy is plotted
   against the DFT value. Grey band = &plusmn;0.3 eV.</div>

 <div class="kpi">
   <div><b>{n_pairs}</b><span>matched pairs (filtered)</span></div>
   <div><b>{len(systems)}</b><span>unique systems</span></div>
   <div><b>{best_txt.split('(')[0].strip()}</b><span>best-agreeing functional</span></div>
   <div><b>{best[1]['mae']:.2f} eV</b><span>best-functional MAE</span></div>
   <div><b>3</b><span>metals (Cu, Pd, Pt)</span></div>
 </div>

 <div class="note">Statistics use the same sanity filter as the accuracy report:
   drop pairs whose DFT E<sub>ads</sub> &gt; 0.5 eV (broken reference) or |&Delta;| &gt; 5 eV
   (gross outlier). <b>&Delta; = E<sub>ads</sub>(ML) &minus; E<sub>ads</sub>(DFT)</b>; negative
   &rArr; ML binds more strongly (over-binds).</div>

 <h2>1 &middot; Which DFT functional does GOAD+SevenNet match?</h2>
 <div class="legend">Pooled over the matched Cu/Pd/Pt systems. &starf; = best MAE.</div>
 {ftable()}

 <div class="note info"><b>Read with care.</b> In this re-run PBE agrees best (near-zero bias)
   while PBE+D3 and r&sup2;SCAN show ML <i>under</i>-binding &mdash; the opposite of the physically
   expected dispersion trend and of the earlier run. That, together with the broken BEEF-vdW
   references, points to a problem in the D3 / r&sup2;SCAN / BEEF reference energies rather than
   in the MLIP. Treat PBE as the trustworthy column until the vdW references are re-run.</div>

 <h2>2 &middot; By surface metal</h2>
 {pooled_table(metals, "metal")}

 <h2>3 &middot; By molecule</h2>
 {pooled_table(mols, "molecule")}

 <h2>4 &middot; Every matched pair</h2>
 <div class="controls">
   <label>functional <select id="ff"><option value="">all</option>{"".join(f'<option>{FUNC_LABEL[f]}</option>' for f in funcs)}</select></label>
   <label>metal <select id="fm"><option value="">all</option><option>Cu</option><option>Pd</option><option>Pt</option></select></label>
   <input id="fq" placeholder="filter molecule/surface&hellip;" size="18">
   <span class="legend" id="cnt"></span>
 </div>
 <table class="data" id="tbl"><thead><tr>
   <th class="l" data-k="f">functional</th><th class="l" data-k="s">surface</th>
   <th class="l" data-k="m">molecule</th><th class="l" data-k="cls">class</th>
   <th data-k="dft">E_ads DFT</th><th data-k="ml">E_ads ML</th><th data-k="d">&Delta; (ML&minus;DFT)</th>
 </tr></thead><tbody></tbody></table>

 <script>
 const DATA={json.dumps(data, separators=(",", ":"))};
 const tb=document.querySelector("#tbl tbody"), cnt=document.getElementById("cnt");
 let sortK="s", sortA=true;
 function draw(){{
   const ff=document.getElementById("ff").value, fm=document.getElementById("fm").value,
         fq=document.getElementById("fq").value.toLowerCase();
   let rows=DATA.filter(r=>(!ff||r.f===ff)&&(!fm||r.s.startsWith(fm))&&
     (!fq||(r.m+r.s).toLowerCase().includes(fq)));
   rows.sort((a,b)=>{{let x=a[sortK],y=b[sortK];return (x<y?-1:x>y?1:0)*(sortA?1:-1);}});
   tb.innerHTML=rows.map(r=>{{
     const dc=r.d<-0.05?'u':r.d>0.05?'o':'';
     return `<tr><td class="l">${{r.f}}</td><td class="l">${{r.s}}</td><td class="l">${{r.m}}</td>`+
       `<td class="l"><span class="pill">${{r.cls}}</span></td><td>${{r.dft.toFixed(3)}}</td>`+
       `<td>${{r.ml.toFixed(3)}}</td><td class="${{dc}}">${{r.d>=0?'+':''}}${{r.d.toFixed(3)}}</td></tr>`;
   }}).join("");
   cnt.textContent=rows.length+" rows";
 }}
 document.querySelectorAll("#tbl th").forEach(th=>th.onclick=()=>{{
   const k=th.dataset.k; sortA=(sortK===k)?!sortA:true; sortK=k; draw();}});
 ["ff","fm","fq"].forEach(id=>document.getElementById(id).oninput=draw);
 draw();
 </script>
"""
    write_page(out_dir / "dft_vs_goad_energy.html",
               "DFT vs GOAD+SevenNet — relaxed adsorption energies",
               "DFT vs GOAD+SevenNet &mdash; relaxed adsorption energies",
               "Each system is <b>independently relaxed</b> by DFT and by GOAD+SevenNet-OMNI; "
               "we compare the adsorption energies of the two <b>separately optimized</b> "
               "structures. Cu/Pd/Pt &middot; C1&ndash;C3 molecules &middot; "
               "<a href=\"dft_vs_mlip_pairs.csv\" download>download pairs (CSV)</a>",
               body)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def page_structure(pairs, geom, struct, gallery, out_dir, dft_png_dir=None):
    # unique systems from pairs, plus any geom systems
    keys = sorted(set((r["surface"], r["molecule"]) for r in pairs) | set(geom.keys()))
    # per-system per-functional (E_ads_DFT, E_ads_ML) straight from the matched
    # pairs (same fully-relaxed organic source as the energy page); the broad
    # dft_binding_energies_all.csv grid covers a different (small-molecule) set.
    sysfunc = defaultdict(dict)
    ml_eads = {}
    for r in pairs:
        key = (r["surface"], r["molecule"])
        sysfunc[key][r["functional"]] = (r["E_ads_DFT"], r["E_ads_ML"])
        ml_eads.setdefault(key, r["E_ads_ML"])

    # copy MLIP pngs (and DFT pngs if a render dir was supplied)
    png_dir = out_dir / "dftcmp" / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    have_img, have_img_dft = {}, {}

    def _find_png(base_dir, surf, mol, suffix):
        """First existing <surf>_<key><suffix>.png trying every molecule spelling."""
        if not base_dir:
            return None
        for key in (mol, *mol_match_keys(mol)):
            cand = Path(base_dir) / f"{surf}_{key}{suffix}.png"
            if cand.exists():
                return cand
        return None

    for surf, mol in keys:
        src = _find_png(gallery, surf, mol, "")
        if src:
            shutil.copy(src, png_dir / f"{surf}_{mol}_mlip.png")
            have_img[(surf, mol)] = f"dftcmp/png/{surf}_{mol}_mlip.png"
        dsrc = _find_png(dft_png_dir, surf, mol, "_dft")
        if dsrc:
            shutil.copy(dsrc, png_dir / f"{surf}_{mol}_dft.png")
            have_img_dft[(surf, mol)] = f"dftcmp/png/{surf}_{mol}_dft.png"

    def _struct(surf, mol):
        for k in mol_match_keys(mol):
            if (surf, k) in struct:
                return struct[(surf, k)]
        return None

    n_dftgeom = sum(1 for surf, mol in keys
                    if _struct(surf, mol)
                    and _struct(surf, mol).get("min_dist_dft") not in (None, "", "None"))

    cards = []
    for surf, mol in keys:
        g = geom.get((surf, mol), {})
        md = g.get("mlip_min_dist", "")
        pair = g.get("mlip_pair", "")
        site = g.get("mlip_site", "")
        img = have_img.get((surf, mol))
        img_html = (f'<img src="{img}" alt="{surf} {mol} MLIP">' if img
                    else '<div class="ph">no MLIP image</div>')

        sc = _struct(surf, mol)
        dft_d = sc.get("min_dist_dft") if sc else None
        have_dft_geom = sc is not None and str(dft_d) not in ("", "None", "n/a", None)
        dimg = have_img_dft.get((surf, mol))
        if dimg:
            dft_inner = f'<img src="{dimg}" alt="{surf} {mol} DFT">'
        elif have_dft_geom:
            dft_inner = (f'<div class="ph">DFT geom only<br>{esc(sc.get("pair_dft",""))} '
                         f'{esc(dft_d)} &#8491;</div>')
        else:
            dft_inner = '<div class="ph">DFT structure<br>pending cluster<br>extraction</div>'
        dft_box = dft_inner

        # geometry comparison line (metal-adsorbate contact): MLIP vs DFT
        geom_line = f'MLIP contact: <b>{esc(pair)} {esc(md)} &#8491;</b> &middot; site <b>{esc(site)}</b>'
        if have_dft_geom:
            mv_, dv_ = _num(md), _num(dft_d)
            dd = (f' &middot; &Delta;d {mv_-dv_:+.2f} &#8491;'
                  if (mv_ is not None and dv_ is not None) else '')
            rmsd = sc.get("rmsd", "")
            geom_line += (f'<br>DFT contact: <b>{esc(sc.get("pair_dft",""))} {esc(dft_d)} &#8491;</b>'
                          f'{dd}' + (f' &middot; RMSD {esc(rmsd)} &#8491;' if str(rmsd) not in ("", "None") else ''))

        # per-functional energy mini-table (from the matched pairs)
        mlv = ml_eads.get((surf, mol))
        fmap = sysfunc.get((surf, mol), {})
        mini = ['<table class="mini"><tr><th class="l">func</th><th>E_ads DFT</th>'
                '<th>E_ads ML</th><th>&Delta;</th></tr>']
        for fn in FUNCS:
            if fn not in fmap:
                note = "ref broken" if fn == "beef_vdw" else "&mdash;"
                mini.append(f'<tr><td class="l">{FUNC_LABEL[fn]}</td><td class="l" '
                            f'style="color:#6b7280">{note}</td>'
                            f'<td>{fmt(mlv,2) if mlv is not None else "n/a"}</td><td>&mdash;</td></tr>')
                continue
            dv, mv = fmap[fn]
            d = mv - dv
            dc = "u" if d < -0.05 else ("o" if d > 0.05 else "")
            mini.append(f'<tr><td class="l">{FUNC_LABEL[fn]}</td><td>{fmt(dv,2)}</td>'
                        f'<td>{fmt(mv,2)}</td>'
                        f'<td class="{dc}">{fmt(d,2,plus=True)}</td></tr>')
        mini.append("</table>")

        cards.append(f"""
   <div class="g">
     <h3>{esc(surf)} &middot; {mlabel(mol)} <span class="tag">{MOL_CLASS.get(mol,'')}</span></h3>
     <div class="m">{geom_line}</div>
     <div class="pair">
       <div>{img_html}<div class="cap">GOAD+SevenNet-OMNI</div></div>
       <div>{dft_box}<div class="cap">DFT (relaxed)</div></div>
     </div>
     {''.join(mini)}
   </div>""")

    n_dftimg = len(have_img_dft)
    status = f"""
 <div class="status">
   <span class="live">MLIP SIDE READY</span>
   <b>{len(have_img)} systems</b> show the GOAD+SevenNet-OMNI relaxed structure, its
   metal&ndash;adsorbate bond distance and binding site (computed with the same
   minimum-image contact metric used cluster-side).
   <span class="live pend">DFT GEOMETRY {n_dftgeom}/{len(keys)}</span>
   <span class="live pend">DFT IMAGES {n_dftimg}/{len(keys)}</span>
   the DFT final structures, bond distances and sites populate here once
   <code>compare_dft_mlip_structures.py</code> (+ <code>render_dft_structures.py</code>)
   are run on the cluster (where the CONTCARs live) and their CSV + PNGs are synced back.
 </div>
 <div class="note">Binding site is a <b>coarse coordination label</b> (metal neighbours of the
   closest adsorbate atom within +0.45 &#8491; of the shortest contact: 1&rarr;atop, 2&rarr;bridge,
   &ge;3&rarr;hollow). For weakly physisorbed alkanes the site is indicative only; the
   <b>bond distance</b> and <b>contact pair</b> are the robust metrics.</div>
"""
    body = status + f'<h2>Per-system structure &amp; geometry ({len(keys)} systems)</h2>' \
                    f'<div class="grid">{"".join(cards)}</div>'
    write_page(out_dir / "dft_comparison.html",
               "GOAD+SevenNet vs DFT — structure & geometry",
               "GOAD+SevenNet vs DFT &mdash; structure &amp; geometry",
               "Side-by-side relaxed structures, metal&ndash;adsorbate bond distance and "
               "binding site for each benchmark system. MLIP side is live; DFT side fills "
               "in from the cluster geometry extraction.",
               body)


def write_page(path, title, h1, sub, body):
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style></head>
<body>
<header>
 <h1>{h1}</h1>
 <div class="sub">{sub}<br>{NAV}</div>
</header>
<div class="wrap">
{body}
</div>
</body></html>
"""
    path.write_text(doc)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", default="analysis_out")
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--struct-compare", default=None,
                    help="dft_mlip_structure_compare.csv (from the cluster) if available.")
    ap.add_argument("--dft-png-dir", default=None,
                    help="Directory of rendered DFT PNGs named <surface>_<molecule>_dft.png "
                         "(from render_dft_structures.py) if available.")
    ap.add_argument("--max-diff", type=float, default=5.0)
    ap.add_argument("--max-eads", type=float, default=0.5)
    args = ap.parse_args()

    A = Path(args.analysis_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(A / "dft_vs_mlip_pairs.csv",
                       max_diff=args.max_diff, max_eads=args.max_eads)
    binding = load_binding(A / "dft_binding_energies_all.csv")
    geom = load_mlip_geom(A / "mlip_geom.csv")
    struct = load_struct_compare(args.struct_compare)

    # parity plot
    parity_src = A / "dft_vs_mlip_all.png"
    parity_name = "dft_vs_mlip_all.png"
    if parity_src.exists():
        shutil.copy(parity_src, out / parity_name)
    # publish the raw pairs csv for download
    if (A / "dft_vs_mlip_pairs.csv").exists():
        shutil.copy(A / "dft_vs_mlip_pairs.csv", out / "dft_vs_mlip_pairs.csv")

    print(f"pairs(filtered)={len(pairs)}  binding_ok={len(binding)}  "
          f"mlip_geom={len(geom)}  struct_compare={len(struct)}")
    page_energy(pairs, out, parity_name)
    page_structure(pairs, geom, struct, args.gallery, out, dft_png_dir=args.dft_png_dir)
    print("done.")


if __name__ == "__main__":
    raise SystemExit(main())
