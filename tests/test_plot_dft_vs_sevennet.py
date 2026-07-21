from pathlib import Path

import matplotlib.pyplot as plt

from plot_dft_vs_sevennet import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    LEGEND_TITLE_FONTSIZE,
    SUPTITLE_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    make_figure,
)


def _build_plot_inputs(functionals):
    calculators = ["sevennet_omni", "5m"]
    surfaces = ["Cu111", "Pt111", "Pd111", "Ni111", "Ag111", "Au111"]
    molecules = [
        "ethane", "ethene", "benzene", "ethanol", "acetaldehyde", "acetone",
        "acetic_acid", "lactic_acid", "DMSO", "CO2", "N2",
    ]

    dft_data = {func: {} for func in functionals}
    ml_data = {calc: {} for calc in calculators}
    calc_pairs_per_func = {}

    for f_idx, func in enumerate(functionals):
        per_calc = {}
        for c_idx, calc in enumerate(calculators):
            pairs = []
            for s_idx, surf in enumerate(surfaces):
                for m_idx, mol in enumerate(molecules):
                    key = (surf, mol)
                    e_dft = -2.3 + 0.035 * s_idx + 0.012 * m_idx + 0.02 * f_idx
                    e_ml = e_dft + 0.02 * (c_idx + 1)
                    dft_data[func][key] = e_dft
                    ml_data[calc][key] = e_ml
                    pairs.append(key)
            per_calc[calc] = pairs
        calc_pairs_per_func[func] = per_calc

    return calculators, calc_pairs_per_func, dft_data, ml_data


def _capture_figure(monkeypatch):
    captured = {}

    def _close(fig=None):
        captured["fig"] = fig

    monkeypatch.setattr(plt, "close", _close)
    return captured


def _assert_legends_do_not_overlap_axes(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(fig.legends) == 3
    for legend in fig.legends:
        legend_bbox = legend.get_window_extent(renderer=renderer)
        for ax in axes:
            assert not legend_bbox.overlaps(ax.get_window_extent(renderer=renderer))


def test_make_figure_uses_larger_fonts_and_legends_outside(tmp_path, monkeypatch):
    functionals = ["pbe", "pbe_d3", "beef_vdw", "r2scan"]
    calculators, calc_pairs_per_func, dft_data, ml_data = _build_plot_inputs(functionals)
    output_path = tmp_path / "parity_4funcs.png"
    captured = _capture_figure(monkeypatch)

    make_figure(
        functionals=functionals,
        calc_pairs_per_func=calc_pairs_per_func,
        dft_data=dft_data,
        ml_data=ml_data,
        calculators=calculators,
        output_path=output_path,
        axis_min=-2.5,
        axis_max=0.3,
        max_diff=5.0,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    fig = captured["fig"]
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == 4

    for ax in visible_axes:
        assert ax.title.get_fontsize() >= 14
        assert ax.xaxis.label.get_fontsize() >= AXIS_LABEL_FONTSIZE
        assert ax.yaxis.label.get_fontsize() >= AXIS_LABEL_FONTSIZE
        assert ax.get_xticklabels()[0].get_fontsize() >= TICK_LABEL_FONTSIZE
        assert ax.get_yticklabels()[0].get_fontsize() >= TICK_LABEL_FONTSIZE

    assert fig._suptitle.get_fontsize() >= SUPTITLE_FONTSIZE
    for legend in fig.legends:
        assert legend.get_texts()[0].get_fontsize() >= LEGEND_FONTSIZE
        assert legend.get_title().get_fontsize() >= LEGEND_TITLE_FONTSIZE

    _assert_legends_do_not_overlap_axes(fig)


def test_make_figure_legend_layout_for_one_to_three_functionals(tmp_path, monkeypatch):
    for n_funcs in [1, 2, 3]:
        functionals = ["pbe", "pbe_d3", "beef_vdw"][:n_funcs]
        calculators, calc_pairs_per_func, dft_data, ml_data = _build_plot_inputs(functionals)
        output_path = tmp_path / f"parity_{n_funcs}funcs.png"
        captured = _capture_figure(monkeypatch)

        make_figure(
            functionals=functionals,
            calc_pairs_per_func=calc_pairs_per_func,
            dft_data=dft_data,
            ml_data=ml_data,
            calculators=calculators,
            output_path=output_path,
            axis_min=-2.5,
            axis_max=0.3,
            max_diff=5.0,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        fig = captured["fig"]
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == n_funcs
        _assert_legends_do_not_overlap_axes(fig)
