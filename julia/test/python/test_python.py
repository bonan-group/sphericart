import subprocess
from pathlib import Path

import numpy as np

import sphericart


L_MAX = 25
N_SAMPLES = 100
JULIA_PROJECT = Path(__file__).parents[2]


def _julia_results(xyz, basis, tmp_path):
    input_path = tmp_path / "coordinates.txt"
    output_path = tmp_path / "results.txt"
    np.savetxt(input_path, xyz)
    code = f"""
using DelimitedFiles
using SpheriCart
using StaticArrays

xyz = readdlm(raw"{input_path}", Float64)
xyz_svectors = [SVector{{3}}(x) for x in eachrow(xyz)]
basis = SpheriCart.{basis}({L_MAX})
results = SpheriCart.compute(basis, xyz_svectors)
writedlm(raw"{output_path}", results)
"""
    subprocess.run(
        ["julia", "--project=.", "-e", code],
        cwd=JULIA_PROJECT,
        check=True,
    )
    return np.loadtxt(output_path)


def test_consistency_solid_harmonics(tmp_path):
    xyz = np.random.default_rng(0).random((N_SAMPLES, 3))
    julia_results = _julia_results(xyz, "SolidHarmonics", tmp_path)

    python_results = sphericart.SolidHarmonics(L_MAX).compute(xyz)
    assert np.allclose(julia_results, python_results)


def test_consistency_spherical_harmonics(tmp_path):
    xyz = np.random.default_rng(1).random((N_SAMPLES, 3))
    julia_results = _julia_results(xyz, "SphericalHarmonics", tmp_path)

    python_results = sphericart.SphericalHarmonics(L_MAX).compute(xyz)
    assert np.allclose(julia_results, python_results)
