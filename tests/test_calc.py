import unittest
import numpy as np
import cloudpickle

from rt1_model import RT1, surface, volume


class TestRT1(unittest.TestCase):
    def setUp(self):
        self.geom = dict(
            t_0=np.deg2rad(60.0),
            t_ex=np.deg2rad(60.0),
            p_0=0.0,
            p_ex=0.0,
            geometry="mono",
        )

        self.V = volume.Rayleigh()
        self.S = surface.Isotropic()

    def test_init(self):
        R = RT1(V=self.V, SRF=self.S)
        R.set_geometry(**self.geom)

        self.assertTrue(R.geometry == self.geom["geometry"])
        self.assertTrue(R.t_0 == self.geom["t_0"])
        self.assertTrue(R.t_ex == R.t_0)
        self.assertTrue(R.p_0 == self.geom["p_0"])
        self.assertTrue(R.p_ex == np.pi)

    def test_calc(self):
        # just try to get it running simply without further testing
        R = RT1(V=self.V, SRF=self.S, dB=False)
        R.set_geometry(**self.geom)
        R.update_params(tau=0.7, omega=0.3, NormBRDF=0.3)

        Itot, Isurf, Ivol, Iint = R.calc()
        self.assertTrue(np.allclose(Itot, Isurf + Ivol + Iint))

        # check values for sig0 = False
        R = RT1(V=self.V, SRF=self.S, dB=False, sig0=False)
        R.set_geometry(**self.geom)
        R.update_params(tau=0.7, omega=0.3, NormBRDF=0.3)

        Itot, Isurf, Ivol, Iint = R.calc()
        self.assertTrue(np.allclose(Itot, Isurf + Ivol + Iint))

        # check values in dB
        R = RT1(V=self.V, SRF=self.S, dB=True)
        R.set_geometry(**self.geom)
        R.update_params(tau=0.7, omega=0.3, NormBRDF=0.3)

        Itot, Isurf, Ivol, Iint = R.calc()
        self.assertTrue(
            np.allclose(
                Itot,
                10
                * np.log10(10 ** (Isurf / 10) + 10 ** (Ivol / 10) + 10 ** (Iint / 10)),
            )
        )

        # check values in dB for sig0 = False
        R = RT1(V=self.V, SRF=self.S, dB=True, sig0=False)
        R.set_geometry(**self.geom)
        R.update_params(tau=0.7, omega=0.3, NormBRDF=0.3)

        Itot, Isurf, Ivol, Iint = R.calc()
        self.assertTrue(
            np.allclose(
                Itot,
                10
                * np.log10(10 ** (Isurf / 10) + 10 ** (Ivol / 10) + 10 ** (Iint / 10)),
            )
        )

        # test results for tau=0 / omega=0
        V = volume.Rayleigh()
        R = RT1(V=V, SRF=self.S, dB=False)
        R.set_geometry(**self.geom)
        R.update_params(tau=0.0, omega=0.0, NormBRDF=0.3)

        Itot, Isurf, Ivol, Iint = R.calc()
        self.assertEqual(Ivol, 0.0)
        self.assertEqual(Iint, 0.0)
        self.assertEqual(Itot, Isurf)
        self.assertTrue(Isurf > 0.0)

    def test_pickle(self):
        SRF = surface.HenyeyGreenstein(t="t_s", ncoefs=8)
        V = volume.HGRayleigh(t="t_v", ncoefs=8)

        R = RT1(V=V, SRF=SRF)
        R.set_geometry(t_0=0.1, p_0=0.2, geometry="mono")
        R.calc(omega=0.3, tau=0.1, NormBRDF=0.3, t_s=0.3, t_v=0.4)

        dump = cloudpickle.dumps(R)

        load = cloudpickle.loads(dump)
        self.assertTrue(np.allclose(load.calc(), R.calc()))


if __name__ == "__main__":
    unittest.main()
