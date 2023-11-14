import unittest

from rt1_model import surface, volume


class TestBasicPlotting(unittest.TestCase):
    def setUp(self):
        # find all names of distribution functions (to make sure all are tested)
        self.SRFnames = [
            key
            for key, val in surface.__dict__.items()
            if (
                isinstance(val, type)
                and issubclass(val, surface._Surface)
                and not key.startswith("_")
                and not key in ["LinComb"]
            )
        ]

        self.Vnames = [
            key
            for key, val in volume.__dict__.items()
            if (
                isinstance(val, type)
                and issubclass(val, volume._Volume)
                and not key.startswith("_")
                and not key in ["LinComb"]
            )
        ]

    def test_surface_init(self):
        a = [0.1, 0.2, 0.3]

        choices = dict(
            Isotropic=dict(),
            CosineLobe=dict(i=3, ncoefs=10, a=a),
            HenyeyGreenstein=dict(t=0.4, ncoefs=10, a=a),
            HG_nadirnorm=dict(t=0.4, ncoefs=10, a=a),
        )

        self.assertTrue(
            all(i in choices for i in self.SRFnames),
            f"Surface functions {set(self.SRFnames).difference(choices)} are not tested!",
        )

        # Check SRF initialization
        for name, params in choices.items():
            SRF = getattr(surface, name)(**params)

            for key, val in params.items():
                self.assertTrue(
                    getattr(SRF, key) == val,
                    f"Parameter {key} incorrecty assigned for surface.{name}!",
                )

            SRF.calc(0.1, 0.2, 0.3, 0.4)
            SRF.legexpansion(0.1, 0.2, 0.3, 0.4)
            SRF._func

    def test_volume_init(self):
        a = [-0.5, 0.6, 0.4]

        choices = dict(
            Isotropic=dict(),
            Rayleigh=dict(a=a),
            HenyeyGreenstein=dict(t=0.4, ncoefs=10, a=a),
            HGRayleigh=dict(t=0.4, ncoefs=10, a=a),
        )

        self.assertTrue(
            all(i in choices for i in self.Vnames),
            f"Volume functions {set(self.Vnames).difference(choices)} are not tested!",
        )

        # Check SRF initialization
        for name, params in choices.items():
            V = getattr(volume, name)(**params)

            for key, val in params.items():
                self.assertTrue(
                    getattr(V, key) == val,
                    f"Parameter {key} incorrecty assigned for volume.{name}!",
                )

            # evaluate function numerical
            V.calc(0.1, 0.2, 0.3, 0.4)
            V.legexpansion(0.1, 0.2, 0.3, 0.4)
            V._func

    def test_linear_combinations_SRF(self):
        a = [0.1, 0.2, 0.3]

        choices = dict(
            Isotropic=dict(),
            CosineLobe=dict(i=3, ncoefs=10, a=a),
            HenyeyGreenstein=dict(t=0.4, ncoefs=10, a=a),
            HG_nadirnorm=dict(t=0.4, ncoefs=10, a=a),
        )

        choices = [
            (1 / len(choices), getattr(surface, name)(**kwargs))
            for name, kwargs in choices.items()
        ]

        SRF = surface.LinComb(choices)

        SRF.calc(0.1, 0.2, 0.3, 0.4)
        SRF.legexpansion(0.1, 0.2, 0.3, 0.4)
        SRF._func

    def test_linear_combinations_V(self):
        a = [0.1, 0.2, 0.3]

        choices = dict(
            Isotropic=dict(),
            Rayleigh=dict(a=a),
            HenyeyGreenstein=dict(t=0.4, ncoefs=10, a=a),
            HGRayleigh=dict(t=0.3, ncoefs=10, a=a),
        )

        choices = [
            (1 / len(choices), getattr(volume, name)(**kwargs))
            for name, kwargs in choices.items()
        ]

        V = volume.LinComb(choices)

        V.calc(0.1, 0.2, 0.3, 0.4)
        V.legexpansion(0.1, 0.2, 0.3, 0.4)
        V._func


if __name__ == "__main__":
    unittest.main()
